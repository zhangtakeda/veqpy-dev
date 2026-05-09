"""
Module: base.reactive

Role:
- Provide reactive caching and dependency invalidation.
- Build cache invalidation topology from property dependencies.

Public API:
- Reactive
- depends_on

Notes:
- Dependencies are inferred from the property AST by default.
- Dependencies that cannot be inferred statically can be declared with ``depends_on``.
"""

import ast
import inspect
import textwrap
from collections import defaultdict, deque
from copy import deepcopy
from typing import Any, Dict, Set

import numpy as np

# -----------------------------------------------------------------------------
# Public interface
# -----------------------------------------------------------------------------


def depends_on(*deps: str):
    """Declare additional dependencies for a property."""

    def decorator(func):
        setattr(func, "_reactive_deps", set(deps))
        return func

    return decorator


class Reactive:
    """Base class for reactive caching."""

    dependency_graph: Dict[str, Set[str]]
    downstream_map: Dict[str, Set[str]]
    root_properties: Set[str]

    def __init__(self):
        self.cache: dict[str, Any] = {}

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)

        roots = cls._validate_root_properties()
        dependency_graph = cls._build_dependency_graph(roots)
        reverse_adj = _build_reverse_adj(dependency_graph)
        downstream_map = _build_downstream_map(roots, dependency_graph, reverse_adj)
        _validate_dependency_graph(roots, dependency_graph, reverse_adj)

        cls.dependency_graph = dependency_graph
        cls.downstream_map = downstream_map
        cls._setup_root_properties(roots, downstream_map)
        cls._wrap_derived_properties(dependency_graph)

    def __deepcopy__(self, memo):
        cls = self.__class__
        cloned = cls.__new__(cls)
        memo[id(self)] = cloned

        for k, v in self.__dict__.items():
            if k == "cache":
                setattr(cloned, k, {})
            else:
                setattr(cloned, k, deepcopy(v, memo))

        return cloned

    def invalidate(self, *names: str):
        """Invalidate the named properties and their downstream cached values."""
        if not names:
            self.cache.clear()
            return

        cache = self.cache
        downstream = self.downstream_map
        for name in names:
            cache.pop(name, None)
            for dep in downstream.get(name, set()):
                cache.pop(dep, None)

    @classmethod
    def _validate_root_properties(cls) -> Set[str]:
        roots = getattr(cls, "root_properties", None)
        if roots is None:
            raise TypeError(f"{cls.__name__} must define root_properties explicitly")
        if not isinstance(roots, set) or any(not isinstance(name, str) for name in roots):
            raise TypeError(f"{cls.__name__}.root_properties must be a set[str]")
        if not roots:
            raise ValueError(f"{cls.__name__}.root_properties must not be empty")
        return roots

    @classmethod
    def _setup_root_properties(
        cls,
        roots: Set[str],
        downstream_map: Dict[str, Set[str]],
    ) -> None:
        """Create cache-invalidating properties for root attributes."""
        for name in roots:
            if isinstance(getattr(cls, name, None), property):
                continue

            deps = downstream_map.get(name, set())
            cached_name = f"cached_{name}"

            def make_fget(cn):
                def fget(self):
                    v = getattr(self, cn, None)
                    if isinstance(v, np.ndarray) and v.flags.writeable:
                        v.flags.writeable = False
                    return v

                return fget

            def make_fset(n, to_clear):
                def fset(self, value):
                    object.__setattr__(self, f"cached_{n}", value)
                    cache = self.cache
                    for key in to_clear:
                        cache.pop(key, None)

                return fset

            root_prop = property(
                fget=make_fget(cached_name),
                fset=make_fset(name, deps),
            )
            setattr(cls, name, root_prop)

    @classmethod
    def _wrap_derived_properties(
        cls,
        dependency_graph: Dict[str, Set[str]],
    ) -> None:
        """Wrap derived properties with lazy cache lookups."""
        for name in dependency_graph:
            attr = cls._find_property(name)
            if attr is None or attr.fget is None:
                continue

            original_fget = _unwrap_function(attr.fget)

            def make_lazy_fget(orig, n):
                def lazy_fget(self):
                    cache = self.cache
                    if n in cache:
                        return cache[n]
                    value = orig(self)
                    cache[n] = value
                    return value

                lazy_fget.__wrapped__ = orig
                return lazy_fget

            lazy_prop = property(
                fget=make_lazy_fget(original_fget, name),
                fset=attr.fset,
                fdel=attr.fdel,
                doc=attr.__doc__,
            )
            setattr(cls, name, lazy_prop)

    @classmethod
    def _build_dependency_graph(cls, roots: Set[str]) -> Dict[str, Set[str]]:
        """Build the dependency graph for all reactive properties."""
        valid_nodes = set(roots)
        props: Dict[str, property] = {}
        base_props = set(Reactive.__dict__.keys())

        for name in dir(cls):
            if name.startswith("__") or name in base_props:
                continue
            attr = cls._find_property(name)
            if attr is None:
                continue
            valid_nodes.add(name)
            if name not in roots:
                props[name] = attr

        graph: Dict[str, Set[str]] = {}
        for name, prop in props.items():
            if prop.fget is None:
                continue

            original_func = _unwrap_function(prop.fget)
            raw_deps = _parse_dependency(original_func)
            explicit = getattr(original_func, "_reactive_deps", None)
            if explicit is not None:
                raw_deps = raw_deps | explicit

            graph[name] = {d for d in raw_deps if d in valid_nodes and d != name}

        return graph

    @classmethod
    def _find_property(cls, name: str) -> property | None:
        for klass in cls.__mro__:
            if name in klass.__dict__:
                obj = klass.__dict__[name]
                if isinstance(obj, property):
                    return obj
                return None
        return None


# -----------------------------------------------------------------------------
# Private implementation
# -----------------------------------------------------------------------------


def _build_reverse_adj(dependency_graph: Dict[str, Set[str]]) -> Dict[str, Set[str]]:
    """Build a one-hop reverse adjacency map: property -> direct dependents."""
    rev: Dict[str, Set[str]] = defaultdict(set)
    for prop_name, deps in dependency_graph.items():
        for dep in deps:
            rev[dep].add(prop_name)
    return dict(rev)


def _build_downstream_map(
    roots: Set[str],
    dependency_graph: Dict[str, Set[str]],
    reverse_adj: Dict[str, Set[str]],
) -> Dict[str, Set[str]]:
    """Build the transitive reverse closure: property -> all downstream properties."""
    all_nodes = set(roots) | set(dependency_graph.keys())
    result: Dict[str, Set[str]] = {}

    for node in all_nodes:
        visited: Set[str] = set()
        queue = deque(reverse_adj.get(node, set()))
        while queue:
            n = queue.popleft()
            if n in visited:
                continue
            visited.add(n)
            for child in reverse_adj.get(n, set()):
                if child not in visited:
                    queue.append(child)
        if visited:
            result[node] = visited

    return result


def _validate_dependency_graph(
    roots: Set[str],
    dependency_graph: Dict[str, Set[str]],
    reverse_adj: Dict[str, Set[str]],
) -> None:
    """Validate that the dependency graph is acyclic."""
    all_props = set(roots) | set(dependency_graph.keys())
    if not all_props:
        return

    in_degree: Dict[str, int] = {p: 0 for p in all_props}

    for prop, deps in dependency_graph.items():
        in_degree[prop] = len(deps)

    for root in roots:
        in_degree.setdefault(root, 0)

    queue = sorted([p for p, d in in_degree.items() if d == 0])

    processed = 0
    while queue:
        next_queue = []
        for node in queue:
            processed += 1
            for child in reverse_adj.get(node, set()):
                in_degree[child] -= 1
                if in_degree[child] == 0:
                    next_queue.append(child)
        queue = sorted(next_queue)

    if processed != len(all_props):
        remaining = [p for p in all_props if in_degree.get(p, 0) > 0]
        cycles = _detect_cycles(remaining, dependency_graph)
        raise ValueError(
            "Circular dependency detected:\n"
            + f"-- involved properties: {remaining}\n"
            + f"-- dependency cycles: {cycles}"
        )


def _detect_cycles(
    nodes: list[str],
    dependency_graph: Dict[str, Set[str]],
) -> list[list[str]]:
    """Detect and return all cyclic dependency paths."""
    cycles: list[list[str]] = []
    node_set = set(nodes)

    def dfs(node: str, path: list[str], visited: Set[str]):
        if node in path:
            cycle_start = path.index(node)
            cycles.append(path[cycle_start:] + [node])
            return

        if node in visited:
            return

        visited.add(node)
        path.append(node)

        for dep in dependency_graph.get(node, set()):
            if dep in node_set:
                dfs(dep, path.copy(), visited)

    visited_global: Set[str] = set()
    for node in nodes:
        if node not in visited_global:
            dfs(node, [], visited_global)

    return cycles


def _unwrap_function(func):
    if hasattr(func, "__wrapped__"):
        return func.__wrapped__
    return func


def _parse_dependency(func) -> Set[str]:
    """Parse names of ``self.xxx`` attributes accessed by a function."""
    source = inspect.getsource(func)
    source = textwrap.dedent(source)
    tree = ast.parse(source)
    names: Set[str] = set()

    class Visitor(ast.NodeVisitor):
        """AST node visitor."""

        def visit_Attribute(self, node):
            if isinstance(node.value, ast.Name) and node.value.id == "self":
                names.add(node.attr)
            self.generic_visit(node)

    Visitor().visit(tree)
    return names
