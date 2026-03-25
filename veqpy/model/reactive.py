"""
Module: model.reactive

Role:
- 负责提供响应式缓存与依赖失效框架.
- 负责根据 property 依赖关系构建缓存拓扑.

Public API:
- Reactive
- depends_on

Notes:
- 依赖默认通过 AST 推断.
- 无法静态分析的依赖可以用 `depends_on` 显式声明.
"""

import ast
import inspect
import textwrap
from collections import defaultdict, deque
from copy import deepcopy
from typing import Any, Dict, List, Set

import numpy as np


def _is_none(node: ast.expr) -> bool:
    """检查 AST 节点是否为 None 字面量."""
    return isinstance(node, ast.Constant) and node.value is None


def _parse_dependency(func) -> Set[str]:
    """分析函数中访问的 self.xxx 属性名."""
    source = inspect.getsource(func)
    source = textwrap.dedent(source)
    tree = ast.parse(source)
    names: Set[str] = set()

    class Visitor(ast.NodeVisitor):
        """AST 节点访问器."""

        def visit_Attribute(self, node):
            if isinstance(node.value, ast.Name) and node.value.id == "self":
                names.add(node.attr)
            self.generic_visit(node)

    Visitor().visit(tree)
    return names


def depends_on(*deps: str):
    """显式声明 property 的额外依赖."""

    def decorator(func):
        setattr(func, "_reactive_deps", set(deps))
        return func

    return decorator


class Reactive:
    """响应式缓存基类."""

    dependency_graph: Dict[str, Set[str]]
    downstream_map: Dict[str, Set[str]]
    property_dag: List[List[str]]
    _reverse_adj: Dict[str, Set[str]]

    root_properties: Set[str]

    def __init__(self):
        self.cache: dict[str, Any] = {}

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)

        roots = getattr(cls, "root_properties", None)
        if roots is None or (callable(roots) and not isinstance(roots, set)):
            inferred = cls._infer_root_properties()
            if not inferred:
                return
            cls.root_properties = inferred

        cls.dependency_graph = cls._build_dependency_graph()
        cls._reverse_adj = cls._build_reverse_adj()
        cls.downstream_map = cls._build_downstream_map()
        cls.property_dag = cls._build_topological_sort()

        cls._setup_root_properties()
        cls._wrap_derived_properties()

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
        """使指定属性及其下游缓存失效."""
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
    def _setup_root_properties(cls):
        """为根属性创建带缓存清除的 property."""

        for name in cls.root_properties:
            if isinstance(getattr(cls, name, None), property):
                continue

            deps = cls.downstream_map.get(name, set())
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
    def _wrap_derived_properties(cls):
        """将派生属性包装成惰性缓存版本."""

        for name in cls.dependency_graph:
            attr = None
            for klass in cls.__mro__:
                if name in klass.__dict__:
                    attr = klass.__dict__[name]
                    break
            if attr is None or not isinstance(attr, property) or attr.fget is None:
                continue

            original_fget = attr.fget
            if hasattr(original_fget, "__wrapped__"):
                original_fget = original_fget.__wrapped__

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
    def _build_dependency_graph(cls) -> Dict[str, Set[str]]:
        """自动构建所有属性的依赖图"""

        valid_nodes = set(cls.root_properties)
        props: Dict[str, property] = {}
        base_props = set(Reactive.__dict__.keys())

        for name in dir(cls):
            if name.startswith("__") or name in base_props:
                continue
            for klass in cls.__mro__:
                if name in klass.__dict__:
                    obj = klass.__dict__[name]
                    if isinstance(obj, property):
                        valid_nodes.add(name)
                        if name not in cls.root_properties:
                            props[name] = obj
                    break

        graph: Dict[str, Set[str]] = {}
        for name, prop in props.items():
            if prop.fget is None:
                continue

            original_func = prop.fget
            if hasattr(original_func, "__wrapped__"):
                original_func = original_func.__wrapped__

            raw_deps = _parse_dependency(original_func)
            explicit = getattr(original_func, "_reactive_deps", None)
            if explicit is not None:
                raw_deps = raw_deps | explicit

            graph[name] = {d for d in raw_deps if d in valid_nodes and d != name}

        return graph

    @classmethod
    def _build_reverse_adj(cls) -> Dict[str, Set[str]]:
        """构建单层反向邻接表: 属性 -> 直接依赖它的属性"""
        rev: Dict[str, Set[str]] = defaultdict(set)
        for prop_name, deps in cls.dependency_graph.items():
            for dep in deps:
                rev[dep].add(prop_name)
        return dict(rev)

    @classmethod
    def _build_downstream_map(cls) -> Dict[str, Set[str]]:
        """构建全属性的反向传递 (闭包): 属性 -> 所有 传递依赖它的派生属性"""
        rev = cls._reverse_adj
        all_nodes = set(cls.root_properties) | set(cls.dependency_graph.keys())
        result: Dict[str, Set[str]] = {}

        for node in all_nodes:
            visited: Set[str] = set()
            queue = deque(rev.get(node, set()))
            while queue:
                n = queue.popleft()
                if n in visited:
                    continue
                visited.add(n)
                for child in rev.get(n, set()):
                    if child not in visited:
                        queue.append(child)
            if visited:
                result[node] = visited

        return result

    @classmethod
    def _build_topological_sort(cls) -> List[List[str]]:
        """构建所有属性的分层拓扑排序 (Kahn, O(V+E))"""

        all_props = set(cls.root_properties) | set(cls.dependency_graph.keys())
        if not all_props:
            return []

        rev = cls._reverse_adj
        in_degree: Dict[str, int] = {p: 0 for p in all_props}

        for prop, deps in cls.dependency_graph.items():
            in_degree[prop] = len(deps)

        for root in cls.root_properties:
            in_degree.setdefault(root, 0)

        result: List[List[str]] = []
        queue = sorted([p for p, d in in_degree.items() if d == 0])

        processed = 0
        while queue:
            result.append(queue)
            next_queue = []
            for node in queue:
                processed += 1
                for child in rev.get(node, set()):
                    in_degree[child] -= 1
                    if in_degree[child] == 0:
                        next_queue.append(child)
            queue = sorted(next_queue)

        if processed != len(all_props):
            remaining = [p for p in all_props if in_degree.get(p, 0) > 0]
            cycles = cls._detect_cycles(remaining)
            raise ValueError(
                "Circular dependency detected:\n"
                + f"-- involved properties: {remaining}\n"
                + f"-- dependency cycles: {cycles}"
            )

        return result

    @classmethod
    def _detect_cycles(cls, nodes: List[str]) -> List[List[str]]:
        """检测并返回所有循环依赖路径"""
        cycles: List[List[str]] = []
        node_set = set(nodes)

        def dfs(node: str, path: List[str], visited: Set[str]):
            if node in path:
                cycle_start = path.index(node)
                cycles.append(path[cycle_start:] + [node])
                return

            if node in visited:
                return

            visited.add(node)
            path.append(node)

            for dep in cls.dependency_graph.get(node, set()):
                if dep in node_set:
                    dfs(dep, path.copy(), visited)

        visited_global: Set[str] = set()
        for node in nodes:
            if node not in visited_global:
                dfs(node, [], visited_global)

        return cycles

    @classmethod
    def _infer_root_properties(cls) -> Set[str]:
        """从子类 __init__ 自动推断根属性

        规则: __init__ 中形如 self.X = Y 的赋值, 其中 Y 不是 None
        """
        init = cls.__dict__.get("__init__")
        if init is None:
            return set()

        try:
            source = inspect.getsource(init)
            source = textwrap.dedent(source)
            tree = ast.parse(source)
        except OSError:
            return set()

        roots: Set[str] = set()
        func_def = tree.body[0]
        if not isinstance(func_def, (ast.FunctionDef, ast.AsyncFunctionDef)):
            return set()

        for node in ast.walk(func_def):
            if (
                isinstance(node, ast.Assign)
                and len(node.targets) == 1
                and isinstance(node.targets[0], ast.Attribute)
                and isinstance(node.targets[0].value, ast.Name)
                and node.targets[0].value.id == "self"
                and not _is_none(node.value)
            ):
                roots.add(node.targets[0].attr)

        return roots
