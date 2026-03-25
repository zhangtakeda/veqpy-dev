from __future__ import annotations

from contextlib import ExitStack, contextmanager
from dataclasses import dataclass, field
import importlib
from time import perf_counter
from typing import Callable

import numpy as np

geometry_module = importlib.import_module("veqpy.model.geometry")
profile_module = importlib.import_module("veqpy.model.profile")
operator_module = importlib.import_module("veqpy.operator.operator")
from veqpy.model import Grid
from veqpy.operator import Operator, OperatorCase
from veqpy.solver import Solver, SolverConfig


COEFFS = {
    "h": [0.0] * 3,
    "v": None,
    "k": [0.0] * 3,
    "c0": None,
    "c1": None,
    "s1": [0.0] * 3,
    "s2": None,
}


@dataclass(slots=True)
class AttemptTiming:
    method: str
    elapsed_us: float
    success: bool
    scipy_us: float = 0.0
    residual_eval_us: float = 0.0


@dataclass(slots=True)
class ProbeSummary:
    attempt_timings: list[AttemptTiming] = field(default_factory=list)
    current_attempt: AttemptTiming | None = None
    metrics: dict[str, float] = field(default_factory=dict)
    counts: dict[str, int] = field(default_factory=dict)

    def add(self, name: str, elapsed_us: float) -> None:
        self.metrics[name] = self.metrics.get(name, 0.0) + float(elapsed_us)
        self.counts[name] = self.counts.get(name, 0) + 1

    def metric(self, name: str) -> float:
        return float(self.metrics.get(name, 0.0))

    def count(self, name: str) -> int:
        return int(self.counts.get(name, 0))

    def render(self, *, total_elapsed_us: float) -> str:
        lines: list[str] = []
        lines.append(f"total elapsed: {total_elapsed_us / 1000:.3f} ms")
        attempt_total_us = sum(item.elapsed_us for item in self.attempt_timings)
        lines.append(
            f"attempt envelope: {attempt_total_us / 1000:.3f} ms "
            f"({attempt_total_us / max(total_elapsed_us, 1e-12) * 100:.1f}%)"
        )
        for idx, attempt in enumerate(self.attempt_timings, start=1):
            lines.append(
                f"  attempt {idx} [{attempt.method}]: "
                f"{attempt.elapsed_us / 1000:.3f} ms, "
                f"scipy={attempt.scipy_us / 1000:.3f} ms, "
                f"extra_residual={attempt.residual_eval_us / 1000:.3f} ms, "
                f"success={attempt.success}"
            )
        lines.append(
            f"residual calls: {self.count('operator.residual')}, "
            f"residual total={self.metric('operator.residual') / 1000:.3f} ms, "
            f"safe_residual={self.metric('solver._safe_residual_norm') / 1000:.3f} ms"
        )
        lines.append("operator stages:")
        stage_names = (
            "operator.stage_a_profile",
            "operator.stage_b_geometry",
            "operator.stage_c_source",
            "operator.stage_d_residual",
        )
        stage_total_us = sum(self.metric(name) for name in stage_names)
        for name in stage_names:
            elapsed_us = self.metric(name)
            lines.append(
                f"  {name.split('.')[-1]}: {elapsed_us / 1000:.3f} ms "
                f"({elapsed_us / max(total_elapsed_us, 1e-12) * 100:.1f}% of solve, "
                f"{elapsed_us / max(stage_total_us, 1e-12) * 100:.1f}% of staged residual time)"
            )

        stage_a = self.metric("operator.stage_a_profile")
        packed_fill = self.metric("stage_a.fill_active_profile_views_from_packed_bulk")
        fill_rows = self.metric("stage_a.fill_profile_views")
        profile_update = self.metric("stage_a.fill_profile_runtime_view_from_packed")
        profile_kernel = (
            self.metric("stage_a.engine.update_profile_packed")
            + self.metric("stage_a.engine.update_profiles_packed_bulk")
        )
        lines.append("stage_a detail:")
        lines.append(
            f"  fill active views from packed x: {packed_fill / 1000:.3f} ms "
            f"(calls={self.count('stage_a.fill_active_profile_views_from_packed_bulk')})"
        )
        lines.append(
            f"  fill profile views: {fill_rows / 1000:.3f} ms "
            f"(calls={self.count('stage_a.fill_profile_views')})"
        )
        lines.append(
            f"  fill_profile_runtime_view_from_packed total: {profile_update / 1000:.3f} ms "
            f"(calls={self.count('stage_a.fill_profile_runtime_view_from_packed')})"
        )
        lines.append(
            f"  engine.update_profile*_packed: {profile_kernel / 1000:.3f} ms "
            f"(calls={self.count('stage_a.engine.update_profile_packed') + self.count('stage_a.engine.update_profiles_packed_bulk')})"
        )
        lines.append(
            f"  stage_a Python/non-kernel remainder: "
            f"{max(stage_a - profile_kernel, 0.0) / 1000:.3f} ms"
        )

        stage_b = self.metric("operator.stage_b_geometry")
        geometry_update = self.metric("stage_b.geometry_update")
        geometry_kernel = self.metric("stage_b.engine.update_geometry")
        lines.append("stage_b detail:")
        lines.append(
            f"  Geometry.update total: {geometry_update / 1000:.3f} ms "
            f"(calls={self.count('stage_b.geometry_update')})"
        )
        lines.append(
            f"  engine.update_geometry: {geometry_kernel / 1000:.3f} ms "
            f"(calls={self.count('stage_b.engine.update_geometry')})"
        )
        lines.append(
            f"  stage_b Python wrapper remainder: "
            f"{max(stage_b - geometry_kernel, 0.0) / 1000:.3f} ms"
        )

        stage_c = self.metric("operator.stage_c_source")
        source_runner = self.metric("stage_c.source_runner")
        source_engine = self.metric("stage_c.engine.operator_kernel")
        lines.append("stage_c detail:")
        lines.append(
            f"  source_runner total: {source_runner / 1000:.3f} ms "
            f"(calls={self.count('stage_c.source_runner')})"
        )
        lines.append(
            f"  engine source kernel: {source_engine / 1000:.3f} ms "
            f"(calls={self.count('stage_c.engine.operator_kernel')})"
        )
        lines.append(
            f"  stage_c wrapper remainder: "
            f"{max(stage_c - source_engine, 0.0) / 1000:.3f} ms"
        )

        stage_d = self.metric("operator.stage_d_residual")
        build_g = self.metric("stage_d.build_G_inplace")
        update_residual = self.metric("stage_d.engine.update_residual")
        assemble_total = self.metric("stage_d.assemble_residual")
        residual_runner = self.metric("stage_d.residual_runner")
        lines.append("stage_d detail:")
        lines.append(
            f"  build_G total: {build_g / 1000:.3f} ms "
            f"(calls={self.count('stage_d.build_G_inplace')})"
        )
        lines.append(
            f"  engine.update_residual: {update_residual / 1000:.3f} ms "
            f"(calls={self.count('stage_d.engine.update_residual')})"
        )
        lines.append(
            f"  assemble_residual total: {assemble_total / 1000:.3f} ms "
            f"(calls={self.count('stage_d.assemble_residual')})"
        )
        lines.append(
            f"  stage_d Python/non-kernel remainder: "
            f"{max(stage_d - update_residual - residual_runner, 0.0) / 1000:.3f} ms"
        )
        lines.append(
            f"  residual_runner total: {residual_runner / 1000:.3f} ms "
            f"(calls={self.count('stage_d.residual_runner')})"
        )
        overhead_us = total_elapsed_us - attempt_total_us
        lines.append(f"outer overhead: {overhead_us / 1000:.3f} ms")
        return "\n".join(lines)


def pf_reference_profiles(rho: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    beta0 = 0.75
    psin = rho**2
    psin_r = 2.0 * rho

    alpha_p, alpha_f = 5.0, 3.32
    exp_ap, exp_af = np.exp(alpha_p), np.exp(alpha_f)
    den_p, den_f = 1.0 + exp_ap * (alpha_p - 1.0), 1.0 + exp_af * (alpha_f - 1.0)

    current_input = (1.0 - beta0) * alpha_f * (np.exp(alpha_f * psin) - exp_af) / den_f * psin_r
    heat_input = beta0 * alpha_p * (np.exp(alpha_p * psin) - exp_ap) / den_p * psin_r
    return current_input, heat_input


def build_demo_solver() -> Solver:
    grid = Grid(Nr=12, Nt=12, scheme="legendre")
    current_input, heat_input = pf_reference_profiles(grid.rho)
    case = OperatorCase(
        coeffs_by_name=COEFFS,
        a=1.05 / 1.85,
        R0=1.05,
        Z0=0.0,
        B0=3.0,
        ka=2.2,
        s1a=float(np.arcsin(0.5)),
        heat_input=heat_input,
        current_input=current_input,
        Ip=3.0e6,
    )
    operator = Operator(grid=grid, case=case, name="PF", derivative="rho")
    config = SolverConfig(
        method="hybr",
        enable_verbose=False,
        enable_warmstart=False,
        enable_history=False,
    )
    return Solver(operator=operator, config=config)


@contextmanager
def _patch_method(cls, method_name: str, factory: Callable[[Callable], Callable]):
    original = getattr(cls, method_name)
    setattr(cls, method_name, factory(original))
    try:
        yield
    finally:
        setattr(cls, method_name, original)


@contextmanager
def _patch_attr(target, attr_name: str, new_value):
    original = getattr(target, attr_name)
    setattr(target, attr_name, new_value)
    try:
        yield
    finally:
        setattr(target, attr_name, original)


@contextmanager
def install_probe(*, solver: Solver, summary: ProbeSummary):
    target_solver = solver
    target_operator = solver.operator
    target_profiles = (
        target_operator.h_profile,
        target_operator.v_profile,
        target_operator.k_profile,
        target_operator.c0_profile,
        target_operator.c1_profile,
        target_operator.s1_profile,
        target_operator.s2_profile,
        target_operator.psin_profile,
        target_operator.F_profile,
    )
    target_profile_ids = {id(profile) for profile in target_profiles}
    def wrap_attempt(original: Callable) -> Callable:
        def wrapped(self, *args, **kwargs):
            if self is not target_solver:
                return original(self, *args, **kwargs)
            solve_config = kwargs["solve_config"]
            method = self._display_method_label(solve_config)
            attempt = AttemptTiming(method=method, elapsed_us=0.0, success=False)
            previous = summary.current_attempt
            summary.current_attempt = attempt
            started = perf_counter()
            try:
                result, error = original(self, *args, **kwargs)
                attempt.success = error is None and result is not None and bool(result[1])
                return result, error
            finally:
                attempt.elapsed_us = (perf_counter() - started) * 1e6
                summary.attempt_timings.append(attempt)
                summary.current_attempt = previous

        return wrapped

    def wrap_scipy(original: Callable) -> Callable:
        def wrapped(self, *args, **kwargs):
            if self is not target_solver or summary.current_attempt is None:
                return original(self, *args, **kwargs)
            started = perf_counter()
            try:
                return original(self, *args, **kwargs)
            finally:
                summary.current_attempt.scipy_us += (perf_counter() - started) * 1e6

        return wrapped

    def wrap_safe_residual(original: Callable) -> Callable:
        def wrapped(self, *args, **kwargs):
            if self is not target_solver:
                return original(self, *args, **kwargs)
            started = perf_counter()
            try:
                return original(self, *args, **kwargs)
            finally:
                elapsed_us = (perf_counter() - started) * 1e6
                summary.add("solver._safe_residual_norm", elapsed_us)
                if summary.current_attempt is not None:
                    summary.current_attempt.residual_eval_us += elapsed_us

        return wrapped

    def wrap_residual(original: Callable) -> Callable:
        def wrapped(self, *args, **kwargs):
            if self is not target_operator:
                return original(self, *args, **kwargs)
            started = perf_counter()
            try:
                return original(self, *args, **kwargs)
            finally:
                summary.add("operator.residual", (perf_counter() - started) * 1e6)

        return wrapped

    def wrap_stage(stage_name: str) -> Callable[[Callable], Callable]:
        def factory(original: Callable) -> Callable:
            def wrapped(self, *args, **kwargs):
                if self is not target_operator:
                    return original(self, *args, **kwargs)
                started = perf_counter()
                try:
                    return original(self, *args, **kwargs)
                finally:
                    summary.add(stage_name, (perf_counter() - started) * 1e6)

            return wrapped

        return factory

    def wrap_fill_views(original: Callable) -> Callable:
        def wrapped(self, *args, **kwargs):
            if self is not target_operator:
                return original(self, *args, **kwargs)
            started = perf_counter()
            try:
                return original(self, *args, **kwargs)
            finally:
                summary.add("stage_a.fill_profile_views", (perf_counter() - started) * 1e6)

        return wrapped

    def wrap_runtime_view_fill(original: Callable) -> Callable:
        def wrapped(view, *args, **kwargs):
            started = perf_counter()
            try:
                return original(view, *args, **kwargs)
            finally:
                summary.add("stage_a.fill_profile_runtime_view_from_packed", (perf_counter() - started) * 1e6)

        return wrapped

    def wrap_profile_kernel(original: Callable) -> Callable:
        def wrapped(*args, **kwargs):
            started = perf_counter()
            try:
                return original(*args, **kwargs)
            finally:
                summary.add("stage_a.engine.update_profile_packed", (perf_counter() - started) * 1e6)

        return wrapped

    def wrap_bulk_profile_kernel(original: Callable) -> Callable:
        def wrapped(*args, **kwargs):
            started = perf_counter()
            try:
                return original(*args, **kwargs)
            finally:
                summary.add("stage_a.engine.update_profiles_packed_bulk", (perf_counter() - started) * 1e6)

        return wrapped

    def wrap_fill_active_views_from_packed_bulk(original: Callable) -> Callable:
        def wrapped(self, *args, **kwargs):
            if self is not target_operator:
                return original(self, *args, **kwargs)
            started = perf_counter()
            try:
                return original(self, *args, **kwargs)
            finally:
                summary.add("stage_a.fill_active_profile_views_from_packed_bulk", (perf_counter() - started) * 1e6)

        return wrapped

    def wrap_geometry_update(original: Callable) -> Callable:
        def wrapped(self, *args, **kwargs):
            if self is not target_operator.geometry:
                return original(self, *args, **kwargs)
            started = perf_counter()
            try:
                return original(self, *args, **kwargs)
            finally:
                summary.add("stage_b.geometry_update", (perf_counter() - started) * 1e6)

        return wrapped

    def wrap_geometry_kernel(original: Callable) -> Callable:
        def wrapped(*args, **kwargs):
            started = perf_counter()
            try:
                return original(*args, **kwargs)
            finally:
                summary.add("stage_b.engine.update_geometry", (perf_counter() - started) * 1e6)

        return wrapped

    original_source_runner = target_operator.source_runner

    def _unwrap_source_runner(original_runner: Callable) -> tuple[int, Callable] | None:
        freevars = original_runner.__code__.co_freevars
        closure = original_runner.__closure__ or ()
        mapping = {name: cell.cell_contents for name, cell in zip(freevars, closure, strict=True)}
        if "derivative_code" not in mapping or "operator_kernel" not in mapping:
            return None
        return int(mapping["derivative_code"]), mapping["operator_kernel"]

    source_runner_parts = _unwrap_source_runner(original_source_runner)

    def wrap_source_runner(original: Callable) -> Callable:
        if source_runner_parts is None:
            def fallback(*args, **kwargs):
                started = perf_counter()
                try:
                    return original(*args, **kwargs)
                finally:
                    elapsed_us = (perf_counter() - started) * 1e6
                    summary.add("stage_c.source_runner", elapsed_us)
                    summary.add("stage_c.engine.operator_kernel", elapsed_us)

            return fallback

        derivative_code, operator_kernel = source_runner_parts

        def wrapped(
            out_psin_r,
            out_psin_rr,
            out_FFn_r,
            out_Pn_r,
            heat_input,
            current_input,
            R0,
            B0,
            weights,
            differentiation_matrix,
            integration_matrix,
            rho,
            V_r,
            Kn,
            Kn_r,
            Ln_r,
            S_r,
            R,
            JdivR,
            F,
            Ip,
            beta,
        ):
            started = perf_counter()
            try:
                kernel_started = perf_counter()
                result = operator_kernel(
                    out_psin_r,
                    out_psin_rr,
                    out_FFn_r,
                    out_Pn_r,
                    heat_input,
                    current_input,
                    derivative_code,
                    R0,
                    B0,
                    weights,
                    differentiation_matrix,
                    integration_matrix,
                    rho,
                    V_r,
                    Kn,
                    Kn_r,
                    Ln_r,
                    S_r,
                    R,
                    JdivR,
                    F,
                    Ip,
                    beta,
                )
                summary.add("stage_c.engine.operator_kernel", (perf_counter() - kernel_started) * 1e6)
                return result
            finally:
                summary.add("stage_c.source_runner", (perf_counter() - started) * 1e6)

        return wrapped

    def wrap_build_g(original: Callable) -> Callable:
        def wrapped(self, *args, **kwargs):
            if self is not target_operator:
                return original(self, *args, **kwargs)
            started = perf_counter()
            try:
                return original(self, *args, **kwargs)
            finally:
                summary.add("stage_d.build_G_inplace", (perf_counter() - started) * 1e6)

        return wrapped

    def wrap_update_residual_kernel(original: Callable) -> Callable:
        def wrapped(*args, **kwargs):
            started = perf_counter()
            try:
                return original(*args, **kwargs)
            finally:
                summary.add("stage_d.engine.update_residual", (perf_counter() - started) * 1e6)

        return wrapped

    def wrap_assemble_residual(original: Callable) -> Callable:
        def wrapped(self, *args, **kwargs):
            if self is not target_operator:
                return original(self, *args, **kwargs)
            started = perf_counter()
            try:
                return original(self, *args, **kwargs)
            finally:
                summary.add("stage_d.assemble_residual", (perf_counter() - started) * 1e6)

        return wrapped

    def wrap_residual_runner(original: Callable) -> Callable:
        def wrapped(*args, **kwargs):
            started = perf_counter()
            try:
                return original(*args, **kwargs)
            finally:
                summary.add("stage_d.residual_runner", (perf_counter() - started) * 1e6)

        return wrapped

    with ExitStack() as stack:
        stack.enter_context(_patch_method(Solver, "_try_solve_attempt", wrap_attempt))
        stack.enter_context(_patch_method(Solver, "_run_root_full", wrap_scipy))
        stack.enter_context(_patch_method(Solver, "_run_root_masked", wrap_scipy))
        stack.enter_context(_patch_method(Solver, "_run_least_squares_full", wrap_scipy))
        stack.enter_context(_patch_method(Solver, "_run_least_squares_masked", wrap_scipy))
        stack.enter_context(_patch_method(Solver, "_safe_residual_norm", wrap_safe_residual))
        stack.enter_context(_patch_method(Operator, "residual", wrap_residual))
        stack.enter_context(_patch_method(Operator, "stage_a_profile", wrap_stage("operator.stage_a_profile")))
        stack.enter_context(_patch_method(Operator, "stage_b_geometry", wrap_stage("operator.stage_b_geometry")))
        stack.enter_context(_patch_method(Operator, "stage_c_source", wrap_stage("operator.stage_c_source")))
        stack.enter_context(_patch_method(Operator, "stage_d_residual", wrap_stage("operator.stage_d_residual")))
        fill_method_name = "_fill_profile_views"
        stack.enter_context(_patch_method(Operator, fill_method_name, wrap_fill_views))
        stack.enter_context(
            _patch_method(
                Operator,
                "_fill_active_profile_views_from_packed_bulk",
                wrap_fill_active_views_from_packed_bulk,
            )
        )
        runtime_fill = getattr(operator_module, "fill_profile_runtime_view_from_packed", None)
        stack.enter_context(
            _patch_attr(
                operator_module,
                "fill_profile_runtime_view_from_packed",
                wrap_runtime_view_fill(runtime_fill),
            )
        )
        stack.enter_context(
            _patch_attr(profile_module, "update_profile_packed", wrap_profile_kernel(profile_module.update_profile_packed))
        )
        stack.enter_context(
            _patch_attr(
                operator_module,
                "update_profiles_packed_bulk",
                wrap_bulk_profile_kernel(operator_module.update_profiles_packed_bulk),
            )
        )
        stack.enter_context(_patch_method(geometry_module.Geometry, "update", wrap_geometry_update))
        stack.enter_context(_patch_attr(geometry_module, "update_geometry", wrap_geometry_kernel(geometry_module.update_geometry)))
        stack.enter_context(_patch_attr(target_operator, "source_runner", wrap_source_runner(original_source_runner)))
        stack.enter_context(_patch_method(Operator, "_build_G_inplace", wrap_build_g))
        stack.enter_context(_patch_attr(operator_module, "update_residual", wrap_update_residual_kernel(operator_module.update_residual)))
        stack.enter_context(_patch_method(Operator, "_assemble_residual", wrap_assemble_residual))
        stack.enter_context(
            _patch_attr(target_operator, "residual_runner", wrap_residual_runner(target_operator.residual_runner))
        )
        yield


def run_profiled_single_solve(*, warmup: bool = True) -> tuple[Solver, ProbeSummary, float | None]:
    solver = build_demo_solver()
    warmup_elapsed_us: float | None = None
    if warmup:
        solver.solve()
        warmup_elapsed_us = float(solver.result.elapsed)
        solver.reset()
    summary = ProbeSummary()
    with install_probe(solver=solver, summary=summary):
        solver.solve(enable_warmstart=False)
    return solver, summary, warmup_elapsed_us


if __name__ == "__main__":
    solver, summary, warmup_elapsed_us = run_profiled_single_solve()
    if warmup_elapsed_us is not None:
        print(f"warm-up solve elapsed: {warmup_elapsed_us / 1000:.3f} ms")
    print(summary.render(total_elapsed_us=solver.result.elapsed))
