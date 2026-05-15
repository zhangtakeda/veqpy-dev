"""
Module: solver.solver

Role:
- 负责执行 nonlinear solve 生命周期.
- 负责管理 x0, history 与 SolverResult 封装.

Public API:
- Solver

Notes:
- `Solver` 是 solver 层 facade.
- 不负责 packed layout/codec, backend 选择, 或 Stage A/B/C/D 数值核实现.
"""

from __future__ import annotations

import warnings
from dataclasses import replace
from time import perf_counter
from typing import Callable

import numpy as np
from rich.console import Console

from veqpy.model.equilibrium import Equilibrium
from veqpy.operator.operator import Operator
from veqpy.operator.operator_case import OperatorCase
from veqpy.solver.solver_config import (
    LEAST_SQUARES_METHODS,
    ROOT_METHODS,
    SUPPORTED_METHODS,
    OptimizeMethod,
    SolverConfig,
)
from veqpy.solver.solver_record import SolverRecord
from veqpy.solver.solver_result import SolverResult


class Solver:
    """固定 packed layout 的求解 facade."""

    def __init__(
        self,
        *,
        operator: Operator,
        config: SolverConfig | None = None,
    ) -> None:
        """绑定一个 Operator 和一份默认求解配置."""

        self.operator = operator
        self.config = SolverConfig() if config is None else config
        self.result: SolverResult | None = None
        self.history: list[SolverRecord] = []
        self.x0 = self.operator.encode_initial_state()

    def reset(self) -> None:
        """将 solver 持有的 x0 原地清零."""

        self.x0.fill(0.0)

    def clear(self) -> None:
        """清空 solve history, 不改当前 x0."""

        self.history.clear()

    def replace_case(self, case: OperatorCase) -> None:
        """替换兼容工况."""

        self.operator.replace_case(case)

    def solve(
        self,
        x0: np.ndarray | None = None,
        *,
        method: str | None = None,
        max_residual: float | None = None,
        max_evaluations: int | None = None,
        enable_warmstart: bool | None = None,
        initial_policy: str | None = None,
        initial_homothetic_lambda: float | None = None,
        enable_fallback: bool | None = None,
        fallback_methods: tuple[str, ...] | list[str] | None = None,
        enable_verbose: bool | None = None,
        enable_history: bool | None = None,
        residual_normalization: str | None = None,
        residual_normalization_floor: float | None = None,
        residual_normalization_max_ratio: float | None = None,
        residual_normalization_huber_tau: float | None = None,
        residual_normalization_probe_count: int | None = None,
        residual_normalization_probe_step: float | None = None,
        residual_normalization_sensitivity_lambda: float | None = None,
        enable_collocation: bool | None = None,
        collocation_method: str | None = None,
        collocation_max_residual: float | None = None,
        collocation_max_evaluations: int | None = None,
    ) -> np.ndarray:
        """执行一次求解并返回收敛后的 packed x."""

        solve_config = self._resolve_solve_config(
            method=method,
            max_residual=max_residual,
            max_evaluations=max_evaluations,
            enable_warmstart=enable_warmstart,
            initial_policy=initial_policy,
            initial_homothetic_lambda=initial_homothetic_lambda,
            enable_fallback=enable_fallback,
            fallback_methods=fallback_methods,
            enable_verbose=enable_verbose,
            enable_history=enable_history,
            residual_normalization=residual_normalization,
            residual_normalization_floor=residual_normalization_floor,
            residual_normalization_max_ratio=residual_normalization_max_ratio,
            residual_normalization_huber_tau=residual_normalization_huber_tau,
            residual_normalization_probe_count=residual_normalization_probe_count,
            residual_normalization_probe_step=residual_normalization_probe_step,
            residual_normalization_sensitivity_lambda=residual_normalization_sensitivity_lambda,
            enable_collocation=enable_collocation,
            collocation_method=collocation_method,
            collocation_max_residual=collocation_max_residual,
            collocation_max_evaluations=collocation_max_evaluations,
        )
        _validate_stage_solve_config(solve_config, residual_kind="variational")

        if x0 is not None:
            self.x0 = self.operator.coerce_x(x0).copy()
        elif solve_config.initial_policy == "warm":
            self.x0 = self.x0.copy()
        else:
            self.x0 = _build_initial_state(self.operator, solve_config).copy()
        if x0 is not None or solve_config.initial_policy != "warm":
            self.operator.invalidate_source_state()

        x_guess = self.x0.copy()

        started = perf_counter()
        if solve_config.enable_collocation:
            (
                x_opt,
                success,
                message,
                function_evaluations,
                jacobian_evaluations,
                iterations,
                residual_norm_final,
            ) = self._solve_with_collocation_polish(
                x_guess,
                solve_config=solve_config,
                x0_was_provided=x0 is not None,
            )
        else:
            (
                x_opt,
                success,
                message,
                function_evaluations,
                jacobian_evaluations,
                iterations,
                residual_norm_final,
            ) = self._solve_with_fallbacks(
                x_guess,
                solve_config=solve_config,
                residual_kind="variational",
                x0_was_provided=x0 is not None,
            )
        elapsed = (perf_counter() - started) * 1e6

        x_final = self.operator.coerce_x(x_opt)
        residual_final_exc = None
        if not bool(success) and not np.isfinite(residual_norm_final):
            residual_norm_final, residual_final_exc = self._safe_residual_norm(
                x_final,
                solve_config=self._final_residual_config(solve_config),
                residual_kind=self._final_residual_kind(solve_config),
            )
        if residual_final_exc is not None:
            success = False
            message = (
                f"{message} [final residual evaluation failed: "
                f"{type(residual_final_exc).__name__}: {residual_final_exc}]"
            )

        self.result = SolverResult(
            x0=x_guess,
            x=x_final,
            success=bool(success),
            message=str(message),
            residual_norm_final=residual_norm_final,
            function_evaluations=int(function_evaluations),
            jacobian_evaluations=int(jacobian_evaluations),
            iterations=int(iterations),
            elapsed=elapsed,
        )

        record = SolverRecord(
            case_snapshot=self.operator.case.copy(),
            config_snapshot=solve_config,
            result_snapshot=self.result,
        )

        if solve_config.enable_verbose:
            Console().print(record)

        if solve_config.enable_history:
            self.history.append(record)

        self.x0 = x_final.copy()
        return x_final

    def build_coeffs(
        self,
        *,
        include_none: bool = True,
    ) -> dict[str, list[float] | None]:
        """从当前 solver 持有的 x0 重建 profile 系数字典."""

        return self.operator.build_coeffs(self.x0, include_none=include_none)

    def build_equilibrium(self) -> Equilibrium:
        """从当前 solver 持有的 x0 物化一个 Equilibrium snapshot."""

        return self.operator.build_equilibrium(self.x0)

    def _resolve_solve_config(
        self,
        *,
        method: str | None,
        max_residual: float | None,
        max_evaluations: int | None,
        enable_warmstart: bool | None,
        initial_policy: str | None,
        initial_homothetic_lambda: float | None,
        enable_fallback: bool | None,
        fallback_methods: tuple[str, ...] | list[str] | None,
        enable_verbose: bool | None,
        enable_history: bool | None,
        residual_normalization: str | None,
        residual_normalization_floor: float | None,
        residual_normalization_max_ratio: float | None,
        residual_normalization_huber_tau: float | None,
        residual_normalization_probe_count: int | None,
        residual_normalization_probe_step: float | None,
        residual_normalization_sensitivity_lambda: float | None,
        enable_collocation: bool | None,
        collocation_method: str | None,
        collocation_max_residual: float | None,
        collocation_max_evaluations: int | None,
    ) -> SolverConfig:
        """基于默认配置生成一次 solve 的临时配置快照."""

        overrides: dict[str, object] = {}
        if method is not None:
            overrides["method"] = str(method)
        if max_residual is not None:
            overrides["max_residual"] = float(max_residual)
        if max_evaluations is not None:
            overrides["max_evaluations"] = int(max_evaluations)
        if enable_warmstart is not None:
            overrides["enable_warmstart"] = bool(enable_warmstart)
        if initial_policy is not None:
            overrides["initial_policy"] = str(initial_policy)
        if initial_homothetic_lambda is not None:
            overrides["initial_homothetic_lambda"] = float(initial_homothetic_lambda)
        if enable_fallback is not None:
            overrides["enable_fallback"] = bool(enable_fallback)
        if fallback_methods is not None:
            overrides["fallback_methods"] = tuple(
                str(method_name) for method_name in fallback_methods
            )
        if enable_verbose is not None:
            overrides["enable_verbose"] = bool(enable_verbose)
        if enable_history is not None:
            overrides["enable_history"] = bool(enable_history)
        if residual_normalization is not None:
            overrides["residual_normalization"] = residual_normalization
        if residual_normalization_floor is not None:
            overrides["residual_normalization_floor"] = float(residual_normalization_floor)
        if residual_normalization_max_ratio is not None:
            overrides["residual_normalization_max_ratio"] = float(residual_normalization_max_ratio)
        if residual_normalization_huber_tau is not None:
            overrides["residual_normalization_huber_tau"] = float(residual_normalization_huber_tau)
        if residual_normalization_probe_count is not None:
            overrides["residual_normalization_probe_count"] = int(
                residual_normalization_probe_count
            )
        if residual_normalization_probe_step is not None:
            overrides["residual_normalization_probe_step"] = float(
                residual_normalization_probe_step
            )
        if residual_normalization_sensitivity_lambda is not None:
            overrides["residual_normalization_sensitivity_lambda"] = float(
                residual_normalization_sensitivity_lambda
            )
        if enable_collocation is not None:
            overrides["enable_collocation"] = bool(enable_collocation)
        if collocation_method is not None:
            overrides["collocation_method"] = str(collocation_method)
        if collocation_max_residual is not None:
            overrides["collocation_max_residual"] = float(collocation_max_residual)
        if collocation_max_evaluations is not None:
            overrides["collocation_max_evaluations"] = int(collocation_max_evaluations)
        if not overrides:
            return self.config
        return replace(self.config, **overrides)

    def _solve_with_collocation_polish(
        self,
        x_guess: np.ndarray,
        *,
        solve_config: SolverConfig,
        x0_was_provided: bool,
    ) -> tuple[np.ndarray, bool, str, int, int, int, float]:
        """先做 variational solve, 再从该结果热启动 collocation polish."""

        variational_config = self._variational_stage_config(solve_config)
        collocation_config = self._collocation_stage_config(solve_config)
        _validate_stage_solve_config(variational_config, residual_kind="variational")
        _validate_stage_solve_config(collocation_config, residual_kind="collocation")

        variational_result = self._solve_with_fallbacks(
            x_guess,
            solve_config=variational_config,
            residual_kind="variational",
            x0_was_provided=x0_was_provided,
        )
        collocation_result, collocation_error = self._try_solve_attempt(
            variational_result[0],
            solve_config=collocation_config,
            residual_kind="collocation",
        )
        if collocation_result is None:
            if collocation_error is not None:
                raise RuntimeError(
                    "Collocation polish failed without a usable result"
                ) from collocation_error
            raise RuntimeError("Collocation polish failed without a usable result")

        return self._combine_variational_collocation_results(
            variational_result=variational_result,
            collocation_result=collocation_result,
            collocation_error=collocation_error,
        )

    def _variational_stage_config(self, solve_config: SolverConfig) -> SolverConfig:
        """返回两阶段 workflow 的 variational stage 配置."""

        return replace(solve_config, enable_collocation=False)

    def _collocation_stage_config(self, solve_config: SolverConfig) -> SolverConfig:
        """返回两阶段 workflow 的 collocation polish 配置."""

        max_residual = (
            solve_config.max_residual
            if solve_config.collocation_max_residual is None
            else solve_config.collocation_max_residual
        )
        max_evaluations = (
            solve_config.max_evaluations
            if solve_config.collocation_max_evaluations is None
            else solve_config.collocation_max_evaluations
        )
        return replace(
            solve_config,
            method=solve_config.collocation_method,
            max_residual=max_residual,
            max_evaluations=max_evaluations,
            enable_collocation=False,
            enable_fallback=False,
            fallback_methods=(),
        )

    def _final_residual_config(self, solve_config: SolverConfig) -> SolverConfig:
        """返回 SolverResult 最终 x 应使用的 residual 评估配置."""

        if solve_config.enable_collocation:
            return self._collocation_stage_config(solve_config)
        return solve_config

    def _final_residual_kind(self, solve_config: SolverConfig) -> str:
        """返回 SolverResult 最终 x 应使用的 residual kind."""

        if solve_config.enable_collocation:
            return "collocation"
        return "variational"

    def _combine_variational_collocation_results(
        self,
        *,
        variational_result: tuple[np.ndarray, bool, str, int, int, int, float],
        collocation_result: tuple[np.ndarray, bool, str, int, int, int, float],
        collocation_error: Exception | None,
    ) -> tuple[np.ndarray, bool, str, int, int, int, float]:
        """合并两阶段计数, 但成功状态与最终 x 以 collocation stage 为准."""

        variational_status = "succeeded" if bool(variational_result[1]) else "failed"
        collocation_status = (
            "succeeded"
            if self._attempt_succeeded(collocation_result, collocation_error)
            else "failed"
        )
        collocation_failure = self._format_attempt_failure(
            method="collocation-polish",
            result=collocation_result,
            error=collocation_error,
        )
        message = (
            f"variational stage {variational_status}: {variational_result[2]}; "
            f"collocation polish {collocation_status}: {collocation_failure}"
        )
        return (
            collocation_result[0],
            self._attempt_succeeded(collocation_result, collocation_error),
            message,
            int(variational_result[3]) + int(collocation_result[3]),
            int(variational_result[4]) + int(collocation_result[4]),
            int(variational_result[5]) + int(collocation_result[5]),
            float(collocation_result[6]),
        )

    def _solve_with_fallbacks(
        self,
        x_guess: np.ndarray,
        *,
        solve_config: SolverConfig,
        residual_kind: str,
        x0_was_provided: bool,
    ) -> tuple[np.ndarray, bool, str, int, int, int, float]:
        """按主方法求解, 必要时按配置顺序回退到备用 solver 方法."""

        attempts: list[
            tuple[str, tuple[np.ndarray, bool, str, int, int, int, float] | None, Exception | None]
        ] = []

        attempt_plans = self._build_attempt_plans(
            x_guess,
            solve_config=solve_config,
            residual_kind=residual_kind,
            x0_was_provided=x0_was_provided,
        )

        for idx, attempt_plan in enumerate(attempt_plans):
            label, x_attempt_guess, attempt_config = attempt_plan
            result, error = self._try_solve_attempt(
                x_attempt_guess,
                solve_config=attempt_config,
                residual_kind=residual_kind,
            )
            attempts.append((label, result, error))
            if self._attempt_succeeded(result, error):
                if result is None:
                    raise RuntimeError("Solve attempt succeeded without a result")
                if idx == 0:
                    return result
                return self._finalize_attempts(attempts)

            if idx + 1 >= len(attempt_plans):
                break

            next_label = attempt_plans[idx + 1][0]
            failure = self._format_attempt_failure(
                method=label,
                result=result,
                error=error,
            )
            if solve_config.enable_verbose:
                warnings.warn(
                    (
                        f"Solve with method={label!r} failed ({failure}). "
                        f"Retrying with {next_label!r}."
                    ),
                    RuntimeWarning,
                    stacklevel=2,
                )

        return self._finalize_attempts(attempts)

    def _build_attempt_plans(
        self,
        x_guess: np.ndarray,
        *,
        solve_config: SolverConfig,
        residual_kind: str,
        x0_was_provided: bool,
    ) -> list[tuple[str, np.ndarray, SolverConfig]]:
        x_initial = self.operator.coerce_x(x_guess).copy()
        x_cold = np.zeros_like(x_initial)
        attempt_plans = [
            (
                self._display_attempt_label(
                    solve_config,
                    start_kind="warm-start"
                    if self._is_warm_initial_guess(x_initial, x0_was_provided)
                    else "cold-start",
                ),
                x_initial,
                solve_config,
            )
        ]

        if self._should_retry_from_reset(x_initial, x0_was_provided=x0_was_provided):
            attempt_plans.append(
                (
                    self._display_attempt_label(solve_config, start_kind="reset"),
                    x_cold.copy(),
                    solve_config,
                )
            )

        seen_methods = {solve_config.method}
        for fallback_method in self._ordered_fallback_methods(solve_config):
            if fallback_method in seen_methods:
                continue
            seen_methods.add(fallback_method)
            fallback_config = replace(solve_config, method=fallback_method)
            attempt_plans.append(
                (
                    self._display_attempt_label(fallback_config, start_kind="cold-fallback"),
                    x_cold.copy(),
                    fallback_config,
                )
            )
        return attempt_plans

    def _ordered_fallback_methods(self, solve_config: SolverConfig) -> tuple[str, ...]:
        if not solve_config.enable_fallback:
            return ()
        return tuple(solve_config.fallback_methods)

    def _try_solve_attempt(
        self,
        x_guess: np.ndarray,
        *,
        solve_config: SolverConfig,
        residual_kind: str,
    ) -> tuple[tuple[np.ndarray, bool, str, int, int, int, float] | None, Exception | None]:
        """包装一次 solve stage, 让 fallback 流程也能处理数值异常."""

        x_guess_eval = self.operator.coerce_x(x_guess).copy()
        try:
            return self._solve_opt_problem(
                x_guess_eval,
                solve_config=solve_config,
                residual_kind=residual_kind,
            ), None
        except Exception as exc:
            residual_norm_x0, residual_exc = self._safe_residual_norm(
                x_guess_eval,
                solve_config=solve_config,
                residual_kind=residual_kind,
            )
            if residual_exc is None and _residual_within_acceptance(residual_norm_x0, solve_config):
                return (
                    (
                        x_guess_eval.copy(),
                        True,
                        f"{type(exc).__name__}: {exc} "
                        f"[accepted by x0 residual={residual_norm_x0:.6e}]",
                        0,
                        0,
                        0,
                        float(residual_norm_x0),
                    ),
                    None,
                )
            return (
                (
                    x_guess_eval.copy(),
                    False,
                    f"{type(exc).__name__}: {exc}",
                    0,
                    0,
                    0,
                    float("nan") if residual_exc is not None else float(residual_norm_x0),
                ),
                exc,
            )

    def _format_attempt_failure(
        self,
        *,
        method: str,
        result: tuple[np.ndarray, bool, str, int, int, int, float] | None,
        error: Exception | None,
    ) -> str:
        if error is not None:
            return f"{type(error).__name__}: {error}"
        if result is None:
            return f"method={method} produced no result"
        return result[2]

    def _attempt_residual_norm(
        self,
        attempt: tuple[np.ndarray, bool, str, int, int, int, float] | None,
    ) -> float:
        if attempt is None:
            return float("inf")
        residual_norm = float(attempt[6])
        if not np.isfinite(residual_norm):
            return float("inf")
        return residual_norm

    def _safe_residual_norm(
        self,
        x: np.ndarray,
        *,
        solve_config: SolverConfig | None = None,
        residual_kind: str = "variational",
    ) -> tuple[float, Exception | None]:
        try:
            _ = self.config if solve_config is None else solve_config
            return _residual_array_norm(self._residual_function_for(residual_kind)(x)), None
        except Exception as exc:
            return float("inf"), exc

    def _attempt_succeeded(
        self,
        attempt: tuple[np.ndarray, bool, str, int, int, int, float] | None,
        error: Exception | None,
    ) -> bool:
        return bool(
            error is None
            and attempt is not None
            and bool(attempt[1])
            and np.isfinite(float(attempt[6]))
        )

    def _display_attempt_label(self, solve_config: SolverConfig, *, start_kind: str) -> str:
        return f"{self._display_method_label(solve_config)} [{start_kind}]"

    def _display_method_label(self, solve_config: SolverConfig) -> str:
        if _uses_least_squares_api(solve_config):
            return f"least_squares/{solve_config.method}"
        return f"root/{solve_config.method}"

    def _is_warm_initial_guess(self, x_guess: np.ndarray, x0_was_provided: bool) -> bool:
        return bool(x0_was_provided or not self._is_zero_guess(x_guess))

    def _should_retry_from_reset(self, x_guess: np.ndarray, *, x0_was_provided: bool) -> bool:
        if not self._is_warm_initial_guess(x_guess, x0_was_provided):
            return False
        return not self._is_zero_guess(x_guess)

    def _is_zero_guess(self, x_guess: np.ndarray) -> bool:
        x_eval = self.operator.coerce_x(x_guess)
        return bool(np.all(x_eval == 0.0))

    def _finalize_attempts(
        self,
        attempts: list[
            tuple[str, tuple[np.ndarray, bool, str, int, int, int, float] | None, Exception | None]
        ],
    ) -> tuple[np.ndarray, bool, str, int, int, int, float]:
        for label, result, error in reversed(attempts):
            if self._attempt_succeeded(result, error):
                return self._build_attempts_result(
                    attempts, selected_label=label, selected_result=result
                )

        candidate_idx = self._best_attempt_index(attempts)
        if candidate_idx is None:
            tail_label, _, tail_exc = attempts[-1]
            if tail_exc is not None:
                raise RuntimeError(
                    f"All solve attempts failed; last method={tail_label}"
                ) from tail_exc
            raise RuntimeError("All solve attempts failed without a usable result")

        selected_label, selected_result, _ = attempts[candidate_idx]
        if selected_result is None:
            raise RuntimeError("Selected solve attempt has no result")
        return self._build_attempts_result(
            attempts, selected_label=selected_label, selected_result=selected_result
        )

    def _build_attempts_result(
        self,
        attempts: list[
            tuple[str, tuple[np.ndarray, bool, str, int, int, int, float] | None, Exception | None]
        ],
        *,
        selected_label: str,
        selected_result: tuple[np.ndarray, bool, str, int, int, int, float],
    ) -> tuple[np.ndarray, bool, str, int, int, int, float]:
        message = "; ".join(
            f"attempt(method={label}) "
            f"{'succeeded' if self._attempt_succeeded(res, err) else 'failed'}: "
            f"{self._format_attempt_failure(method=label, result=res, error=err)}"
            for label, res, err in attempts
        )
        return (
            selected_result[0],
            self._attempt_succeeded(selected_result, None),
            f"{message}; selected method={selected_label}",
            sum(int(result[3]) for _, result, _ in attempts if result is not None),
            sum(int(result[4]) for _, result, _ in attempts if result is not None),
            sum(int(result[5]) for _, result, _ in attempts if result is not None),
            float(selected_result[6]),
        )

    def _best_attempt_index(
        self,
        attempts: list[
            tuple[str, tuple[np.ndarray, bool, str, int, int, int, float] | None, Exception | None]
        ],
    ) -> int | None:
        candidate_indices = [
            idx
            for idx, (_, result, error) in enumerate(attempts)
            if result is not None and error is None
        ]
        if not candidate_indices:
            candidate_indices = [
                idx for idx, (_, result, _) in enumerate(attempts) if result is not None
            ]
        if not candidate_indices:
            return None
        return min(candidate_indices, key=lambda idx: self._attempt_residual_norm(attempts[idx][1]))

    def _solve_opt_problem(
        self,
        x_guess: np.ndarray,
        *,
        solve_config: SolverConfig,
        residual_kind: str,
    ) -> tuple[np.ndarray, bool, str, int, int, int, float]:
        """执行一次完整 nonlinear solve."""

        opt = self._run_solve_full(x_guess, solve_config=solve_config, residual_kind=residual_kind)
        x_opt = self.operator.coerce_x(opt.x)
        residual_norm = self._optimizer_residual_norm(opt)
        if residual_norm is None or not np.isfinite(residual_norm):
            residual_norm, _ = self._safe_residual_norm(
                x_opt,
                solve_config=solve_config,
                residual_kind=residual_kind,
            )
        accepted_by_residual = _residual_within_acceptance(residual_norm, solve_config)
        accepted = bool(
            accepted_by_residual
            or (
                bool(opt.success)
                and residual_norm is not None
                and np.isfinite(residual_norm)
                and not _requires_strict_residual_acceptance(
                    solve_config, residual_kind=residual_kind
                )
            )
        )
        message = str(opt.message)
        if not bool(opt.success) and accepted:
            message = f"{message} [accepted by residual]"
        if (
            bool(opt.success)
            and not accepted
            and _requires_strict_residual_acceptance(
                solve_config,
                residual_kind=residual_kind,
            )
        ):
            message = f"{message} [rejected by residual={residual_norm:.6e}]"
        return (
            x_opt,
            accepted,
            message,
            _count_opt_attr(opt, "nfev"),
            _count_opt_attr(opt, "njev"),
            _count_opt_attr(opt, "nit"),
            float("nan") if residual_norm is None else float(residual_norm),
        )

    def _run_solve_full(
        self,
        x_guess: np.ndarray,
        *,
        solve_config: SolverConfig,
        residual_kind: str,
    ):
        _validate_stage_method(solve_config, residual_kind=residual_kind)
        optimize_method = _registered_method_for(solve_config)
        if solve_config.method in ROOT_METHODS:
            return self._run_root_full(
                x_guess, solve_config=solve_config, optimize_method=optimize_method
            )
        return self._run_least_squares_full(
            x_guess,
            solve_config=solve_config,
            optimize_method=optimize_method,
            residual_kind=residual_kind,
        )

    def _residual_function_for(self, residual_kind: str) -> Callable[[np.ndarray], np.ndarray]:
        def residual_fun(x: np.ndarray) -> np.ndarray:
            x_eval = self.operator.coerce_x(x)
            if residual_kind == "variational":
                return self.operator.residual_var(x_eval)
            if residual_kind == "collocation":
                return self.operator.residual_collocation(x_eval)
            raise ValueError(f"Unsupported residual kind {residual_kind!r}.")

        return residual_fun

    def _optimizer_residual_norm(self, opt) -> float | None:
        fun = getattr(opt, "fun", None)
        if fun is None:
            return None
        return _residual_array_norm(fun)

    def _run_root_once(
        self,
        root_fun,
        x_guess: np.ndarray,
        *,
        solve_config: SolverConfig,
        optimize_method: OptimizeMethod,
        options: dict[str, object],
        get_raw_residual: Callable[[np.ndarray], np.ndarray] | None = None,
        decode_x: Callable[[np.ndarray], np.ndarray] | None = None,
    ):
        opt = optimize_method(
            root_fun,
            x_guess,
            tol=solve_config.max_residual,
            options=options,
        )
        if decode_x is not None:
            opt.x = decode_x(opt.x)
        if get_raw_residual is not None:
            x_opt = self.operator.coerce_x(opt.x)
            opt.fun = get_raw_residual(x_opt)
        return opt

    def _run_root_full(
        self,
        x_guess: np.ndarray,
        *,
        solve_config: SolverConfig,
        optimize_method: OptimizeMethod,
    ):
        """在完整 packed x 上调用一次 `scipy.optimize.root`."""

        root_fun = self.operator
        get_raw_residual: Callable[[np.ndarray], np.ndarray] | None = None
        options = _root_options_for(solve_config)
        balanced_scope = "block"
        initial_residual: np.ndarray | None = None
        scaled_fun, get_raw_residual = self._build_normalized_residual_wrapper(
            x_guess,
            solve_config=solve_config,
            residual_kind="variational",
            legacy_transform="linear",
            balanced_scope=balanced_scope,
            initial_residual=initial_residual if balanced_scope == "block" else None,
        )
        x_root_guess = x_guess
        decode_x: Callable[[np.ndarray], np.ndarray] | None = None
        x_transform_fun, x_root_guess, decode_x = self._build_x_transform_wrapper(x_guess)
        if x_transform_fun is not None:
            if scaled_fun is not None:

                def root_fun(z_eval: np.ndarray) -> np.ndarray:
                    return scaled_fun(x_transform_fun(z_eval))
            else:

                def root_fun(z_eval: np.ndarray) -> np.ndarray:
                    return self.operator(x_transform_fun(z_eval))
        elif scaled_fun is not None:
            root_fun = scaled_fun
        if scaled_fun is not None and solve_config.method == "hybr":
            normalization_mode = getattr(solve_config, "residual_normalization", "robust_balanced")
            if normalization_mode in {"legacy", "robust_balanced", "sensitivity_balanced"}:
                options = {**options, "factor": 1.0}

        return self._run_root_once(
            root_fun,
            x_root_guess,
            solve_config=solve_config,
            optimize_method=optimize_method,
            options=options,
            get_raw_residual=get_raw_residual,
            decode_x=decode_x,
        )

    def _build_normalized_residual_wrapper(
        self,
        x_guess: np.ndarray,
        *,
        solve_config: SolverConfig,
        residual_kind: str,
        legacy_transform: str = "linear",
        balanced_scope: str = "block",
        initial_residual: np.ndarray | None = None,
    ) -> tuple[
        Callable[[np.ndarray], np.ndarray] | None, Callable[[np.ndarray], np.ndarray] | None
    ]:
        """构造 solver 层残差规范化 wrapper."""

        mode = getattr(solve_config, "residual_normalization", "robust_balanced")
        if mode == "none":
            return None, None
        if mode == "legacy":
            return self._build_legacy_residual_transform_wrapper(
                x_guess, transform=legacy_transform
            )
        if mode not in {"robust_balanced", "sensitivity_balanced"}:
            raise ValueError(f"Unsupported residual_normalization {mode!r}.")
        return self._build_balanced_residual_transform_wrapper(
            x_guess,
            solve_config=solve_config,
            residual_kind=residual_kind,
            scope=balanced_scope,
            initial_x=x_guess,
            initial_residual=initial_residual,
            mode=mode,
        )

    def _build_legacy_residual_transform_wrapper(
        self,
        x_guess: np.ndarray,
        *,
        transform: str,
    ) -> tuple[
        Callable[[np.ndarray], np.ndarray] | None, Callable[[np.ndarray], np.ndarray] | None
    ]:
        """旧版 block RMS 残差变换 wrapper; 仅保留为对比与兼容模式."""

        block_lengths = getattr(self.operator, "active_lengths", None)
        if block_lengths is None:
            return None, None

        try:
            self.operator.coerce_x(x_guess)
        except Exception:
            return None, None
        block_lengths_eval = np.asarray(block_lengths, dtype=np.int64)
        scale: np.ndarray | None = None
        last_x: np.ndarray | None = None
        last_raw_residual: np.ndarray | None = None

        def wrapped(x: np.ndarray) -> np.ndarray:
            nonlocal scale, last_x, last_raw_residual
            x_eval = self.operator.coerce_x(x)
            raw_residual = np.asarray(self.operator(x_eval), dtype=np.float64)
            last_x = x_eval.copy()
            last_raw_residual = raw_residual.copy()
            if scale is None:
                scale = _build_block_rms_scale(raw_residual, block_lengths_eval)
                if scale is None:
                    scale = np.ones_like(raw_residual)
            scaled_residual = raw_residual / scale
            if transform == "asinh":
                return np.arcsinh(scaled_residual)
            return scaled_residual

        def get_raw_residual(x: np.ndarray) -> np.ndarray:
            x_eval = self.operator.coerce_x(x)
            if (
                last_x is not None
                and last_raw_residual is not None
                and np.array_equal(last_x, x_eval)
            ):
                return last_raw_residual.copy()
            return np.asarray(self.operator(x_eval), dtype=np.float64)

        return wrapped, get_raw_residual

    def _build_balanced_residual_transform_wrapper(
        self,
        x_guess: np.ndarray,
        *,
        solve_config: SolverConfig,
        residual_kind: str,
        scope: str,
        initial_x: np.ndarray | None = None,
        initial_residual: np.ndarray | None = None,
        mode: str = "robust_balanced",
    ) -> tuple[
        Callable[[np.ndarray], np.ndarray] | None, Callable[[np.ndarray], np.ndarray] | None
    ]:
        """一次 O(n) 建模的线性残差左预条件器."""

        try:
            self.operator.coerce_x(x_guess)
        except Exception:
            return None, None

        residual_fun = self._residual_function_for(residual_kind)
        if scope not in {"block", "global"}:
            raise ValueError(f"Unsupported balanced residual scope {scope!r}.")
        block_lengths = (
            getattr(self.operator, "active_lengths", None)
            if residual_kind == "variational" and scope == "block"
            else None
        )
        block_lengths_eval = (
            None if block_lengths is None else np.asarray(block_lengths, dtype=np.int64)
        )
        floor = float(solve_config.residual_normalization_floor)
        max_ratio = float(solve_config.residual_normalization_max_ratio)
        huber_tau = float(solve_config.residual_normalization_huber_tau)
        scale: np.ndarray | None = None
        last_x: np.ndarray | None = None
        last_raw_residual: np.ndarray | None = None
        if initial_residual is None and mode in {
            "robust_balanced",
            "sensitivity_balanced",
        }:
            try:
                initial_x_eval = self.operator.coerce_x(x_guess)
                initial_residual = np.asarray(residual_fun(initial_x_eval), dtype=np.float64)
                initial_x = initial_x_eval
            except Exception:
                initial_residual = None
        if initial_residual is not None:
            initial_residual_eval = np.asarray(initial_residual, dtype=np.float64)
            scale = self._build_residual_scale_for_mode(
                initial_residual_eval,
                block_lengths_eval,
                solve_config=solve_config,
                residual_fun=residual_fun,
                x_guess=self.operator.coerce_x(x_guess),
                mode=mode,
                floor=floor,
                max_ratio=max_ratio,
                huber_tau=huber_tau,
            )
            if initial_x is not None:
                last_x = self.operator.coerce_x(initial_x).copy()
                last_raw_residual = initial_residual_eval.copy()

        def wrapped(x: np.ndarray) -> np.ndarray:
            nonlocal scale, last_x, last_raw_residual
            x_eval = self.operator.coerce_x(x)
            if (
                last_x is not None
                and last_raw_residual is not None
                and np.array_equal(last_x, x_eval)
            ):
                raw_residual = last_raw_residual.copy()
            else:
                raw_residual = np.asarray(residual_fun(x_eval), dtype=np.float64)
                last_x = x_eval.copy()
                last_raw_residual = raw_residual.copy()
            if scale is None:
                scale = self._build_residual_scale_for_mode(
                    raw_residual,
                    block_lengths_eval,
                    solve_config=solve_config,
                    residual_fun=residual_fun,
                    x_guess=x_eval,
                    mode=mode,
                    floor=floor,
                    max_ratio=max_ratio,
                    huber_tau=huber_tau,
                )
            return raw_residual / scale

        def get_raw_residual(x: np.ndarray) -> np.ndarray:
            x_eval = self.operator.coerce_x(x)
            if (
                last_x is not None
                and last_raw_residual is not None
                and np.array_equal(last_x, x_eval)
            ):
                return last_raw_residual.copy()
            return np.asarray(residual_fun(x_eval), dtype=np.float64)

        return wrapped, get_raw_residual

    def _build_residual_scale_for_mode(
        self,
        residual: np.ndarray,
        block_lengths: np.ndarray | None,
        *,
        solve_config: SolverConfig,
        residual_fun: Callable[[np.ndarray], np.ndarray],
        x_guess: np.ndarray,
        mode: str,
        floor: float,
        max_ratio: float,
        huber_tau: float,
    ) -> np.ndarray:
        if mode == "robust_balanced":
            return _build_robust_balanced_residual_scale(
                residual,
                block_lengths,
                floor=floor,
                max_ratio=max_ratio,
                huber_tau=huber_tau,
            )
        if mode == "sensitivity_balanced":
            return _build_sensitivity_balanced_residual_scale(
                residual_fun,
                x_guess,
                residual,
                block_lengths,
                x_scale=_build_x_block_scale_vector(self.operator, x_guess),
                floor=floor,
                max_ratio=max_ratio,
                huber_tau=huber_tau,
                probe_count=int(solve_config.residual_normalization_probe_count),
                probe_step=float(solve_config.residual_normalization_probe_step),
                sensitivity_lambda=float(solve_config.residual_normalization_sensitivity_lambda),
            )
        raise ValueError(f"Unsupported residual_normalization {mode!r}.")

    def _build_x_transform_wrapper(
        self,
        x_guess: np.ndarray,
    ) -> tuple[
        Callable[[np.ndarray], np.ndarray] | None,
        np.ndarray,
        Callable[[np.ndarray], np.ndarray] | None,
    ]:
        x_eval = self.operator.coerce_x(x_guess)
        x_scale = _build_x_block_scale_vector(self.operator, x_eval)
        if x_scale is None:
            return None, x_eval, None

        inv_scale = 1.0 / x_scale

        def map_z_to_x(z: np.ndarray) -> np.ndarray:
            z_eval = np.asarray(z, dtype=np.float64)
            if z_eval.ndim != 1 or z_eval.shape[0] != x_eval.shape[0]:
                raise ValueError(f"Expected z to have shape {x_eval.shape}, got {z_eval.shape}")
            return self.operator.coerce_x(z_eval * x_scale)

        return map_z_to_x, x_eval * inv_scale, map_z_to_x

    def _initial_residual_stats(
        self,
        x_guess: np.ndarray,
        *,
        residual_kind: str,
    ) -> tuple[np.ndarray | None, float | None]:
        try:
            residual_fun = self._residual_function_for(residual_kind)
            residual = np.asarray(residual_fun(self.operator.coerce_x(x_guess)), dtype=np.float64)
        except Exception:
            return None, None
        return residual, _residual_array_norm(residual)

    def _run_least_squares_full(
        self,
        x_guess: np.ndarray,
        *,
        solve_config: SolverConfig,
        optimize_method: OptimizeMethod,
        residual_kind: str,
    ):
        """在完整 packed x 上调用一次 `scipy.optimize.least_squares`."""

        least_squares_fun = self._residual_function_for(residual_kind)
        get_raw_residual: Callable[[np.ndarray], np.ndarray] | None = None
        kwargs = _least_squares_kwargs_for(solve_config)
        normalizer_applied = False

        if residual_kind == "variational":
            legacy_transform = "asinh" if solve_config.method == "lm" else "linear"
            normalized_fun, get_raw_residual = self._build_normalized_residual_wrapper(
                x_guess,
                solve_config=solve_config,
                residual_kind=residual_kind,
                legacy_transform=legacy_transform,
            )
            if normalized_fun is not None:
                least_squares_fun = normalized_fun
                normalizer_applied = True
                if solve_config.method == "lm":
                    kwargs["x_scale"] = 1.0

        if (
            not normalizer_applied
            and solve_config.method == "trf"
            and residual_kind == "variational"
        ):
            residual0, _ = self._initial_residual_stats(x_guess, residual_kind=residual_kind)
            if _should_use_robust_trf_loss(
                residual0,
                getattr(self.operator, "active_lengths", None),
            ):
                kwargs["loss"] = "cauchy"
                kwargs["f_scale"] = max(_residual_rms(residual0), 1.0)

        opt = optimize_method(
            least_squares_fun,
            x_guess,
            **kwargs,
        )
        if get_raw_residual is not None:
            x_opt = self.operator.coerce_x(opt.x)
            opt.fun = get_raw_residual(x_opt)
        return opt


def _validate_stage_solve_config(solve_config: SolverConfig, *, residual_kind: str) -> None:
    _validate_stage_method(solve_config, residual_kind=residual_kind)
    if residual_kind != "collocation" or not solve_config.enable_fallback:
        return

    root_fallbacks = [
        method for method in solve_config.fallback_methods if method not in LEAST_SQUARES_METHODS
    ]
    if root_fallbacks:
        unsupported = ", ".join(repr(method) for method in root_fallbacks)
        raise ValueError(
            f"Collocation needs least_squares ('trf' or 'lm'); bad fallback(s): {unsupported}."
        )


def _validate_stage_method(solve_config: SolverConfig, *, residual_kind: str) -> None:
    if residual_kind == "collocation" and not _uses_least_squares_api(solve_config):
        raise ValueError("Collocation needs least_squares ('trf' or 'lm').")
    if residual_kind not in {"variational", "collocation"}:
        raise ValueError(f"Unsupported residual kind {residual_kind!r}.")


def _registered_method_for(solve_config: SolverConfig) -> OptimizeMethod:
    try:
        return SUPPORTED_METHODS[solve_config.method]
    except KeyError as exc:
        raise ValueError(f"Unsupported solver method {solve_config.method!r}.") from exc


def _root_options_for(solve_config: SolverConfig) -> dict[str, object]:
    """将 `SolverConfig` 映射到 `scipy.optimize.root(..., options=...)`."""

    options: dict[str, object] = {}
    method = solve_config.method

    if method in {"hybr", "df-sane"}:
        if solve_config.max_evaluations > 0:
            options["maxfev"] = max(int(solve_config.max_evaluations), 500)
        if method == "hybr":
            options["eps"] = 1.0e-6
    return options


def _least_squares_kwargs_for(solve_config: SolverConfig) -> dict[str, object]:
    """将 `SolverConfig` 映射到 `scipy.optimize.least_squares(...)`."""

    kwargs: dict[str, object] = {
        "ftol": float(solve_config.max_residual),
        "xtol": float(solve_config.max_residual),
        "gtol": float(solve_config.max_residual),
    }
    if solve_config.max_evaluations > 0:
        kwargs["max_nfev"] = max(int(solve_config.max_evaluations), 500)
    return kwargs


def _count_opt_attr(opt, name: str) -> int:
    value = getattr(opt, name, 0)
    if value is None:
        return 0
    return int(value)


def _residual_array_norm(residual: np.ndarray) -> float:
    """返回 residual 数组的 Euclidean norm, 标量 residual 视作长度 1 向量."""

    residual_eval = np.asarray(residual, dtype=np.float64)
    if residual_eval.ndim == 0:
        residual_eval = residual_eval.reshape(1)
    return float(np.linalg.norm(residual_eval))


def _uses_least_squares_api(solve_config: SolverConfig) -> bool:
    return solve_config.method in LEAST_SQUARES_METHODS


def _build_initial_state(operator: Operator, solve_config: SolverConfig) -> np.ndarray:
    """Build the packed initial state requested by ``solve_config.initial_policy``."""

    initial_policy = solve_config.initial_policy
    if initial_policy is None:
        return operator.encode_initial_state()
    if initial_policy == "zeros":
        return np.zeros(operator.x_size, dtype=np.float64)
    if initial_policy == "homothetic":
        return _build_boundary_homothetic_initial_state(
            operator, boundary_slope_factor=solve_config.initial_homothetic_lambda
        )
    if initial_policy == "warm":
        raise RuntimeError("_build_initial_state('warm') needs the current solver x0")
    if initial_policy == "optimize":
        raise NotImplementedError(
            "initial_policy='optimize' is reserved for the line-search initializer"
        )
    raise ValueError(f"Unsupported initial_policy {initial_policy!r}")


def _build_boundary_homothetic_initial_state(
    operator: Operator, *, boundary_slope_factor: float = 1.0
) -> np.ndarray:
    """Return a cheap boundary-scaled x0 for nested, homothetic surfaces.

    ``boundary_slope_factor`` sets the target boundary slope ratio
    ``u_m'(1)=lambda*offset`` for each active c/s mode. The default ``1.0``
    matches homothetic scaling; smaller values relax the boundary more
    aggressively.
    """

    return _build_boundary_slope_initial_state(
        operator, boundary_slope_factor=boundary_slope_factor
    )


def _build_boundary_slope_initial_state(
    operator: Operator, *, boundary_slope_factor: float
) -> np.ndarray:
    """Set first c/s coefficients so ``u_m'(1)=lambda*offset``."""

    x = np.zeros(operator.x_size, dtype=np.float64)
    target_factor = float(boundary_slope_factor)
    for profile_id, name in enumerate(operator.profile_names):
        if not (name.startswith("c") or name.startswith("s")):
            continue
        slot = int(operator.active_slot_by_profile_id[int(profile_id)])
        if slot < 0 or int(operator.active_lengths[slot]) <= 0:
            continue
        profile = operator.profiles_by_name[name]
        power = int(profile.power)
        offset = float(profile.offset)
        if power <= 0 or abs(offset) <= 1.0e-14:
            continue
        coeff_index = int(operator.active_coeff_index_rows[slot, 0])
        x[coeff_index] = 0.5 * (float(power) - target_factor) * offset
    return x


def _accepted_residual_norm(solve_config: SolverConfig) -> float:
    return max(float(solve_config.max_residual) * 10.0, 1.0e-5)


def _residual_within_acceptance(residual_norm: float | None, solve_config: SolverConfig) -> bool:
    return bool(
        residual_norm is not None
        and np.isfinite(residual_norm)
        and residual_norm <= _accepted_residual_norm(solve_config)
    )


def _requires_strict_residual_acceptance(solve_config: SolverConfig, *, residual_kind: str) -> bool:
    return residual_kind == "variational"


def _hard_residual_norm_threshold() -> float:
    return 1.0e3


def _trf_robust_block_rms_threshold() -> float:
    return 2.0e40


def _residual_rms(residual: np.ndarray) -> float:
    residual_eval = np.asarray(residual, dtype=np.float64)
    if residual_eval.ndim != 1 or residual_eval.size == 0:
        return 1.0
    return float(np.linalg.norm(residual_eval) / np.sqrt(residual_eval.size))


def _block_rms_values(residual: np.ndarray, block_lengths: np.ndarray) -> np.ndarray | None:
    residual_eval = np.asarray(residual, dtype=np.float64)
    lengths_eval = np.asarray(block_lengths, dtype=np.int64)
    if (
        residual_eval.ndim != 1
        or lengths_eval.ndim != 1
        or residual_eval.size == 0
        or lengths_eval.size == 0
    ):
        return None
    if int(np.sum(lengths_eval)) != int(residual_eval.size):
        return None

    values = np.empty_like(lengths_eval, dtype=np.float64)
    offset = 0
    for idx, length in enumerate(lengths_eval):
        block_size = int(length)
        if block_size <= 0:
            return None
        block = residual_eval[offset : offset + block_size]
        block_rms = float(np.linalg.norm(block) / np.sqrt(block_size))
        if not np.isfinite(block_rms):
            return None
        values[idx] = block_rms
        offset += block_size
    return values


def _should_use_robust_trf_loss(
    residual: np.ndarray | None,
    block_lengths: np.ndarray | None,
) -> bool:
    if residual is None or block_lengths is None:
        return False
    block_rms = _block_rms_values(residual, np.asarray(block_lengths, dtype=np.int64))
    if block_rms is None or block_rms.size == 0:
        return False
    return bool(np.median(block_rms) >= _trf_robust_block_rms_threshold())


def _build_block_rms_scale(residual: np.ndarray, block_lengths: np.ndarray) -> np.ndarray | None:
    residual_eval = np.asarray(residual, dtype=np.float64)
    lengths_eval = np.asarray(block_lengths, dtype=np.int64)
    block_rms = _block_rms_values(residual_eval, lengths_eval)
    if block_rms is None:
        return None

    scale = np.empty_like(residual_eval)
    offset = 0
    for value, length in zip(block_rms, lengths_eval, strict=False):
        block_scale = max(float(value), 1.0)
        scale[offset : offset + int(length)] = block_scale
        offset += int(length)
    return scale


def _build_robust_balanced_residual_scale(
    residual: np.ndarray,
    block_lengths: np.ndarray | None,
    *,
    floor: float,
    max_ratio: float,
    huber_tau: float,
) -> np.ndarray:
    residual_eval = np.asarray(residual, dtype=np.float64)
    if residual_eval.ndim != 1 or residual_eval.size == 0:
        return np.ones_like(residual_eval, dtype=np.float64)

    floor_eval = max(float(floor), np.finfo(np.float64).tiny)
    block_values = _robust_balanced_block_rms_values(
        residual_eval,
        block_lengths,
        huber_tau=float(huber_tau),
    )
    if block_values is None:
        global_scale = _robust_rms(residual_eval, huber_tau=float(huber_tau))
        global_scale = _clip_scale_by_anchor(
            np.asarray([global_scale], dtype=np.float64),
            floor=floor_eval,
            max_ratio=max_ratio,
        )[0]
        return np.full_like(residual_eval, global_scale, dtype=np.float64)

    clipped_values = _clip_scale_by_anchor(block_values, floor=floor_eval, max_ratio=max_ratio)
    lengths_eval = np.asarray(block_lengths, dtype=np.int64)
    scale = np.empty_like(residual_eval, dtype=np.float64)
    offset = 0
    for value, length in zip(clipped_values, lengths_eval, strict=False):
        block_size = int(length)
        scale[offset : offset + block_size] = float(value)
        offset += block_size
    return scale


def _build_sensitivity_balanced_residual_scale(
    residual_fun: Callable[[np.ndarray], np.ndarray],
    x_guess: np.ndarray,
    residual: np.ndarray,
    block_lengths: np.ndarray | None,
    *,
    x_scale: np.ndarray | None,
    floor: float,
    max_ratio: float,
    huber_tau: float,
    probe_count: int,
    probe_step: float,
    sensitivity_lambda: float,
) -> np.ndarray:
    residual_eval = np.asarray(residual, dtype=np.float64)
    amplitude_values = _robust_balanced_block_rms_values(
        residual_eval,
        block_lengths,
        huber_tau=huber_tau,
    )
    if amplitude_values is None or block_lengths is None or int(probe_count) <= 0:
        return _build_robust_balanced_residual_scale(
            residual_eval,
            block_lengths,
            floor=floor,
            max_ratio=max_ratio,
            huber_tau=huber_tau,
        )

    x_eval = np.asarray(x_guess, dtype=np.float64)
    if x_scale is None or np.asarray(x_scale).shape != x_eval.shape:
        x_scale_eval = np.maximum(np.abs(x_eval), 1.0)
    else:
        x_scale_eval = np.asarray(x_scale, dtype=np.float64)
    q = int(probe_count)
    step = float(probe_step)
    rng = np.random.default_rng(0)
    sensitivity_sq = np.zeros_like(amplitude_values, dtype=np.float64)
    for _ in range(q):
        signs = rng.integers(0, 2, size=x_eval.shape, dtype=np.int8).astype(np.float64)
        signs = signs * 2.0 - 1.0
        probe_x = x_eval + step * x_scale_eval * signs
        diff = (np.asarray(residual_fun(probe_x), dtype=np.float64) - residual_eval) / step
        probe_values = _robust_balanced_block_rms_values(diff, block_lengths, huber_tau=huber_tau)
        if probe_values is None:
            continue
        sensitivity_sq += probe_values * probe_values
    sensitivity_values = np.sqrt(sensitivity_sq / float(q))
    combined = np.sqrt(
        amplitude_values * amplitude_values + (float(sensitivity_lambda) * sensitivity_values) ** 2
    )
    clipped_values = _clip_scale_by_anchor(combined, floor=floor, max_ratio=max_ratio)

    lengths_eval = np.asarray(block_lengths, dtype=np.int64)
    scale = np.empty_like(residual_eval, dtype=np.float64)
    offset = 0
    for value, length in zip(clipped_values, lengths_eval, strict=False):
        block_size = int(length)
        scale[offset : offset + block_size] = float(value)
        offset += block_size
    return scale


def _robust_balanced_block_rms_values(
    residual: np.ndarray,
    block_lengths: np.ndarray | None,
    *,
    huber_tau: float,
) -> np.ndarray | None:
    residual_eval = np.asarray(residual, dtype=np.float64)
    if block_lengths is None:
        return None
    lengths_eval = np.asarray(block_lengths, dtype=np.int64)
    if (
        residual_eval.ndim != 1
        or lengths_eval.ndim != 1
        or residual_eval.size == 0
        or lengths_eval.size == 0
        or int(np.sum(lengths_eval)) != int(residual_eval.size)
    ):
        return None

    values = np.empty_like(lengths_eval, dtype=np.float64)
    offset = 0
    for idx, length in enumerate(lengths_eval):
        block_size = int(length)
        if block_size <= 0:
            return None
        values[idx] = _robust_rms(
            residual_eval[offset : offset + block_size],
            huber_tau=huber_tau,
        )
        offset += block_size
    return values


def _robust_rms(values: np.ndarray, *, huber_tau: float) -> float:
    arr = np.asarray(values, dtype=np.float64)
    if arr.size == 0:
        return 0.0
    finite = np.abs(arr[np.isfinite(arr)])
    if finite.size == 0:
        return 0.0
    center = float(np.median(finite))
    mad = float(np.median(np.abs(finite - center)))
    cutoff = center + max(float(huber_tau), 0.0) * 1.4826 * mad
    if not np.isfinite(cutoff) or cutoff <= 0.0:
        cutoff = center
    clipped = np.minimum(finite, cutoff)
    return _stable_rms(clipped)


def _stable_rms(values: np.ndarray) -> float:
    """Overflow-resistant RMS of finite entries, with nonfinite values ignored."""

    arr = np.asarray(values, dtype=np.float64)
    if arr.size == 0:
        return 0.0
    finite = np.abs(arr[np.isfinite(arr)])
    if finite.size == 0:
        return 0.0
    max_abs = float(np.max(finite))
    if max_abs == 0.0:
        return 0.0
    scaled = finite / max_abs
    return float(max_abs * np.sqrt(np.mean(scaled * scaled)))


def _balanced_residual_anchor(values: np.ndarray, *, floor: float) -> float:
    finite_positive = np.asarray(values, dtype=np.float64)
    finite_positive = finite_positive[np.isfinite(finite_positive) & (finite_positive > 0.0)]
    if finite_positive.size == 0:
        return float(floor)
    anchor = float(np.median(finite_positive))
    if not np.isfinite(anchor) or anchor < floor:
        return float(floor)
    return anchor


def _clip_scale_by_anchor(values: np.ndarray, *, floor: float, max_ratio: float) -> np.ndarray:
    values_eval = np.asarray(values, dtype=np.float64)
    floor_eval = max(float(floor), np.finfo(np.float64).tiny)
    ratio_eval = max(float(max_ratio), 1.0)
    scale = np.maximum(values_eval, floor_eval)
    scale[~np.isfinite(scale)] = floor_eval
    anchor = _balanced_residual_anchor(scale, floor=floor_eval)
    lower = max(floor_eval, anchor / ratio_eval)
    upper = _balanced_residual_upper_cap(anchor, floor=floor_eval, max_ratio=ratio_eval)
    return np.clip(scale, lower, upper)


def _balanced_residual_upper_cap(anchor: float, *, floor: float, max_ratio: float) -> float:
    cap = float(anchor) * float(max_ratio)
    if not np.isfinite(cap) or cap < floor:
        return float(np.finfo(np.float64).max)
    return max(float(floor), cap)


def _x_scale_floor() -> float:
    return 1.0e-2


def _x_scale_core_profile_prior() -> float:
    return 1.5e-1


def _x_scale_fourier_profile_prior() -> float:
    return 5.0e-2


def _x_scale_f_profile_prior() -> float:
    return 2.5e-1


def _x_scale_kappa_profile_prior() -> float:
    return 1.0


def _x_scale_profile_prior(name: str) -> float:
    if name in {"h", "v", "psin"}:
        return _x_scale_core_profile_prior()
    if name == "k":
        return _x_scale_kappa_profile_prior()
    if name.startswith(("c", "s")):
        return _x_scale_fourier_profile_prior()
    if name == "F":
        return _x_scale_f_profile_prior()
    return _x_scale_f_profile_prior()


def _use_offset_for_x_scale(name: str) -> bool:
    return name not in {"h", "v", "psin"}


def _build_x_block_scale_vector(operator, x_guess: np.ndarray) -> np.ndarray | None:
    x_eval = np.asarray(x_guess, dtype=np.float64)
    active_lengths = getattr(operator, "active_lengths", None)
    active_offsets = getattr(operator, "active_offsets", None)
    active_scales = getattr(operator, "active_scales", None)
    active_coeff_index_rows = getattr(operator, "active_coeff_index_rows", None)
    if (
        active_lengths is None
        or active_offsets is None
        or active_scales is None
        or active_coeff_index_rows is None
    ):
        return None

    lengths_eval = np.asarray(active_lengths, dtype=np.int64)
    offsets_eval = np.asarray(active_offsets, dtype=np.float64)
    scales_eval = np.asarray(active_scales, dtype=np.float64)
    coeff_rows_eval = np.asarray(active_coeff_index_rows, dtype=np.int64)
    if (
        lengths_eval.ndim != 1
        or offsets_eval.shape != lengths_eval.shape
        or scales_eval.shape != lengths_eval.shape
    ):
        return None

    scale = np.ones_like(x_eval)
    floor = _x_scale_floor()
    for slot, (p, block_len) in enumerate(
        zip(operator.active_profile_ids, lengths_eval, strict=False)
    ):
        length = int(block_len)
        if length <= 0:
            continue
        coeff_indices = coeff_rows_eval[slot, :length]
        if np.any(coeff_indices < 0) or np.any(coeff_indices >= x_eval.size):
            return None
        block_guess = x_eval[coeff_indices]
        guess_rms = float(np.linalg.norm(block_guess) / np.sqrt(length))
        profile_name = operator.profile_names[int(p)]
        offset_scale = (
            abs(float(offsets_eval[slot])) if _use_offset_for_x_scale(profile_name) else 0.0
        )
        profile_scale = abs(float(scales_eval[slot]))
        profile_prior = _x_scale_profile_prior(profile_name)
        if abs(profile_scale - 1.0) <= 1.0e-12:
            profile_scale = profile_prior
        block_scale = max(offset_scale, profile_scale, profile_prior, guess_rms, floor)
        scale[coeff_indices] = block_scale
    return scale
