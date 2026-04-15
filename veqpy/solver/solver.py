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
from scipy.optimize import least_squares, root

from veqpy.model.equilibrium import Equilibrium
from veqpy.operator.operator import Operator
from veqpy.operator.operator_case import OperatorCase
from veqpy.solver.solver_config import LEAST_SQUARES_METHODS, SolverConfig
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
        rtol: float | None = None,
        atol: float | None = None,
        root_maxiter: int | None = None,
        root_maxfev: int | None = None,
        enable_warmstart: bool | None = None,
        enable_fallback: bool | None = None,
        fallback_methods: tuple[str, ...] | list[str] | None = None,
        enable_verbose: bool | None = None,
        enable_history: bool | None = None,
    ) -> np.ndarray:
        """执行一次求解并返回收敛后的 packed x."""

        solve_config = self._resolve_solve_config(
            method=method,
            rtol=rtol,
            atol=atol,
            root_maxiter=root_maxiter,
            root_maxfev=root_maxfev,
            enable_warmstart=enable_warmstart,
            enable_fallback=enable_fallback,
            fallback_methods=fallback_methods,
            enable_verbose=enable_verbose,
            enable_history=enable_history,
        )

        if x0 is not None:
            self.x0 = self.operator.coerce_x(x0).copy()
        elif not solve_config.enable_warmstart:
            self.reset()
        if x0 is not None or not solve_config.enable_warmstart:
            self.operator.invalidate_source_state()

        x_guess = self.x0.copy()

        residual_norm_initial = float("nan")

        started = perf_counter()
        x_opt, success, message, nfev, njev, nit, residual_norm_final = self._solve_with_fallbacks(
            x_guess,
            solve_config=solve_config,
            x0_was_provided=x0 is not None,
        )
        elapsed = (perf_counter() - started) * 1e6

        x_final = self.operator.coerce_x(x_opt)
        residual_final_exc = None
        if not bool(success) and not np.isfinite(residual_norm_final):
            residual_norm_final, residual_final_exc = self._safe_residual_norm(x_final)
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
            residual_norm_initial=residual_norm_initial,
            residual_norm_final=residual_norm_final,
            nfev=int(nfev),
            njev=int(njev),
            nit=int(nit),
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
        rtol: float | None,
        atol: float | None,
        root_maxiter: int | None,
        root_maxfev: int | None,
        enable_warmstart: bool | None,
        enable_fallback: bool | None,
        fallback_methods: tuple[str, ...] | list[str] | None,
        enable_verbose: bool | None,
        enable_history: bool | None,
    ) -> SolverConfig:
        """基于默认配置生成一次 solve 的临时配置快照."""

        overrides: dict[str, object] = {}
        if method is not None:
            overrides["method"] = str(method)
        if rtol is not None:
            overrides["rtol"] = float(rtol)
        if atol is not None:
            overrides["atol"] = float(atol)
        if root_maxiter is not None:
            overrides["root_maxiter"] = int(root_maxiter)
        if root_maxfev is not None:
            overrides["root_maxfev"] = int(root_maxfev)
        if enable_warmstart is not None:
            overrides["enable_warmstart"] = bool(enable_warmstart)
        if enable_fallback is not None:
            overrides["enable_fallback"] = bool(enable_fallback)
        if fallback_methods is not None:
            overrides["fallback_methods"] = tuple(str(method_name) for method_name in fallback_methods)
        if enable_verbose is not None:
            overrides["enable_verbose"] = bool(enable_verbose)
        if enable_history is not None:
            overrides["enable_history"] = bool(enable_history)
        if not overrides:
            return self.config
        return replace(self.config, **overrides)

    def _solve_with_fallbacks(
        self,
        x_guess: np.ndarray,
        *,
        solve_config: SolverConfig,
        x0_was_provided: bool,
    ) -> tuple[np.ndarray, bool, str, int, int, int, float]:
        """按主方法求解, 必要时按配置顺序回退到备用 solver 方法."""

        attempts: list[tuple[str, tuple[np.ndarray, bool, str, int, int, int, float] | None, Exception | None]] = []

        attempt_plans = self._build_attempt_plans(
            x_guess,
            solve_config=solve_config,
            x0_was_provided=x0_was_provided,
        )

        for idx, attempt_plan in enumerate(attempt_plans):
            label, x_attempt_guess, attempt_config = attempt_plan
            result, error = self._try_solve_attempt(x_attempt_guess, solve_config=attempt_config)
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
                    (f"Solve with method={label!r} failed ({failure}). Retrying with {next_label!r}."),
                    RuntimeWarning,
                    stacklevel=2,
                )

        return self._finalize_attempts(attempts)

    def _build_attempt_plans(
        self,
        x_guess: np.ndarray,
        *,
        solve_config: SolverConfig,
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
    ) -> tuple[tuple[np.ndarray, bool, str, int, int, int, float] | None, Exception | None]:
        """包装一次 solve stage, 让 fallback 流程也能处理数值异常."""

        x_guess_eval = self.operator.coerce_x(x_guess).copy()
        try:
            return self._solve_opt_problem(x_guess_eval, solve_config=solve_config), None
        except Exception as exc:
            residual_norm_x0, residual_exc = self._safe_residual_norm(x_guess_eval)
            if residual_exc is None and _residual_within_acceptance(residual_norm_x0, solve_config):
                return (
                    (
                        x_guess_eval.copy(),
                        True,
                        f"{type(exc).__name__}: {exc} [accepted by x0 residual={residual_norm_x0:.6e}]",
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

    def _safe_residual_norm(self, x: np.ndarray) -> tuple[float, Exception | None]:
        try:
            return float(np.linalg.norm(self.operator(x))), None
        except Exception as exc:
            return float("inf"), exc

    def _attempt_succeeded(
        self,
        attempt: tuple[np.ndarray, bool, str, int, int, int, float] | None,
        error: Exception | None,
    ) -> bool:
        return bool(error is None and attempt is not None and bool(attempt[1]) and np.isfinite(float(attempt[6])))

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
        attempts: list[tuple[str, tuple[np.ndarray, bool, str, int, int, int, float] | None, Exception | None]],
    ) -> tuple[np.ndarray, bool, str, int, int, int, float]:
        for label, result, error in reversed(attempts):
            if self._attempt_succeeded(result, error):
                return self._build_attempts_result(attempts, selected_label=label, selected_result=result)

        candidate_idx = self._best_attempt_index(attempts)
        if candidate_idx is None:
            tail_label, _, tail_exc = attempts[-1]
            if tail_exc is not None:
                raise RuntimeError(f"All solve attempts failed; last method={tail_label}") from tail_exc
            raise RuntimeError("All solve attempts failed without a usable result")

        selected_label, selected_result, _ = attempts[candidate_idx]
        if selected_result is None:
            raise RuntimeError("Selected solve attempt has no result")
        return self._build_attempts_result(attempts, selected_label=selected_label, selected_result=selected_result)

    def _build_attempts_result(
        self,
        attempts: list[tuple[str, tuple[np.ndarray, bool, str, int, int, int, float] | None, Exception | None]],
        *,
        selected_label: str,
        selected_result: tuple[np.ndarray, bool, str, int, int, int, float],
    ) -> tuple[np.ndarray, bool, str, int, int, int, float]:
        message = "; ".join(
            f"attempt(method={label}) {'succeeded' if self._attempt_succeeded(res, err) else 'failed'}: "
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
        attempts: list[tuple[str, tuple[np.ndarray, bool, str, int, int, int, float] | None, Exception | None]],
    ) -> int | None:
        candidate_indices = [
            idx for idx, (_, result, error) in enumerate(attempts) if result is not None and error is None
        ]
        if not candidate_indices:
            candidate_indices = [idx for idx, (_, result, _) in enumerate(attempts) if result is not None]
        if not candidate_indices:
            return None
        return min(candidate_indices, key=lambda idx: self._attempt_residual_norm(attempts[idx][1]))

    def _solve_opt_problem(
        self,
        x_guess: np.ndarray,
        *,
        solve_config: SolverConfig,
    ) -> tuple[np.ndarray, bool, str, int, int, int, float]:
        """执行一次完整 nonlinear solve."""

        opt = self._run_solve_full(x_guess, solve_config=solve_config)
        x_opt = self.operator.coerce_x(opt.x)
        residual_norm = _opt_residual_norm(opt)
        if residual_norm is None or not np.isfinite(residual_norm):
            residual_norm, _ = self._safe_residual_norm(x_opt)
        accepted_by_residual = _residual_within_acceptance(residual_norm, solve_config)
        accepted = bool(
            accepted_by_residual
            or (
                bool(opt.success)
                and residual_norm is not None
                and np.isfinite(residual_norm)
                and not _requires_strict_residual_acceptance(solve_config)
            )
        )
        message = str(opt.message)
        if not bool(opt.success) and accepted:
            message = f"{message} [accepted by residual]"
        if bool(opt.success) and not accepted and _requires_strict_residual_acceptance(solve_config):
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
    ):
        if _uses_least_squares_api(solve_config):
            return self._run_least_squares_full(x_guess, solve_config=solve_config)
        return self._run_root_full(x_guess, solve_config=solve_config)

    def _run_root_once(
        self,
        root_fun,
        x_guess: np.ndarray,
        *,
        solve_config: SolverConfig,
        options: dict[str, object],
        get_raw_residual: Callable[[np.ndarray], np.ndarray] | None = None,
    ):
        opt = root(
            root_fun,
            x_guess,
            method=_root_method_name_for(solve_config),
            tol=solve_config.atol,
            options=options,
        )
        if get_raw_residual is not None:
            x_opt = self.operator.coerce_x(opt.x)
            opt.fun = get_raw_residual(x_opt)
        return opt

    def _run_root_full(
        self,
        x_guess: np.ndarray,
        *,
        solve_config: SolverConfig,
    ):
        """在完整 packed x 上调用一次 `scipy.optimize.root`."""

        root_fun = self.operator
        get_raw_residual: Callable[[np.ndarray], np.ndarray] | None = None
        options = _root_options_for(solve_config)
        scaled_fun, get_raw_residual = self._build_residual_transform_wrapper(
            x_guess,
            transform="linear",
        )
        if scaled_fun is not None:
            root_fun = scaled_fun
            if solve_config.method == "hybr":
                options = {**options, "factor": 1.0}

        return self._run_root_once(
            root_fun,
            x_guess,
            solve_config=solve_config,
            options=options,
            get_raw_residual=get_raw_residual,
        )

    def _build_residual_transform_wrapper(
        self,
        x_guess: np.ndarray,
        *,
        transform: str,
    ) -> tuple[Callable[[np.ndarray], np.ndarray] | None, Callable[[np.ndarray], np.ndarray] | None]:
        """为 solver 层构造带 block scaling 的残差变换 wrapper."""

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
            if last_x is not None and last_raw_residual is not None and np.array_equal(last_x, x_eval):
                return last_raw_residual.copy()
            return np.asarray(self.operator(x_eval), dtype=np.float64)

        return wrapped, get_raw_residual

    def _initial_residual_stats(self, x_guess: np.ndarray) -> tuple[np.ndarray | None, float | None]:
        try:
            residual = np.asarray(self.operator(self.operator.coerce_x(x_guess)), dtype=np.float64)
        except Exception:
            return None, None
        return residual, float(np.linalg.norm(residual))

    def _run_least_squares_full(
        self,
        x_guess: np.ndarray,
        *,
        solve_config: SolverConfig,
    ):
        """在完整 packed x 上调用一次 `scipy.optimize.least_squares`."""

        least_squares_fun = self.operator
        get_raw_residual: Callable[[np.ndarray], np.ndarray] | None = None
        kwargs = _least_squares_kwargs_for(solve_config)
        if solve_config.method == "lm":
            least_squares_fun, get_raw_residual = self._build_residual_transform_wrapper(x_guess, transform="asinh")
            if least_squares_fun is not None:
                kwargs["x_scale"] = 1.0
            else:
                least_squares_fun = self.operator
        elif solve_config.method == "trf":
            residual0, _ = self._initial_residual_stats(x_guess)
            if _should_use_robust_trf_loss(
                residual0,
                getattr(self.operator, "active_lengths", None),
            ):
                kwargs["loss"] = "cauchy"
                kwargs["f_scale"] = max(_residual_rms(residual0), 1.0)

        opt = least_squares(
            least_squares_fun,
            x_guess,
            **kwargs,
        )
        if get_raw_residual is not None:
            x_opt = self.operator.coerce_x(opt.x)
            opt.fun = get_raw_residual(x_opt)
        return opt


def _root_options_for(solve_config: SolverConfig) -> dict[str, object]:
    """将 `SolverConfig` 映射到 `scipy.optimize.root(..., options=...)`."""

    options: dict[str, object] = {}
    method = solve_config.method

    if method in {"hybr", "df-sane"}:
        if solve_config.root_maxfev > 0:
            options["maxfev"] = max(int(solve_config.root_maxfev), 500)
    else:
        if solve_config.root_maxiter > 0:
            options["maxiter"] = int(solve_config.root_maxiter)
    return options


def _least_squares_kwargs_for(solve_config: SolverConfig) -> dict[str, object]:
    """将 `SolverConfig` 映射到 `scipy.optimize.least_squares(...)`."""

    kwargs: dict[str, object] = {
        "method": solve_config.method,
        "ftol": float(solve_config.atol),
        "xtol": float(solve_config.atol),
        "gtol": float(solve_config.atol),
    }
    if solve_config.root_maxfev > 0:
        kwargs["max_nfev"] = max(int(solve_config.root_maxfev), 500)
    return kwargs


def _count_opt_attr(opt, name: str) -> int:
    value = getattr(opt, name, 0)
    if value is None:
        return 0
    return int(value)


def _opt_residual_norm(opt) -> float | None:
    fun = getattr(opt, "fun", None)
    if fun is None:
        return None
    arr = np.asarray(fun, dtype=np.float64)
    if arr.ndim == 0:
        arr = arr.reshape(1)
    return float(np.linalg.norm(arr))


def _uses_least_squares_api(solve_config: SolverConfig) -> bool:
    return solve_config.method in LEAST_SQUARES_METHODS


def _accepted_residual_norm(solve_config: SolverConfig) -> float:
    return max(float(solve_config.atol) * 10.0, 1.0e-5)


def _residual_within_acceptance(residual_norm: float | None, solve_config: SolverConfig) -> bool:
    return bool(
        residual_norm is not None
        and np.isfinite(residual_norm)
        and residual_norm <= _accepted_residual_norm(solve_config)
    )


def _root_method_name_for(solve_config: SolverConfig) -> str:
    return solve_config.method


def _requires_strict_residual_acceptance(solve_config: SolverConfig) -> bool:
    return solve_config.method in {"lm", "trf"}


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
    if residual_eval.ndim != 1 or lengths_eval.ndim != 1 or residual_eval.size == 0 or lengths_eval.size == 0:
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
