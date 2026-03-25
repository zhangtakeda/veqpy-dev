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

from dataclasses import replace
from time import perf_counter
from types import SimpleNamespace
import warnings

import numpy as np
from rich.console import Console
from scipy.optimize import least_squares, root

from veqpy.model import Equilibrium
from veqpy.operator.operator import Operator
from veqpy.operator.operator_case import OperatorCase
from veqpy.solver.solver_config import (
    SUPPORTED_LEAST_SQUARES_METHODS,
    SUPPORTED_ROOT_METHODS,
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

    def replace_config(self, config: SolverConfig) -> None:
        """替换 solver 的长期默认配置."""

        self.config = config

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
        enable_homotopy: bool | None = None,
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
            enable_homotopy=enable_homotopy,
            enable_verbose=enable_verbose,
            enable_history=enable_history,
        )

        if x0 is not None:
            self.x0 = self.operator.coerce_x(x0).copy()
        elif not solve_config.enable_warmstart:
            self.reset()

        x_guess = self.x0.copy()

        residual_norm_initial = float("nan")

        started = perf_counter()
        x_opt, success, message, nfev, njev, nit, residual_norm_final = self._solve_with_fallbacks(
            x_guess,
            solve_config=solve_config,
        )
        elapsed = (perf_counter() - started) * 1e6

        x_final = self.operator.coerce_x(x_opt)
        residual_final_exc = None
        if not bool(success) and not np.isfinite(residual_norm_final):
            residual_norm_final, residual_final_exc = self._safe_residual_norm(x_final)
        if residual_final_exc is not None:
            success = False
            message = f"{message} [final residual evaluation failed: {type(residual_final_exc).__name__}:"
            f" {residual_final_exc}]"

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

    def build_coeffs_history(
        self,
        *,
        include_none: bool = True,
    ) -> list[dict[str, list[float] | None]]:
        """从 history 中每个结果快照重建 profile 系数字典."""

        history = []
        for record in self.history:
            history.append(self.operator.build_coeffs(record.result_snapshot.x, include_none=include_none))
        return history

    def build_equilibrium(self) -> Equilibrium:
        """从当前 solver 持有的 x0 物化一个 Equilibrium snapshot."""

        return self.operator.build_equilibrium(self.x0)

    def build_equilibrium_history(self) -> list[Equilibrium]:
        """从 history 中每个结果快照物化 Equilibrium."""

        saved_case = self.operator.case.copy()
        saved_config = self.config
        saved_result = self.result
        saved_x0 = self.x0.copy()

        try:
            equilibria: list[Equilibrium] = []
            for record in self.history:
                self.replace_case(record.case_snapshot)
                self.config = record.config_snapshot
                equilibria.append(self.operator.build_equilibrium(record.result_snapshot.x))
            return equilibria
        finally:
            self.replace_case(saved_case)
            self.config = saved_config
            self.result = saved_result
            self.x0 = saved_x0
            self.operator(self.x0)

    def _resolve_solve_config(
        self,
        *,
        method: str | None,
        rtol: float | None,
        atol: float | None,
        root_maxiter: int | None,
        root_maxfev: int | None,
        enable_warmstart: bool | None,
        enable_homotopy: bool | None,
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
        if enable_homotopy is not None:
            overrides["enable_homotopy"] = bool(enable_homotopy)
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
    ) -> tuple[np.ndarray, bool, str, int, int, int, float]:
        """按主方法求解, 必要时回退到 `least_squares` 的 `lm` 和 `trf`."""

        attempts: list[tuple[str, tuple[np.ndarray, bool, str, int, int, int, float] | None, Exception | None]] = []

        primary_label = self._display_method_label(solve_config)
        primary, primary_exc = self._try_solve_attempt(x_guess, solve_config=solve_config)
        attempts.append((primary_label, primary, primary_exc))
        if self._attempt_succeeded(primary, primary_exc):
            if primary is None:
                raise RuntimeError("Primary solve attempt succeeded without a result")
            return primary

        if solve_config.method not in {"lm", "trf"}:
            primary_failure = self._format_attempt_failure(
                method=primary_label,
                result=primary,
                error=primary_exc,
            )
            warnings.warn(
                (f"Solve with method={primary_label!r} failed ({primary_failure}). Retrying with method='lm'."),
                RuntimeWarning,
                stacklevel=2,
            )
            lm_config = replace(solve_config, method="lm")
            lm_result, lm_exc = self._try_solve_attempt(x_guess, solve_config=lm_config)
            attempts.append((self._display_method_label(lm_config), lm_result, lm_exc))
            if self._attempt_succeeded(lm_result, lm_exc):
                return self._finalize_attempts(attempts)

        if solve_config.method != "trf" and attempts[-1][0] != self._display_method_label(
            replace(solve_config, method="trf")
        ):
            trf_label = self._display_method_label(replace(solve_config, method="trf"))
            last_label, last_result, last_exc = attempts[-1]
            last_failure = self._format_attempt_failure(
                method=last_label,
                result=last_result,
                error=last_exc,
            )
            warnings.warn(
                (f"Solve with method={last_label!r} still failed ({last_failure}). Retrying with {trf_label!r}."),
                RuntimeWarning,
                stacklevel=2,
            )
            trf_config = replace(solve_config, method="trf")
            trf_result, trf_exc = self._try_solve_attempt(x_guess, solve_config=trf_config)
            attempts.append((trf_label, trf_result, trf_exc))

        return self._finalize_attempts(attempts)

    def _try_solve_attempt(
        self,
        x_guess: np.ndarray,
        *,
        solve_config: SolverConfig,
    ) -> tuple[tuple[np.ndarray, bool, str, int, int, int, float] | None, Exception | None]:
        """包装一次 solve stage, 让 fallback 流程也能处理数值异常."""

        try:
            return self._solve_opt_problem(x_guess, solve_config=solve_config), None
        except Exception as exc:
            return (
                (
                    self.operator.coerce_x(x_guess).copy(),
                    False,
                    f"{type(exc).__name__}: {exc}",
                    0,
                    0,
                    0,
                    float("nan"),
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
        return error is None and attempt is not None and bool(attempt[1])

    def _display_method_label(self, solve_config: SolverConfig) -> str:
        if _uses_least_squares_api(solve_config):
            return f"least_squares/{solve_config.method}"
        return f"root/{solve_config.method}"

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
            f"attempt(method={label}) {'succeeded' if res is not None and res[1] and err is None else 'failed'}: "
            f"{self._format_attempt_failure(method=label, result=res, error=err)}"
            for label, res, err in attempts
        )
        return (
            selected_result[0],
            bool(selected_result[1]),
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
        """执行 full solve 或 homotopy solve."""

        stage_groups = self._build_stage_groups(solve_config=solve_config)
        run_full = self._run_solve_full
        run_masked = self._run_solve_masked
        if len(stage_groups) == 1 and _stage_group_is_full(stage_groups[0], self.operator.x_size):
            opt = run_full(x_guess, solve_config=solve_config)
            x_opt = self.operator.coerce_x(opt.x)
            residual_norm = _opt_residual_norm(opt)
            if residual_norm is None and not bool(opt.success):
                residual_norm, _ = self._safe_residual_norm(x_opt)
            accepted = bool(opt.success) or _residual_within_acceptance(residual_norm, solve_config)
            message = str(opt.message)
            if not bool(opt.success) and accepted:
                message = f"{message} [accepted by residual]"
            return (
                x_opt,
                accepted,
                message,
                _count_opt_attr(opt, "nfev"),
                _count_opt_attr(opt, "njev"),
                _count_opt_attr(opt, "nit"),
                float("nan") if residual_norm is None else float(residual_norm),
            )

        x_stage = x_guess.copy()
        active_indices = np.zeros(0, dtype=np.int64)
        total_nfev = 0
        total_njev = 0
        total_nit = 0
        stage_message = "homotopy continuation completed"
        last_opt = None
        truncation_state = self._init_homotopy_truncation_state()
        last_active_size = 0
        last_stage_residual_norm: float | None = None
        last_stage_accepted = False

        for stage_group in stage_groups:
            stage_new_indices = self._select_stage_indices(stage_group, truncation_state)
            if stage_new_indices.size == 0:
                continue
            active_indices = np.concatenate([active_indices, stage_new_indices])
            opt = run_masked(x_stage, active_indices=active_indices, solve_config=solve_config)
            x_stage[active_indices] = np.asarray(opt.x, dtype=np.float64)
            total_nfev += _count_opt_attr(opt, "nfev")
            total_njev += _count_opt_attr(opt, "njev")
            total_nit += _count_opt_attr(opt, "nit")
            last_opt = opt
            last_active_size = int(active_indices.shape[0])
            stage_message = f"{opt.message} [active_size={last_active_size}] [order={stage_group.order}]"
            stage_residual_norm = _opt_residual_norm(opt)
            if stage_residual_norm is None and not bool(opt.success):
                try:
                    stage_residual = self.operator.residual_masked(
                        x_stage[active_indices],
                        active_indices=active_indices,
                        x_template=x_stage,
                    )
                except Exception:
                    stage_residual_norm = None
                else:
                    stage_residual_norm = float(np.linalg.norm(stage_residual))
            accepted = bool(opt.success) or _residual_within_acceptance(stage_residual_norm, solve_config)
            last_stage_residual_norm = stage_residual_norm
            last_stage_accepted = bool(accepted)
            if not bool(opt.success) and accepted:
                stage_message = f"{stage_message} [accepted by residual]"
            if accepted:
                self._update_homotopy_truncation_state(
                    stage_group=stage_group,
                    x_stage=x_stage,
                    truncation_state=truncation_state,
                    solve_config=solve_config,
                )
            if active_indices.shape[0] == self.operator.x_size and accepted:
                break
            if not accepted:
                break

        if last_opt is None:
            raise RuntimeError("No solve stage was executed")

        residual_norm_final = float("nan")
        if last_stage_residual_norm is not None:
            residual_norm_final = float(last_stage_residual_norm)
        elif not bool(last_opt.success):
            try:
                stage_residual = self.operator.residual_masked(
                    x_stage[active_indices],
                    active_indices=active_indices,
                    x_template=x_stage,
                )
            except Exception:
                residual_norm_final = float("nan")
            else:
                residual_norm_final = float(np.linalg.norm(stage_residual))

        return (
            self.operator.coerce_x(x_stage),
            bool(last_stage_accepted),
            stage_message,
            total_nfev,
            total_njev,
            total_nit,
            residual_norm_final,
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

    def _run_solve_prefix(
        self,
        x_stage: np.ndarray,
        *,
        active_len: int,
        solve_config: SolverConfig,
    ):
        active_indices = np.arange(int(active_len), dtype=np.int64)
        return self._run_solve_masked(x_stage, active_indices=active_indices, solve_config=solve_config)

    def _run_solve_masked(
        self,
        x_stage: np.ndarray,
        *,
        active_indices: np.ndarray,
        solve_config: SolverConfig,
    ):
        if _uses_least_squares_api(solve_config):
            return self._run_least_squares_masked(x_stage, active_indices=active_indices, solve_config=solve_config)
        return self._run_root_masked(x_stage, active_indices=active_indices, solve_config=solve_config)

    def _run_root_full(
        self,
        x_guess: np.ndarray,
        *,
        solve_config: SolverConfig,
    ):
        """在完整 packed x 上调用一次 `scipy.optimize.root`."""

        return root(
            self.operator,
            x_guess,
            method=_root_method_name_for(solve_config),
            tol=solve_config.atol,
            options=_root_options_for(solve_config),
        )

    def _run_least_squares_full(
        self,
        x_guess: np.ndarray,
        *,
        solve_config: SolverConfig,
    ):
        """在完整 packed x 上调用一次 `scipy.optimize.least_squares`."""

        return least_squares(
            self.operator,
            x_guess,
            **_least_squares_kwargs_for(solve_config),
        )

    def _run_root_prefix(
        self,
        x_stage: np.ndarray,
        *,
        active_len: int,
        solve_config: SolverConfig,
    ):
        """在 prefix 子问题上调用 `root`."""

        active_indices = np.arange(int(active_len), dtype=np.int64)
        return self._run_root_masked(x_stage, active_indices=active_indices, solve_config=solve_config)

    def _run_root_masked(
        self,
        x_stage: np.ndarray,
        *,
        active_indices: np.ndarray,
        solve_config: SolverConfig,
    ):
        active_indices = np.asarray(active_indices, dtype=np.int64)
        x_active0 = np.asarray(x_stage[active_indices], dtype=np.float64).copy()
        x_template = self.operator.coerce_x(x_stage).copy()

        return root(
            lambda x_active: self.operator.residual_masked(
                x_active,
                active_indices=active_indices,
                x_template=x_template,
            ),
            x_active0,
            method=_root_method_name_for(solve_config),
            tol=solve_config.atol,
            options=_root_options_for(solve_config),
        )

    def _run_least_squares_prefix(
        self,
        x_stage: np.ndarray,
        *,
        active_len: int,
        solve_config: SolverConfig,
    ):
        """在 prefix 子问题上调用 `least_squares`."""

        active_indices = np.arange(int(active_len), dtype=np.int64)
        return self._run_least_squares_masked(x_stage, active_indices=active_indices, solve_config=solve_config)

    def _run_least_squares_masked(
        self,
        x_stage: np.ndarray,
        *,
        active_indices: np.ndarray,
        solve_config: SolverConfig,
    ):
        active_indices = np.asarray(active_indices, dtype=np.int64)
        x_active0 = np.asarray(x_stage[active_indices], dtype=np.float64).copy()
        x_template = self.operator.coerce_x(x_stage).copy()

        return least_squares(
            lambda x_active: self.operator.residual_masked(
                x_active,
                active_indices=active_indices,
                x_template=x_template,
            ),
            x_active0,
            **_least_squares_kwargs_for(solve_config),
        )

    def _build_stage_groups(
        self,
        *,
        solve_config: SolverConfig,
    ) -> list:
        x_size = self.operator.x_size
        if not solve_config.enable_homotopy:
            return [_full_stage_group(x_size)]

        groups = list(self.operator.homotopy_stage_groups())
        if not groups:
            return [_full_stage_group(x_size)]

        covered = np.concatenate([stage.indices for stage in groups]) if groups else np.zeros(0, dtype=np.int64)
        missing = np.setdiff1d(np.arange(x_size, dtype=np.int64), np.unique(covered), assume_unique=False)
        if missing.size:
            groups.append(type(groups[-1])(order=groups[-1].order + 1, indices=missing, shape_profile_ids=np.zeros(0, dtype=np.int64)))
        return groups

    def _init_homotopy_truncation_state(self) -> dict[str, np.ndarray]:
        profile_ids = self.operator.homotopy_truncation_profile_ids()
        shape = self.operator.profile_L.shape
        return {
            "profile_ids": profile_ids,
            "small_streak": np.zeros(shape, dtype=np.int64),
            "frozen_profile_mask": np.zeros(shape, dtype=bool),
        }

    def _select_stage_indices(self, stage_group, truncation_state: dict[str, np.ndarray]) -> np.ndarray:
        if int(stage_group.order) == 0:
            return np.asarray(stage_group.indices, dtype=np.int64)

        frozen_profile_mask = truncation_state["frozen_profile_mask"]
        indices: list[int] = []
        for profile_id in np.asarray(stage_group.shape_profile_ids, dtype=np.int64):
            if bool(frozen_profile_mask[int(profile_id)]):
                continue
            idx = int(self.operator.coeff_index[int(profile_id), int(stage_group.order)])
            if idx >= 0:
                indices.append(idx)
        return np.asarray(indices, dtype=np.int64)

    def _update_homotopy_truncation_state(
        self,
        *,
        stage_group,
        x_stage: np.ndarray,
        truncation_state: dict[str, np.ndarray],
        solve_config: SolverConfig,
    ) -> None:
        order = int(stage_group.order)
        profile_ids = truncation_state["profile_ids"]
        small_streak = truncation_state["small_streak"]
        frozen_profile_mask = truncation_state["frozen_profile_mask"]
        tol = float(solve_config.homotopy_truncation_tol)
        patience = int(solve_config.homotopy_truncation_patience)

        for profile_id in profile_ids:
            profile_id = int(profile_id)
            if bool(frozen_profile_mask[profile_id]):
                continue
            if int(self.operator.profile_L[profile_id]) < order:
                continue
            coeff_idx = int(self.operator.coeff_index[profile_id, order])
            if coeff_idx < 0:
                continue
            if abs(float(x_stage[coeff_idx])) < tol:
                small_streak[profile_id] += 1
            else:
                small_streak[profile_id] = 0
            if small_streak[profile_id] >= patience:
                frozen_profile_mask[profile_id] = True

    def _build_stage_lengths(
        self,
        *,
        solve_config: SolverConfig,
    ) -> list[int]:
        """生成 homotopy frontier 序列."""

        x_size = self.operator.x_size
        if not solve_config.enable_homotopy:
            return [x_size]

        frontiers = [int(frontier) for frontier in self.operator.homotopy_frontiers()]
        if not frontiers:
            return [x_size]

        if frontiers[-1] != x_size:
            frontiers.append(x_size)
        return frontiers


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


def _uses_root_api(solve_config: SolverConfig) -> bool:
    return solve_config.method in SUPPORTED_ROOT_METHODS


def _uses_least_squares_api(solve_config: SolverConfig) -> bool:
    return solve_config.method in SUPPORTED_LEAST_SQUARES_METHODS


def _full_stage_group(x_size: int):
    return SimpleNamespace(
        order=0,
        indices=np.arange(int(x_size), dtype=np.int64),
        shape_profile_ids=np.zeros(0, dtype=np.int64),
    )


def _stage_group_is_full(stage_group, x_size: int) -> bool:
    indices = np.asarray(stage_group.indices, dtype=np.int64)
    return indices.shape[0] == int(x_size) and np.array_equal(indices, np.arange(int(x_size), dtype=np.int64))


def _accepted_residual_norm(solve_config: SolverConfig) -> float:
    return max(float(solve_config.atol) * 10.0, 1.0e-5)


def _residual_within_acceptance(residual_norm: float | None, solve_config: SolverConfig) -> bool:
    return residual_norm is not None and np.isfinite(residual_norm) and residual_norm <= _accepted_residual_norm(
        solve_config
    )


def _root_method_name_for(solve_config: SolverConfig) -> str:
    if solve_config.method == "root-lm":
        return "lm"
    return solve_config.method
