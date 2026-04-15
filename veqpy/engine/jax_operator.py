"""
Module: engine.jax_operator

Role:
- 提供最小 JAX fused residual route 绑定.
- 当前只实现 `PF/rho/uniform` 的可行性路径.

Public API:
- bind_fused_residual_runner

Notes:
- 本模块当前只覆盖最小 JAX 里程碑, 其他 route 明确报不支持.
- profile/geometry hot refresh 仍暂时复用 numba reference path.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Callable

import jax
import jax.numpy as jnp
import numpy as np
from jax import lax

jax.config.update("jax_enable_x64", True)

if TYPE_CHECKING:
    from veqpy.engine.orchestration import SourcePlan
    from veqpy.operator.layouts import BackendState


_SUPPORTED_ROUTE = ("PF", "rho", "uniform")


def _jax_quadrature(arr: jnp.ndarray, weights: jnp.ndarray) -> jnp.ndarray:
    if arr.ndim == 1:
        return jnp.sum(arr * weights)
    return (2.0 * jnp.pi / arr.shape[1]) * jnp.sum(weights[:, None] * arr)


def _jax_corrected_integration(
    arr: jnp.ndarray,
    differentiation_matrix: jnp.ndarray,
    rho: jnp.ndarray,
    p: int,
) -> jnp.ndarray:
    rho_safe = jnp.where(rho > 1.0e-10, rho, 1.0e-10)
    q_int = arr / (rho_safe**p)
    system = rho[:, None] * differentiation_matrix + jnp.eye(rho.shape[0], dtype=arr.dtype) * float(p + 1)
    q_solution = jnp.linalg.solve(system, q_int)
    return q_solution * (rho ** (p + 1))


def _jax_enforce_axis_linear_psin_r(psin_r: jnp.ndarray, rho: jnp.ndarray) -> jnp.ndarray:
    if psin_r.shape[0] < 2:
        return psin_r

    def keep(arr: jnp.ndarray) -> jnp.ndarray:
        return arr

    def use_linear_fix(arr: jnp.ndarray) -> jnp.ndarray:
        if psin_r.shape[0] >= 3:

            def use_third_point(src: jnp.ndarray) -> jnp.ndarray:
                slope = src[2] / rho[2]
                return src.at[0].set(slope * rho[0]).at[1].set(slope * rho[1])

            def use_second_point(src: jnp.ndarray) -> jnp.ndarray:
                return src.at[0].set(src[1] * rho[0] / rho[1])

            return lax.cond(jnp.abs(rho[2]) >= 1.0e-14, use_third_point, use_second_point, arr)
        return arr.at[0].set(arr[1] * rho[0] / rho[1])

    return lax.cond(jnp.abs(rho[1]) < 1.0e-14, keep, use_linear_fix, psin_r)


def _jax_enforce_axis_even_profile(profile: jnp.ndarray, rho: jnp.ndarray) -> jnp.ndarray:
    if profile.shape[0] < 3:
        return profile
    x1 = rho[1] * rho[1]
    x2 = rho[2] * rho[2]

    def keep(arr: jnp.ndarray) -> jnp.ndarray:
        return arr

    def enforce(arr: jnp.ndarray) -> jnp.ndarray:
        slope = (arr[2] - arr[1]) / (x2 - x1)
        intercept = arr[1] - slope * x1
        arr = arr.at[0].set(intercept + slope * rho[0] * rho[0])
        arr = arr.at[1].set(intercept + slope * x1)
        return arr

    return lax.cond(jnp.abs(x2 - x1) < 1.0e-14, keep, enforce, profile)


def _jax_corrected_linear_derivative(
    arr: jnp.ndarray,
    differentiation_matrix: jnp.ndarray,
    rho: jnp.ndarray,
) -> jnp.ndarray:
    if arr.shape[0] == 0:
        return arr
    if arr.shape[0] == 1:
        return jnp.zeros_like(arr)
    reduced = jnp.where(rho > 1.0e-10, arr / rho, 0.0)
    reduced = reduced.at[0].set(reduced[1])
    reduced = _jax_enforce_axis_even_profile(reduced, rho)
    reduced_r = differentiation_matrix @ reduced
    reduced_r = _jax_enforce_axis_linear_psin_r(reduced_r, rho)
    out = reduced + rho * reduced_r
    out = out.at[0].set(reduced[0])
    return out


def _jax_update_psin_coordinate(
    psin_r: jnp.ndarray,
    differentiation_matrix: jnp.ndarray,
    rho: jnp.ndarray,
) -> jnp.ndarray:
    psin = _jax_corrected_integration(psin_r, differentiation_matrix, rho, 2)
    offset = psin[0]
    scale = psin[-1] - offset
    scale = jnp.where(jnp.abs(scale) < 1.0e-12, 1.0, scale)
    psin = (psin - offset) / scale
    psin = psin.at[0].set(0.0)
    psin = psin.at[-1].set(1.0)
    return psin


def _jax_fill_scaled_ratio(num: jnp.ndarray, den: jnp.ndarray, scale: float) -> jnp.ndarray:
    return scale * num / den


def _jax_compute_Pn(Pn_r: jnp.ndarray, integration_matrix: jnp.ndarray, weights: jnp.ndarray) -> jnp.ndarray:
    return integration_matrix @ Pn_r - _jax_quadrature(Pn_r, weights)


def _slot_for_view(active_u_fields: np.ndarray, target_fields: np.ndarray) -> int:
    for slot in range(active_u_fields.shape[0]):
        if np.shares_memory(active_u_fields[slot], target_fields):
            return slot
    return -1


def _select_hot_profile_fields(
    refreshed_active_u_fields: jnp.ndarray,
    *,
    slot: int,
    fallback_fields: jnp.ndarray,
) -> jnp.ndarray:
    if slot >= 0:
        return refreshed_active_u_fields[slot]
    return fallback_fields


def _jax_update_profiles_packed_bulk(
    x: jnp.ndarray,
    *,
    T_fields: jnp.ndarray,
    active_rp_fields: jnp.ndarray,
    active_env_fields: jnp.ndarray,
    active_offsets: jnp.ndarray,
    active_scales: jnp.ndarray,
    active_coeff_index_rows: jnp.ndarray,
    active_lengths: jnp.ndarray,
) -> jnp.ndarray:
    max_len = active_coeff_index_rows.shape[1]
    coeff_mask = (jnp.arange(max_len)[None, :] < active_lengths[:, None]).astype(jnp.float64)
    coeff_index_rows = jnp.maximum(active_coeff_index_rows, 0)
    coeff = x[coeff_index_rows] * coeff_mask
    series = jnp.einsum("pk,dkn->pdn", coeff, T_fields[:, :max_len, :])
    env = active_env_fields
    rp = active_rp_fields
    base = env[:, 0] * series[:, 0]
    base_r = env[:, 1] * series[:, 0] + env[:, 0] * series[:, 1]
    base_rr = env[:, 2] * series[:, 0] + 2.0 * env[:, 1] * series[:, 1] + env[:, 0] * series[:, 2]
    amp = active_offsets[:, None] + base
    out0 = active_scales[:, None] * (rp[:, 0] * amp)
    out1 = active_scales[:, None] * (rp[:, 1] * amp + rp[:, 0] * base_r)
    out2 = active_scales[:, None] * (rp[:, 2] * amp + 2.0 * rp[:, 1] * base_r + rp[:, 0] * base_rr)
    return jnp.stack((out0, out1, out2), axis=1)


def _jax_apply_f2_profile_fields(fields: jnp.ndarray, eps: float = 1.0e-10) -> jnp.ndarray:
    F2 = jnp.maximum(fields[0], eps)
    F2_r = fields[1]
    F2_rr = fields[2]
    F = jnp.sqrt(F2)
    inv_F = 1.0 / F
    inv_F3 = inv_F / F2
    return jnp.stack(
        (
            F,
            0.5 * F2_r * inv_F,
            0.5 * F2_rr * inv_F - 0.25 * F2_r * F2_r * inv_F3,
        ),
        axis=0,
    )


def _jax_update_fourier_family_fields(
    *,
    base_c_fields: jnp.ndarray,
    base_s_fields: jnp.ndarray,
    active_u_fields: jnp.ndarray,
    c_source_slots: jnp.ndarray,
    s_source_slots: jnp.ndarray,
    c_active_order: int,
    s_active_order: int,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    c_fields = jnp.array(base_c_fields)
    s_fields = jnp.array(base_s_fields)
    c_orders = c_fields.shape[0]
    s_orders = s_fields.shape[0]

    for order in range(c_orders):
        if order <= c_active_order:
            slot = int(c_source_slots[order])
            c_fields = c_fields.at[order].set(active_u_fields[slot] if slot >= 0 else base_c_fields[order])
        else:
            c_fields = c_fields.at[order].set(jnp.zeros_like(c_fields[order]))

    s_fields = s_fields.at[0].set(base_s_fields[0])
    for order in range(1, s_orders):
        if order <= s_active_order:
            slot = int(s_source_slots[order])
            s_fields = s_fields.at[order].set(active_u_fields[slot] if slot >= 0 else base_s_fields[order])
        else:
            s_fields = s_fields.at[order].set(jnp.zeros_like(s_fields[order]))

    return c_fields, s_fields


def _jax_update_geometry_hot(
    *,
    a: float,
    R0: float,
    rho: jnp.ndarray,
    theta: jnp.ndarray,
    cos_ktheta: jnp.ndarray,
    sin_ktheta: jnp.ndarray,
    k_cos_ktheta: jnp.ndarray,
    k_sin_ktheta: jnp.ndarray,
    k2_cos_ktheta: jnp.ndarray,
    k2_sin_ktheta: jnp.ndarray,
    h_fields: jnp.ndarray,
    v_fields: jnp.ndarray,
    k_fields: jnp.ndarray,
    c_fields: jnp.ndarray,
    s_fields: jnp.ndarray,
    c_active_order: int,
    s_active_order: int,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    nr = rho.shape[0]
    nt = theta.shape[0]
    theta_scale = 2.0 * np.pi / nt
    mean_scale = 1.0 / nt
    two_pi = 2.0 * np.pi
    rho2d = rho[:, None]
    theta2d = theta[None, :]
    sin_t = sin_ktheta[1][None, :]
    cos_t = cos_ktheta[1][None, :]

    c_limit = min(c_active_order + 1, c_fields.shape[0], cos_ktheta.shape[0])
    s_limit = min(s_active_order + 1, s_fields.shape[0], sin_ktheta.shape[0])
    c0_i = c_fields[0, 0][:, None]
    c0_r_i = c_fields[0, 1][:, None]
    c0_rr_i = c_fields[0, 2][:, None]

    tb = theta2d + c0_i
    tb_r = jnp.broadcast_to(c0_r_i, (nr, nt))
    tb_t = jnp.ones((nr, nt), dtype=jnp.float64)
    tb_rr = jnp.broadcast_to(c0_rr_i, (nr, nt))
    tb_rt = jnp.zeros((nr, nt), dtype=jnp.float64)
    tb_tt = jnp.zeros((nr, nt), dtype=jnp.float64)

    if c_limit > 1:
        c_slice = c_fields[1:c_limit]
        tb = tb + jnp.einsum("on,ot->nt", c_slice[:, 0], cos_ktheta[1:c_limit])
        tb_r = tb_r + jnp.einsum("on,ot->nt", c_slice[:, 1], cos_ktheta[1:c_limit])
        tb_t = tb_t - jnp.einsum("on,ot->nt", c_slice[:, 0], k_sin_ktheta[1:c_limit])
        tb_rr = tb_rr + jnp.einsum("on,ot->nt", c_slice[:, 2], cos_ktheta[1:c_limit])
        tb_rt = tb_rt - jnp.einsum("on,ot->nt", c_slice[:, 1], k_sin_ktheta[1:c_limit])
        tb_tt = tb_tt - jnp.einsum("on,ot->nt", c_slice[:, 0], k2_cos_ktheta[1:c_limit])

    if s_limit > 1:
        s_slice = s_fields[1:s_limit]
        tb = tb + jnp.einsum("on,ot->nt", s_slice[:, 0], sin_ktheta[1:s_limit])
        tb_r = tb_r + jnp.einsum("on,ot->nt", s_slice[:, 1], sin_ktheta[1:s_limit])
        tb_t = tb_t + jnp.einsum("on,ot->nt", s_slice[:, 0], k_cos_ktheta[1:s_limit])
        tb_rr = tb_rr + jnp.einsum("on,ot->nt", s_slice[:, 2], sin_ktheta[1:s_limit])
        tb_rt = tb_rt + jnp.einsum("on,ot->nt", s_slice[:, 1], k_cos_ktheta[1:s_limit])
        tb_tt = tb_tt - jnp.einsum("on,ot->nt", s_slice[:, 0], k2_sin_ktheta[1:s_limit])

    cos_tb = jnp.cos(tb)
    sin_tb = jnp.sin(tb)

    h_i = h_fields[0][:, None]
    h_r_i = h_fields[1][:, None]
    h_rr_i = h_fields[2][:, None]
    v_r_i = v_fields[1][:, None]
    v_rr_i = v_fields[2][:, None]
    k_i = k_fields[0][:, None]
    k_r_i = k_fields[1][:, None]
    k_rr_i = k_fields[2][:, None]

    R_surface = jnp.maximum(R0 + a * (h_i + rho2d * cos_tb), 1.0e-15)
    R_r = a * (h_r_i + cos_tb - rho2d * sin_tb * tb_r)
    R_t = -a * rho2d * sin_tb * tb_t
    R_rr = a * (h_rr_i - 2.0 * sin_tb * tb_r - rho2d * (cos_tb * tb_r * tb_r + sin_tb * tb_rr))
    R_rt = -a * (sin_tb * tb_t + rho2d * (cos_tb * tb_r * tb_t + sin_tb * tb_rt))
    R_tt = -a * rho2d * (cos_tb * tb_t * tb_t + sin_tb * tb_tt)

    Z_r = a * (v_r_i - (k_i + rho2d * k_r_i) * sin_t)
    Z_t = -a * rho2d * k_i * cos_t
    Z_rr = a * (v_rr_i - (2.0 * k_r_i + rho2d * k_rr_i) * sin_t)
    Z_rt = -a * (k_i + rho2d * k_r_i) * cos_t
    Z_tt = a * rho2d * k_i * sin_t

    J_surface = jnp.maximum(R_t * Z_r - R_r * Z_t, 1.0e-15)
    J_r = -(R_rr * Z_t - R_rt * Z_r + R_r * Z_rt - R_t * Z_rr)
    J_t = -(R_rt * Z_t - R_tt * Z_r + R_r * Z_tt - R_t * Z_rt)
    JR = J_surface * R_surface
    JR_r = J_r * R_surface + J_surface * R_r
    JR_t = J_t * R_surface + J_surface * R_t
    JdivR = J_surface / R_surface

    grt = R_r * R_t + Z_r * Z_t
    grt_t = R_rt * R_t + R_r * R_tt + Z_rt * Z_t + Z_r * Z_tt
    gtt = R_t * R_t + Z_t * Z_t
    gtt_r = 2.0 * (R_t * R_rt + Z_t * Z_rt)
    inv_JR = 1.0 / JR
    grtdivJR_t = (grt_t - grt * JR_t * inv_JR) * inv_JR
    gttdivJR = gtt * inv_JR
    gttdivJR_r = gtt_r * inv_JR - gtt * JR_r * inv_JR * inv_JR

    S_r = jnp.sum(J_surface, axis=1) * theta_scale
    V_r = jnp.sum(JR, axis=1) * theta_scale * two_pi
    Kn = jnp.sum(gttdivJR, axis=1) * mean_scale
    Kn_r = jnp.sum(gttdivJR_r, axis=1) * mean_scale
    Ln_r = jnp.sum(JdivR, axis=1) * mean_scale

    surface_workspace = jnp.stack((sin_tb, R_surface, R_t, Z_t, J_surface, JdivR, grtdivJR_t, gttdivJR, gttdivJR_r))
    radial_workspace = jnp.stack((S_r, V_r, Kn, Kn_r, Ln_r))
    return surface_workspace, radial_workspace


def _jax_pf_rho_source_eval(
    *,
    radial_workspace: jnp.ndarray,
    surface_workspace: jnp.ndarray,
    heat_input: jnp.ndarray,
    current_input: jnp.ndarray,
    weights: jnp.ndarray,
    differentiation_matrix: jnp.ndarray,
    integration_matrix: jnp.ndarray,
    rho: jnp.ndarray,
    B0: float,
    Ip: float,
    beta: float,
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    V_r = radial_workspace[1]
    Kn = radial_workspace[2]
    Ln_r = radial_workspace[4]
    R_surface = surface_workspace[1]
    JdivR = surface_workspace[5]

    has_Ip = not np.isnan(Ip)
    has_beta = not np.isnan(beta)
    pressure_factor = 1.0 / (4.0 * np.pi**2)
    integrand = Kn * (current_input * Ln_r + V_r * heat_input * pressure_factor)
    psin_r = -2.0 * _jax_corrected_integration(integrand, differentiation_matrix, rho, 1)
    psin_r = jnp.where(psin_r < 0.0, 0.0, psin_r)
    psin_r = jnp.sqrt(psin_r)
    psin_r = psin_r / Kn
    psin_r = _jax_enforce_axis_linear_psin_r(psin_r, rho)

    integral_prof = _jax_quadrature(psin_r, weights)
    integral_prof_safe = jnp.where(jnp.abs(integral_prof) < 1.0e-12, 1.0, integral_prof)
    psin_r = psin_r / integral_prof_safe
    psin_rr = _jax_corrected_linear_derivative(psin_r, differentiation_matrix, rho)
    psin = _jax_update_psin_coordinate(psin_r, differentiation_matrix, rho)
    psin_r_safe = jnp.maximum(psin_r, 1.0e-10)

    if (not has_Ip) and (not has_beta):
        alpha2 = integral_prof_safe
        alpha1 = -_jax_quadrature(heat_input, weights) / alpha2
        Pn_psin = _jax_fill_scaled_ratio(heat_input, psin_r_safe, 1.0 / (alpha1 * alpha2))
        FFn_psin = _jax_fill_scaled_ratio(current_input, psin_r_safe, 1.0 / (alpha1 * alpha2))
        return psin, psin_r, psin_rr, FFn_psin, Pn_psin, alpha1, alpha2

    c2 = integral_prof * integral_prof
    if has_Ip and (not has_beta):
        g1n_integrand = (
            JdivR * (current_input[:, None] + R_surface * R_surface * heat_input[:, None]) / psin_r_safe[:, None]
        )
        alpha1 = -Ip / _jax_quadrature(g1n_integrand, weights)
    elif has_beta and (not has_Ip):
        Pn = _jax_compute_Pn(heat_input, integration_matrix, weights)
        c1 = 0.5 * beta * B0 * B0 * _jax_quadrature(V_r, weights) / _jax_quadrature(Pn * V_r, weights)
        alpha1 = jnp.sqrt(c1 / c2)
    else:
        raise ValueError("PF does not support applying Ip and beta constraints simultaneously")

    alpha2 = c2 * alpha1
    Pn_psin = _jax_fill_scaled_ratio(heat_input, psin_r_safe, 1.0)
    FFn_psin = _jax_fill_scaled_ratio(current_input, psin_r_safe, 1.0)
    return psin, psin_r, psin_rr, FFn_psin, Pn_psin, alpha1, alpha2


def _jax_update_residual_workspace(
    alpha1: float,
    alpha2: float,
    root_fields: jnp.ndarray,
    geometry_surface_workspace: jnp.ndarray,
) -> jnp.ndarray:
    sin_tb_surface = geometry_surface_workspace[0]
    R_surface = geometry_surface_workspace[1]
    R_t_surface = geometry_surface_workspace[2]
    Z_t_surface = geometry_surface_workspace[3]
    J_surface = geometry_surface_workspace[4]
    JdivR_surface = geometry_surface_workspace[5]
    grtdivJR_t_surface = geometry_surface_workspace[6]
    gttdivJR_surface = geometry_surface_workspace[7]
    gttdivJR_r_surface = geometry_surface_workspace[8]

    psin_r = root_fields[1][:, None]
    psin_rr = root_fields[2][:, None]
    FFn_psin = root_fields[3][:, None]
    Pn_psin = root_fields[4][:, None]

    inv_J = 1.0 / J_surface
    psin_R = -Z_t_surface * inv_J * psin_r
    psin_Z = R_t_surface * inv_J * psin_r
    G1n = JdivR_surface * (FFn_psin + R_surface * R_surface * Pn_psin)
    G2n = gttdivJR_surface * psin_rr + (gttdivJR_r_surface - grtdivJR_t_surface) * psin_r
    G = alpha1 * G1n + alpha2 * G2n
    Gpsin_R = G * psin_R
    Gpsin_Z = G * psin_Z
    Gpsin_R_sin_tb = Gpsin_R * sin_tb_surface
    return jnp.stack((G, Gpsin_R, Gpsin_Z, Gpsin_R_sin_tb), axis=0)


def _jax_project_rows_to_packed(
    packed: jnp.ndarray,
    coeff_indices: np.ndarray,
    T: jnp.ndarray,
    weighted_rho: jnp.ndarray,
) -> jnp.ndarray:
    values = T[: coeff_indices.shape[0], :] @ weighted_rho
    return packed.at[jnp.asarray(coeff_indices)].set(values)


def _jax_pack_residual_output(
    residual_workspace: jnp.ndarray,
    *,
    active_residual_block_codes: np.ndarray,
    active_residual_block_orders: np.ndarray,
    active_coeff_index_rows: np.ndarray,
    active_lengths: np.ndarray,
    sin_ktheta: jnp.ndarray,
    cos_ktheta: jnp.ndarray,
    rho_powers: jnp.ndarray,
    y: jnp.ndarray,
    T_fields: jnp.ndarray,
    weights: jnp.ndarray,
    a: float,
    R0: float,
    B0: float,
    packed_size: int,
) -> jnp.ndarray:
    G = residual_workspace[0]
    Gpsin_R = residual_workspace[1]
    Gpsin_Z = residual_workspace[2]
    Gpsin_R_sin_tb = residual_workspace[3]
    T = T_fields[0]
    sin_theta = sin_ktheta[1]
    rho = rho_powers[1]
    rho2 = rho_powers[2]
    nt = G.shape[1]
    base_scale = 2.0 * np.pi / nt
    packed = jnp.zeros((packed_size,), dtype=jnp.float64)

    for slot, code in enumerate(active_residual_block_codes):
        length = int(active_lengths[slot])
        coeff_indices = active_coeff_index_rows[slot, :length]
        order = int(active_residual_block_orders[slot])
        if code == 0:
            scratch = jnp.sum(Gpsin_R, axis=1)
            weighted = scratch * y * weights * (base_scale * a)
        elif code == 1:
            scratch = jnp.sum(Gpsin_Z, axis=1)
            weighted = scratch * y * weights * (base_scale * a)
        elif code == 2:
            scratch = jnp.sum(Gpsin_Z * sin_theta[None, :], axis=1)
            weighted = scratch * rho * y * weights * (base_scale * (-a))
        elif code == 3:
            scratch = jnp.sum(Gpsin_R_sin_tb, axis=1)
            weighted = scratch * rho * y * weights * (base_scale * (-a))
        elif code == 4:
            scratch = jnp.sum(Gpsin_R_sin_tb * cos_ktheta[order][None, :], axis=1)
            weighted = scratch * rho_powers[order + 1] * y * weights * (base_scale * (-a))
        elif code == 5:
            scratch = jnp.sum(Gpsin_R_sin_tb * sin_ktheta[order][None, :], axis=1)
            weighted = scratch * rho_powers[order + 1] * y * weights * (base_scale * (-a))
        elif code == 6:
            scratch = jnp.sum(G, axis=1)
            weighted = scratch * rho2 * y * weights * base_scale
        elif code == 7:
            scratch = jnp.sum(G, axis=1)
            weighted = scratch * y * y * weights * (base_scale * (R0 * B0) * (R0 * B0))
        else:
            raise ValueError("Unknown residual block code")
        packed = _jax_project_rows_to_packed(packed, coeff_indices, T, weighted)
    return packed


def _build_pf_rho_uniform_kernel(
    *,
    source_plan: "SourcePlan",
    backend_state: "BackendState",
    c_active_order: int,
    s_active_order: int,
    a: float,
    R0: float,
    B0: float,
) -> Callable[[jnp.ndarray], tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]]:
    static_layout = backend_state.static_layout
    residual_binding_layout = backend_state.residual_binding_layout
    runtime_layout = backend_state.runtime_layout
    source_work_state = backend_state.source_runtime_state.work_state
    active_u_fields = runtime_layout.active_u_fields
    heat_input = jnp.asarray(source_work_state.materialized_heat_input, dtype=jnp.float64)
    current_input = jnp.asarray(source_work_state.materialized_current_input, dtype=jnp.float64)
    weights = jnp.asarray(static_layout.weights, dtype=jnp.float64)
    differentiation_matrix = jnp.asarray(static_layout.differentiation_matrix, dtype=jnp.float64)
    integration_matrix = jnp.asarray(static_layout.integration_matrix, dtype=jnp.float64)
    rho = jnp.asarray(static_layout.rho, dtype=jnp.float64)
    sin_ktheta = jnp.asarray(static_layout.sin_ktheta, dtype=jnp.float64)
    cos_ktheta = jnp.asarray(static_layout.cos_ktheta, dtype=jnp.float64)
    rho_powers = jnp.asarray(static_layout.rho_powers, dtype=jnp.float64)
    y = jnp.asarray(static_layout.y, dtype=jnp.float64)
    T_fields = jnp.asarray(static_layout.T_fields, dtype=jnp.float64)
    theta = jnp.asarray(static_layout.theta, dtype=jnp.float64)
    k_cos_ktheta = jnp.asarray(static_layout.k_cos_ktheta, dtype=jnp.float64)
    k_sin_ktheta = jnp.asarray(static_layout.k_sin_ktheta, dtype=jnp.float64)
    k2_cos_ktheta = jnp.asarray(static_layout.k2_cos_ktheta, dtype=jnp.float64)
    k2_sin_ktheta = jnp.asarray(static_layout.k2_sin_ktheta, dtype=jnp.float64)
    active_rp_fields = jnp.asarray(runtime_layout.active_rp_fields, dtype=jnp.float64)
    active_env_fields = jnp.asarray(runtime_layout.active_env_fields, dtype=jnp.float64)
    active_offsets = jnp.asarray(runtime_layout.active_offsets, dtype=jnp.float64)
    active_scales = jnp.asarray(runtime_layout.active_scales, dtype=jnp.float64)
    active_coeff_index_rows = jnp.asarray(runtime_layout.active_coeff_index_rows, dtype=jnp.int64)
    active_lengths = jnp.asarray(runtime_layout.active_lengths, dtype=jnp.int64)
    c_family_base_fields = jnp.asarray(runtime_layout.c_family_base_fields, dtype=jnp.float64)
    s_family_base_fields = jnp.asarray(runtime_layout.s_family_base_fields, dtype=jnp.float64)
    c_family_source_slots = np.asarray(runtime_layout.c_family_source_slots, dtype=np.int64)
    s_family_source_slots = np.asarray(runtime_layout.s_family_source_slots, dtype=np.int64)
    h_fallback_fields = jnp.asarray(runtime_layout.h_fields, dtype=jnp.float64)
    v_fallback_fields = jnp.asarray(runtime_layout.v_fields, dtype=jnp.float64)
    k_fallback_fields = jnp.asarray(runtime_layout.k_fields, dtype=jnp.float64)
    h_slot = _slot_for_view(active_u_fields, runtime_layout.h_fields)
    v_slot = _slot_for_view(active_u_fields, runtime_layout.v_fields)
    k_slot = _slot_for_view(active_u_fields, runtime_layout.k_fields)
    F_slot = _slot_for_view(active_u_fields, runtime_layout.F_profile_fields)
    Ip = float(source_plan.Ip)
    beta = float(source_plan.beta)
    zero_heat = bool(np.all(np.abs(np.asarray(source_work_state.materialized_heat_input, dtype=np.float64)) <= 1.0e-14))
    zero_current = bool(
        np.all(np.abs(np.asarray(source_work_state.materialized_current_input, dtype=np.float64)) <= 1.0e-14)
    )
    use_trivial_source = np.isnan(Ip) and np.isnan(beta) and zero_heat and zero_current
    active_residual_block_codes = np.asarray(residual_binding_layout.active_residual_block_codes, dtype=np.int64)
    active_residual_block_orders = np.asarray(residual_binding_layout.active_residual_block_orders, dtype=np.int64)
    active_coeff_index_rows = np.asarray(runtime_layout.active_coeff_index_rows, dtype=np.int64)
    active_lengths = np.asarray(runtime_layout.active_lengths, dtype=np.int64)
    packed_size = int(runtime_layout.packed_residual.shape[0])

    @jax.jit
    def kernel(x: jnp.ndarray):
        refreshed_active_u_fields = _jax_update_profiles_packed_bulk(
            x,
            T_fields=T_fields,
            active_rp_fields=active_rp_fields,
            active_env_fields=active_env_fields,
            active_offsets=active_offsets,
            active_scales=active_scales,
            active_coeff_index_rows=active_coeff_index_rows,
            active_lengths=active_lengths,
        )
        if F_slot >= 0:
            refreshed_active_u_fields = refreshed_active_u_fields.at[F_slot].set(
                _jax_apply_f2_profile_fields(refreshed_active_u_fields[F_slot])
            )
        c_fields, s_fields = _jax_update_fourier_family_fields(
            base_c_fields=c_family_base_fields,
            base_s_fields=s_family_base_fields,
            active_u_fields=refreshed_active_u_fields,
            c_source_slots=c_family_source_slots,
            s_source_slots=s_family_source_slots,
            c_active_order=c_active_order,
            s_active_order=s_active_order,
        )
        surface_workspace, radial_workspace = _jax_update_geometry_hot(
            a=a,
            R0=R0,
            rho=rho,
            theta=theta,
            cos_ktheta=cos_ktheta,
            sin_ktheta=sin_ktheta,
            k_cos_ktheta=k_cos_ktheta,
            k_sin_ktheta=k_sin_ktheta,
            k2_cos_ktheta=k2_cos_ktheta,
            k2_sin_ktheta=k2_sin_ktheta,
            h_fields=_select_hot_profile_fields(
                refreshed_active_u_fields, slot=h_slot, fallback_fields=h_fallback_fields
            ),
            v_fields=_select_hot_profile_fields(
                refreshed_active_u_fields, slot=v_slot, fallback_fields=v_fallback_fields
            ),
            k_fields=_select_hot_profile_fields(
                refreshed_active_u_fields, slot=k_slot, fallback_fields=k_fallback_fields
            ),
            c_fields=c_fields,
            s_fields=s_fields,
            c_active_order=c_active_order,
            s_active_order=s_active_order,
        )
        if use_trivial_source:
            root_fields = jnp.stack(
                (
                    rho,
                    jnp.ones_like(rho),
                    jnp.zeros_like(rho),
                    jnp.zeros_like(rho),
                    jnp.zeros_like(rho),
                ),
                axis=0,
            )
            alpha1 = 0.0
            alpha2 = 0.0
        else:
            psin, psin_r, psin_rr, FFn_psin, Pn_psin, alpha1, alpha2 = _jax_pf_rho_source_eval(
                radial_workspace=radial_workspace,
                surface_workspace=surface_workspace,
                heat_input=heat_input,
                current_input=current_input,
                weights=weights,
                differentiation_matrix=differentiation_matrix,
                integration_matrix=integration_matrix,
                rho=rho,
                B0=B0,
                Ip=Ip,
                beta=beta,
            )
            root_fields = jnp.stack((psin, psin_r, psin_rr, FFn_psin, Pn_psin), axis=0)
        residual_workspace = _jax_update_residual_workspace(alpha1, alpha2, root_fields, surface_workspace)
        packed_residual = _jax_pack_residual_output(
            residual_workspace,
            active_residual_block_codes=active_residual_block_codes,
            active_residual_block_orders=active_residual_block_orders,
            active_coeff_index_rows=active_coeff_index_rows,
            active_lengths=active_lengths,
            sin_ktheta=sin_ktheta,
            cos_ktheta=cos_ktheta,
            rho_powers=rho_powers,
            y=y,
            T_fields=T_fields,
            weights=weights,
            a=a,
            R0=R0,
            B0=B0,
            packed_size=packed_size,
        )
        return packed_residual, alpha1, alpha2

    return kernel


def bind_fused_residual_runner(
    *,
    source_plan: "SourcePlan",
    backend_state: "BackendState",
    alpha_state: np.ndarray,
    c_active_order: int,
    s_active_order: int,
    a: float,
    R0: float,
    Z0: float,
    B0: float,
) -> Callable[[np.ndarray], np.ndarray]:
    route_key = (source_plan.route, source_plan.coordinate, source_plan.nodes)
    if route_key != _SUPPORTED_ROUTE:
        raise NotImplementedError(f"JAX backend currently supports only {_SUPPORTED_ROUTE!r}, got {route_key!r}")

    kernel = _build_pf_rho_uniform_kernel(
        source_plan=source_plan,
        backend_state=backend_state,
        c_active_order=c_active_order,
        s_active_order=s_active_order,
        a=a,
        R0=R0,
        B0=B0,
    )

    def runner(x: np.ndarray) -> np.ndarray:
        packed_residual_dev, alpha1_dev, alpha2_dev = kernel(jnp.asarray(x, dtype=jnp.float64))
        alpha1 = float(alpha1_dev)
        alpha2 = float(alpha2_dev)
        alpha_state[0] = alpha1
        alpha_state[1] = alpha2
        return np.asarray(packed_residual_dev, dtype=np.float64)

    return runner
