import json

import numpy as np
import pandas as pd
from scipy import interpolate

from quasi_steady import quasi_steady_flow_angle


def run_iag(k, chord, inflow_speed, amplitude, mean, period_res, n_periods, file_polar="S801.dat"):
    pitch_around = 0.5
    alpha_at = 0.75

    overall_res = n_periods * period_res
    omega = 2 * k * inflow_speed / chord
    T = 2 * np.pi / omega
    dt = T / period_res
    t = dt * np.arange(overall_res)

    # --- SETUP ---
    alpha = np.deg2rad(mean + amplitude * np.sin(omega * t))
    alpha_speed = np.deg2rad(amplitude * omega * np.cos(omega * t))
    pos = np.c_[np.zeros((overall_res, 2)), -alpha]
    vel = np.c_[np.zeros((overall_res, 2)), -alpha_speed]
    # inflow = inflow_speed * np.c_[0.9 * np.ones_like(t), 0.1 * np.ones_like(t)]
    inflow = inflow_speed * np.c_[1 * np.ones_like(t), 0.0 * np.ones_like(t)]
    inflow_angle = np.arctan(inflow[:, 1] / inflow[:, 0])

    # --- PREALLOCATE ARRAYS ---
    N = t.size
    alpha_steady = np.zeros(N)
    inflow_angle = np.zeros(N)
    alpha_qs = np.zeros(N)
    ds = np.zeros(N)
    X_lag = np.zeros(N)
    Y_lag = np.zeros(N)
    alpha_eff = np.zeros(N)
    C_nc = np.zeros(N)
    D_i = np.zeros(N)
    C_ni = np.zeros(N)
    C_npot = np.zeros(N)
    D_p = np.zeros(N)
    C_nsEq = np.zeros(N)
    alpha_sEq = np.zeros(N)
    D_bl_n = np.zeros(N)
    f_n = np.zeros(N)
    f_n_Dp = np.zeros(N)
    C_nvisc = np.zeros(N)
    C_nf = np.zeros(N)
    C_tf = np.zeros(N)
    tau_vortex = np.zeros(N)
    C_nv_instant = np.zeros(N)
    C_nv = np.zeros(N)
    C_mf = np.zeros(N)
    C_mV = np.zeros(N)
    C_mC = np.zeros(N)
    C_dus = np.zeros(N)
    C_lus = np.zeros(N)
    C_mus = np.zeros(N)

    file_polar_values = "polar_values.json"
    with open(file_polar_values, "r") as ff_aero:
        aero_values = json.load(ff_aero)

    alpha_0n_visc = aero_values["C_n_visc_root"]
    alpha_0n_inv = aero_values["C_n_inv_root"]
    C_n_slope = aero_values["C_n_inv_max_slope"]
    alpha_crit = aero_values["alpha_critical"]
    C_n_crit = C_n_slope * (alpha_crit - alpha_0n_inv)

    file_polar_characteristics = "polar_characteristics.csv"
    df_polar_chars = pd.read_csv(file_polar_characteristics)
    f_n_lookup = interpolate.interp1d(df_polar_chars["alpha"], df_polar_chars["f_n"])
    C_n_visc = interpolate.interp1d(df_polar_chars["alpha"], df_polar_chars["C_n_visc"])
    C_t_visc = interpolate.interp1d(df_polar_chars["alpha"], df_polar_chars["C_t_visc"])
    C_d_polar = interpolate.interp1d(df_polar_chars["alpha"], df_polar_chars["C_d"])
    C_m_polar = interpolate.interp1d(df_polar_chars["alpha"], df_polar_chars["C_m"])

    def BL_first_order_IAG2(
        i: int,
        A1: float,
        A2: float,
        b1: float,
        b2: float,
        chord: float = chord,
        pitching_around: float = pitch_around,
        alpha_at: float = alpha_at,
        a: float = 343,
        K_alpha: float = 0.75,
        K_fC: float = 0.1,
        K_v: float = 0.2,
        T_p: float = 1.7,
        T_bl: float = 3,
        T_v: float = 6,
        T_MU: float = 1.5,
        T_MD: float = 1.5,
        tau_vortex_pure_decay: float = 6,
        aoa_crit: float = alpha_crit,
    ):
        alpha_steady[i] = -pos[i, 2] + inflow_angle[i]
        # --------- general calculations
        qs_flow_angle, v_x, v_y = quasi_steady_flow_angle(
            vel[i, :], pos[i, :], inflow[i, :], chord, pitching_around, alpha_at
        )
        alpha_qs[i] = -pos[i, 2] + qs_flow_angle
        rel_flow_vel = np.sqrt(v_x**2 + v_y**2)
        ds[i] = 2 * dt * rel_flow_vel / chord

        # --------- MODULE unsteady attached flow
        d_alpha_qs = alpha_qs[i] - alpha_qs[i - 1]
        # d_alpha_qs = alpha_steady[i]-alpha_steady[i-1]
        X_lag[i] = X_lag[i - 1] * np.exp(-b1 * ds[i - 1]) + d_alpha_qs * A1 * np.exp(-0.5 * b1 * ds[i - 1])
        Y_lag[i] = Y_lag[i - 1] * np.exp(-b2 * ds[i - 1]) + d_alpha_qs * A2 * np.exp(-0.5 * b2 * ds[i - 1])
        alpha_eff[i] = alpha_qs[i] - X_lag[i] - Y_lag[i]

        C_nc[i] = C_n_slope * (alpha_eff[i] - alpha_0n_inv)
        # impulsive (non-circulatory) normal force coefficient
        tmp = -a * dt / (K_alpha * chord)
        tmp_2 = -(vel[i, 2] - vel[i - 1, 2])  # minus here needed because of coordinate system
        D_i[i] = D_i[i - 1] * np.exp(tmp) + tmp_2 * np.exp(
            0.5 * tmp
        )  # take out for consistency on no compressibility effects. DON'T TAKE OUT, RESULTS BECOME SHIT
        C_ni[i] = 4 * K_alpha * chord / rel_flow_vel * (-vel[i, 2] - D_i[i])  # -vel because of CS
        # C_ni[i] = -4*K_alpha*chord/rel_flow_vel*-vel[i, 2]  # -vel because of CS

        # add circulatory and impulsive
        C_npot[i] = C_nc[i] + C_ni[i]

        # --------- MODULE nonlinear trailing edge separation
        D_p[i] = D_p[i - 1] * np.exp(-ds[i - 1] / T_p)
        D_p[i] += (C_npot[i] - C_npot[i - 1]) * np.exp(-0.5 * ds[i - 1] / T_p)
        C_nsEq[i] = C_npot[i] - D_p[i]
        alpha_sEq[i] = C_nsEq[i] / C_n_slope + alpha_0n_inv

        f_n[i] = f_n_lookup(alpha_sEq[i])
        tmp_bl = -ds[i - 1] / T_bl
        D_bl_n[i] = D_bl_n[i - 1] * np.exp(tmp_bl) + (f_n[i] - f_n[i - 1]) * np.exp(0.5 * tmp_bl)
        f_n_Dp[i] = f_n[i] - D_bl_n[i]

        if f_n_Dp[i] >= 0:
            C_nvisc[i] = C_n_slope * (alpha_eff[i] - alpha_0n_visc)
            C_nvisc[i] *= ((1 + np.sqrt(f_n_Dp[i])) / 2) ** 2
        else:
            C_nvisc[i] = C_n_visc(alpha_eff[i])

        C_nf[i] = C_nvisc[i] + C_ni[i]

        C_tf[i] = C_t_visc(alpha_sEq[i])

        # --------- MODULE leading-edge vortex position
        tau_vortex[i] = tau_vortex[i - 1]
        if C_nsEq[i] > C_n_crit:
            tau_vortex[i] += 0.45 * ds[i - 1]
        elif C_nsEq[i] < C_n_crit and d_alpha_qs >= 0:
            tau_vortex[i] *= np.exp(-ds[i - 1])

        # --------- MODULE leading-edge vortex lift
        C_nv_instant[i] = C_nc[i] * (1 - ((1 + np.sqrt(f_n_Dp[i])) / 2) ** 2)
        C_nv[i] = C_nv[i - 1] * np.exp(-ds[i - 1] / T_v)
        if 0 < tau_vortex[i] and tau_vortex[i] < tau_vortex_pure_decay:
            C_nv[i] += (C_nv_instant[i] - C_nv_instant[i - 1]) * np.exp(-0.5 * ds[i - 1] / T_v)

        # --------- MODULE moment coefficient
        # viscous
        C_mf[i] = C_m_polar(alpha_sEq[i])

        # vortex
        C_Pv = K_v * (1 - np.cos(np.pi * tau_vortex[i] / tau_vortex_pure_decay))
        C_mV[i] = -C_Pv * C_nv[i]  # vortex

        # circulatory
        C_Pf = K_fC * C_n_crit
        if (tau_vortex[i] < tau_vortex_pure_decay) and (d_alpha_qs >= 0):
            tmp = -ds[i - 1] / T_MU  # T_MU!
            tmp2 = C_Pf * (C_nv_instant[i] - C_nv_instant[i - 1])
            C_mC[i] = C_mC[i - 1] * np.exp(tmp) - tmp2 * np.exp(tmp / 2)
        elif d_alpha_qs < 0:
            tmp = -ds[i - 1] / T_MD  # T_MD!
            tmp2 = C_Pf * (C_nv_instant[i] - C_nv_instant[i - 1])
            C_mC[i] = C_mC[i - 1] * np.exp(tmp) - tmp2 * np.exp(tmp / 2)
        else:
            C_mC[i] = C_mC[i - 1]

        # --------- Combining everything
        coefficients = np.asarray([C_tf[i], C_nf[i] + C_nv[i], C_mf[i] + C_mV[i] + C_mC[i]])
        rot_df = np.asarray(
            [
                [-np.cos(alpha_qs[i]), np.sin(alpha_qs[i]), 0],
                [np.sin(alpha_qs[i]), np.cos(alpha_qs[i]), 0],
                [0, 0, 1],
            ]
        )
        coeffs = rot_df @ coefficients  # now as [drag, lift, mom]

        C_d_p = C_d_polar(alpha_qs[i])
        if coeffs[0] < C_d_p and alpha_qs[i] < aoa_crit:
            coeffs[0] = C_d_p

        C_dus[i] = coeffs[0]
        C_lus[i] = coeffs[1]
        C_mus[i] = coeffs[2]
        return coeffs

    for i in range(N):
        BL_first_order_IAG2(i, 0.3, 0.7, 0.7, 0.53)

    return alpha, C_dus, C_lus, C_mus


if __name__ == "__main__":
    from init_iag import init_BL_first_order_IAG2
    from plot import plot_iag_results

    init_BL_first_order_IAG2("S801_polars.dat")
    res_period = 200
    n_periods = 4
    alpha, C_dus, C_lus, C_mus = run_iag(0.073, 0.457, 23.7, 10.85, 19.25, res_period, n_periods)
    plot_iag_results(alpha, C_dus, C_lus, C_mus, "bangga.dat", res_period, n_periods)
