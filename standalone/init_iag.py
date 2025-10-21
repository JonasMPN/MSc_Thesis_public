import json

import numpy as np
import pandas as pd
from scipy import interpolate

from helper import get_zero_crossing_and_max_slope


def init_BL_first_order_IAG2(
    file_polar: str,
    alpha_critical: float = 15.1,
    resolution: int = 100,
    characteristics_aoa_range: tuple[float, float] = (-10, 20),
):
    df_polar = pd.read_csv(file_polar)
    alpha_interp = np.deg2rad(np.linspace(df_polar["alpha"].min(), df_polar["alpha"].max(), resolution))
    C_l_polar = interpolate.interp1d(np.deg2rad(df_polar["alpha"]), df_polar["C_l"])
    C_d_polar = interpolate.interp1d(np.deg2rad(df_polar["alpha"]), df_polar["C_d"])
    C_m_polar = interpolate.interp1d(np.deg2rad(df_polar["alpha"]), df_polar["C_m"])

    def C_t_visc(alpha):
        return -np.cos(alpha) * C_d_polar(alpha) + np.sin(alpha) * C_l_polar(alpha)

    def C_n_visc(alpha):
        return np.sin(alpha) * C_d_polar(alpha) + np.cos(alpha) * C_l_polar(alpha)

    C_d_use = C_d_polar(alpha_interp)
    C_l_use = C_l_polar(alpha_interp)
    C_n_inv = np.cos(alpha_interp) * C_l_use + np.sin(alpha_interp) * (C_d_use - C_d_use.min())

    aoa_search = np.deg2rad(np.asarray(characteristics_aoa_range))
    ids = np.logical_and(alpha_interp >= aoa_search[0], alpha_interp <= aoa_search[1])
    aero = {"alpha_critical": np.deg2rad(alpha_critical)}
    for coeff, values in {"C_n_visc": C_n_visc(alpha_interp[ids]), "C_n_inv": C_n_inv[ids]}.items():
        root, root_slope, max_slope = get_zero_crossing_and_max_slope(alpha_interp[ids], values)
        aero[coeff + "_root"] = root
        aero[coeff + "_root_slope"] = root_slope
        aero[coeff + "_max_slope"] = max_slope

    with open("polar_values.json", "w") as ff_aero:
        json.dump(aero, ff_aero, indent=4)

    sqr_fn = (
        2 * np.sqrt(C_n_visc(alpha_interp) / (aero["C_n_inv_max_slope"] * (alpha_interp - aero["C_n_visc_root"]))) - 1
    )
    pd.DataFrame(
        {
            "alpha": alpha_interp,
            "C_t_visc": C_t_visc(alpha_interp),
            "C_n_visc": C_n_visc(alpha_interp),
            "C_d": C_d_polar(alpha_interp),
            "C_m": C_m_polar(alpha_interp),
            "f_n": sqr_fn**2 * np.sign(sqr_fn),
        }
    ).to_csv("polar_characteristics.csv", index=False)
