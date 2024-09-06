import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import find_peaks

dfs = {
    "oF": [pd.read_csv("data/FFA_WA3_221/validation/BL_openFAST_Cl_disc/measurement/3/f_aero.dat"), ("lift", "aoa")],
    # "oF": [pd.read_csv("data/FFA_WA3_221/validation/BL_openFAST_Cl_disc/measurement/3/f_aero.dat"), ("drag", "aoa")],
    # "oF": [pd.read_csv("data/FFA_WA3_221/validation/BL_openFAST_Cl_disc/measurement/3/f_aero.dat"), 
        #    ("aoa", "moment_hgm")],
    "iag": [pd.read_csv("data/FFA_WA3_221/validation/BL_first_order_IAG2/measurement/3/f_aero.dat"), ("aoa", 
                                                                                                      "normal")],
    # "iag": [pd.read_csv("data/FFA_WA3_221/validation/BL_first_order_IAG2/measurement/3/f_aero.dat"), ("aoa", 
                                                                                                    #   "tangential")]
    # "iag": [pd.read_csv("data/FFA_WA3_221/validation/BL_first_order_IAG2/measurement/3/f_aero.dat"), ("aoa", "moment")],
    # "aerohor": [pd.read_csv("data/FFA_WA3_221/validation/BL_AEROHOR/measurement/2/f_aero.dat"), ("tangential")],
    # "aerohor": [pd.read_csv("data/FFA_WA3_221/validation/BL_AEROHOR/measurement/2/f_aero.dat"), ("tangential")],
}


def plot(ax, aoa, df, cols, start, end, mulitply=None, apply=None, hw=None):
    for col in cols:
        vals = (df[col].iloc[start:end]).to_numpy()
        if apply is not None:
            vals = apply(vals)
        if mulitply is not None:
            vals *= mulitply
        x1, y1 = aoa[0], vals[0]
        x2, y2 = aoa[5], vals[5]
        ax.plot(aoa, vals, label=col)
        ax.arrow(x1, y1, x2-x1, y2-y1, facecolor="k", zorder=10, head_width=0.007 if hw is None else hw, head_length=0.3)


for title, [df, do] in dfs.items():
    aoa = np.rad2deg(df["alpha_steady"])
    peaks = find_peaks(aoa)[0]
    n = peaks[-1]-peaks[-2]
    start = peaks[-2]-int(n/4)
    end = peaks[-1]-int(n/4)
    aoa = (aoa.iloc[start:end]).to_numpy()
    #------- normal contributions
    if "normal" in do:
        fig, ax = plt.subplots()
        plot(ax, aoa, df, ["C_nvisc", "C_ni", "C_nv", "C_nsEq", "C_nc"], start, end, np.cos(np.deg2rad(aoa)))
        plot(ax, aoa, df, ["C_tf"], start, end, np.sin(np.deg2rad(aoa)))
        plot(ax, aoa, df, ["C_lus"], start, end)
        ax.plot(aoa, df["C_lus"].iloc[start:end], label="C_lus")
        # ax.plot(aoa, df["C_nvisc"]*np.sin(np.deg2rad(aoa)), label="C_nvisc")
        # ax.plot(aoa, df["C_dus"].iloc[start:end], label="C_dus")
        fig.suptitle(title)
        ax.legend()

    #------- angles of attack
    if "aoa" in do:
        fig, ax = plt.subplots()
        plot(ax, aoa, df, ["alpha_eff", "alpha_qs"], start, end, apply=np.rad2deg, hw=0.1)
        ax.legend()
        fig.suptitle(title)

    #------- tangential contribution
    if "tangential" in do:
        fig, ax = plt.subplots()
        plot(ax, aoa, df, ["C_nv", "C_nvisc", "C_ni"], start, end, np.sin(np.deg2rad(aoa)))
        plot(ax, aoa, df, ["C_tf"], start, end, np.cos(np.deg2rad(aoa)), apply=lambda x: np.multiply(x, -1))
        plot(ax, aoa, df, ["C_dus"], start, end)
        fig.suptitle(title)
        ax.legend()

    #------- moment contribution
    if "moment" in do and not "moment_hgm" in do:
        fig, ax = plt.subplots()
        plot(ax, aoa, df, ["C_mC", "C_mV", "C_mf", "C_mus"], start, end)
        fig.suptitle(title)
        ax.legend()

    #------- moment contribution
    if "moment_hgm" in do:
        fig, ax = plt.subplots()
        plot(ax, aoa, df, ["C_ms", "C_mnc", "C_mus"], start, end)
        fig.suptitle(title)
        ax.legend()

    #------- drag contribution
    if "drag" in do:
        fig, ax = plt.subplots()
        plot(ax, aoa, df, ["C_ds", "C_dind", "C_dtors", "C_dsep", "C_dus"], start, end)
        ax.legend()
        fig.suptitle(title)

    #------- lift contribution
    if "lift" in do:
        fig, ax = plt.subplots()
        plot(ax, aoa, df, ["C_lc", "C_lnc", "C_lus"], start, end)
        ax.legend()
        fig.suptitle(title)


plt.show()