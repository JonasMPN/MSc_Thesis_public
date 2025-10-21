import matplotlib.pyplot as plt
import numpy as np


def plot_iag_results(alpha, C_dus, C_lus, C_mus, file_validate_againts, period_res, n_periods):

    ids = slice(period_res * (n_periods - 1) - 1, period_res * n_periods)
    plt.subplots()
    plt.plot(np.rad2deg(alpha[ids]), C_dus[ids])
    plt.xlabel("Steady Angle of Attack (deg)")
    plt.ylabel("C_d unsteady")
    plt.grid()

    plt.subplots()
    plt.plot(np.rad2deg(alpha[ids]), C_lus[ids])
    plt.xlabel("Steady Angle of Attack (deg)")
    plt.ylabel("C_l unsteady")
    plt.grid()

    plt.subplots()
    plt.plot(np.rad2deg(alpha[ids]), C_mus[ids])
    plt.xlabel("Steady Angle of Attack (deg)")
    plt.ylabel("C_m unsteady")
    plt.grid()

    plt.show()
