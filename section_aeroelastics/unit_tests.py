from calculation_utils import compress_compound_oscillation, reconstruct_from_file, zero_oscillations
import pandas as pd
from scipy.signal import find_peaks
import matplotlib.pyplot as plt
from helper_functions import Helper
from utils_plot import PlotHandler
from os.path import join
import numpy as np
helper = Helper()

do = {
    "compress_motion": False,
    "reconstruct_compressed": True,
}

# for compression and reconstruction of motion
file_motion = "data/FFA_WA3_221/simulation/free/BL_openFAST_Cl_disc/base/general.dat"
file_compress = "data/FFA_WA3_221/simulation/free/BL_openFAST_Cl_disc/base/general_compressed.json"
cols = ["pos_x", "pos_y", "pos_tors", "vel_x", "vel_y", "vel_tors"]
case_name = "free_base_xy_new_struc"

if do["compress_motion"]:
    compress_compound_oscillation(file_motion, file_compress, cols, period_res=100)

if do["reconstruct_compressed"]:
    reconstructed, _ = reconstruct_from_file(file_compress)
    
    df = pd.read_csv(file_motion)
    t = df["time"]
    x = df["pos_x"].to_numpy()
    peaks = find_peaks(x)[0]
    for col in ["pos_x", "pos_y", "pos_tors"]:
        timeseries = df[col]
        new_peaks = find_peaks(timeseries.to_numpy())[0]
        if new_peaks[-1]-new_peaks[-2] > peaks[-1]-peaks[-2]:
            peaks = new_peaks
    ids = np.arange(peaks[-2], peaks[-1])

    osci_x = reconstructed["pos_x"]["vals"]
    time_x = reconstructed["pos_x"]["time"]
    mean_x = reconstructed["pos_x"]

    full_time, zeroed = zero_oscillations(time_x, 5, **reconstructed)

    # (pos_x, pos_y)
    fig, axs = plt.subplots()
    axs.plot(x[ids], df["pos_y"].to_numpy()[ids], "k", label="simulation")
    axs.plot(zeroed["pos_x"], zeroed["pos_y"], "or", ms=2, label="reconstruction")
    handler = PlotHandler(fig, axs)
    handler.update(x_labels="x (m)", y_labels="y (m)", grid=True, legend=True)
    handler.save(join("data", "unit_tests", "compression_recompression", case_name, "split_xy.pdf"))

    # (pos_x, pos_tors)
    fig, axs = plt.subplots()
    axs.plot(x[ids], np.rad2deg(df["pos_tors"].to_numpy()[ids]), "k", label="simulation")
    axs.plot(zeroed["pos_x"], np.rad2deg(zeroed["pos_tors"]), "or", ms=2, label="reconstruction")
    handler = PlotHandler(fig, axs)
    handler.update(x_labels="x (m)", y_labels="torsion (deg)", grid=True, legend=True)
    handler.save(join("data", "unit_tests", "compression_recompression", case_name, "split_xtors.pdf"))

    # (vel_x, vel_y)
    fig, axs = plt.subplots()
    axs.plot(df["vel_x"].to_numpy()[ids], df["vel_y"].to_numpy()[ids], "k", label="simulation")
    axs.plot(zeroed["vel_x"], zeroed["vel_y"], "or", ms=2, label="reconstruction")
    handler = PlotHandler(fig, axs)
    handler.update(x_labels="u (m/s)", y_labels="v (m/s)", grid=True, legend=True)
    handler.save(join("data", "unit_tests", "compression_recompression", case_name, "split_uv.pdf"))

    # (vel_x, vel_tors)
    fig, axs = plt.subplots()
    axs.plot(df["vel_x"].to_numpy()[ids], np.rad2deg(df["vel_tors"].to_numpy()[ids]), "k", label="simulation")
    axs.plot(zeroed["vel_x"], np.rad2deg(zeroed["vel_tors"]), "or", ms=2, label="reconstruction")
    handler = PlotHandler(fig, axs)
    handler.update(x_labels="u (m/s)", y_labels="torsion rate (deg/s)", grid=True, legend=True)
    handler.save(join("data", "unit_tests", "compression_recompression", case_name, "split_utorsrate.pdf"))
