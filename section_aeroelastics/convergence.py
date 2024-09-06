import defaults
import matplotlib.pyplot as plt
from plot_utils import PlotHandler
import pandas as pd
import numpy as np
from os.path import join


dts = [0.01, 0.005, 0.001, 0.0005]
# models = ["BL_openFAST_Cl_disc", "BL_openFAST_Cl_disc_f_scaled"]
models = ["BL_AEROHOR","BL_first_order_IAG2"]

map_model = {
    "BL_openFAST_Cl_disc": "HGM openFAST",
    "BL_openFAST_Cl_disc_f_scaled": "HGM $f$-scaled",
    "BL_first_order_IAG2": "1st-order IAG",
    "BL_AEROHOR": "AEROHOR",
}
colours = [defaults._c[0], defaults._c[3], defaults._c[6], defaults._c[7]]

for model in models:
    fig, ax = plt.subplots()
    for i, dt in enumerate(dts):
        df = pd.read_csv(join("data/FFA_WA3_221/simulation/free/", model, f"aoa_20_v_35_dt_{dt}", "general.dat"))
        time = df["time"].to_numpy()
        if "openFAST" in model:
            t_frame = (598, 600)
        else:
            t_frame = (10, 102)
            
        ids = np.logical_and(time>=t_frame[0], time<=t_frame[1])
        ax.plot(df["time"].iloc[ids], df["pos_x"].iloc[ids], label=r"$\Delta t=\qty{"+str(dt)+r"}{\second}$", 
                color=colours[i])
    ax.set_xticks(t_frame)
    y_ticks = ax.get_yticks()
    if y_ticks[0] < 0 and y_ticks[-1] > 0:
        yticks = [y_ticks[0], 0, y_ticks[-1]]
    else:
        yticks = [y_ticks[0], y_ticks[-1]]
    ax.set_yticks(yticks)
    ax.spines[["top", "right"]].set_visible(False)
    handler = PlotHandler(fig, ax)
    handler.update(x_labels=r"time (\second)", y_labels=r"$x$ (\metre)", 
                   titles=map_model[model], legend=False)
    handler.save(f"drawings/{model}_convergence.pdf")
