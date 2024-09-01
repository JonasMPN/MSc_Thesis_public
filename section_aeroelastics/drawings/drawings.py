import matplotlib.pyplot as plt
import pandas as pd
import numpy as np 
import matplotlib as mpl


_plot_backend_latex = True
if _plot_backend_latex:
    mpl.use("pgf")
    plt.rcParams.update({
        "font.family": "Arial",
        "text.usetex": True,
        "pgf.rcfonts": False,
        "pgf.preamble": "\n".join([
            r"\usepackage{amsmath}",
            r"\usepackage{fontspec}",
            r"\setmainfont{Arial}",
            r"\usepackage{siunitx}",
            r"\usepackage{xcolor}",
            r"\definecolor{updatecolour}{RGB}{33, 36, 39}"
        ]),
        "axes.edgecolor": "#212427",
        "axes.labelcolor": "#212427",
        "axes.edgecolor": "#212427",
        "xtick.color": "#212427",
        "ytick.color": "#212427",
        "xtick.labelcolor": "#212427",
        "ytick.labelcolor": "#212427",
    })


do = {
    "profile": False,
    "tors_spring": False,
    "tors_damper": False,
    "Cn_vs_Cl": True
}

colour = "#212427"  # better black
# colour = "#337538"  # green
linewidth = 6


if do["profile"]:
    df_profile = pd.read_csv("../data/FFA_WA3_221/profile.dat", delim_whitespace=True)
    fig, ax = plt.subplots()
    ax.plot(df_profile["x"], df_profile["y"], color=colour)
    ax.plot(1/4, 0, "o", color=colour)
    ax.plot(3/4, 0, "o", color=colour)
    ax.plot([-0, 1.2], [0, 0], "--", color=colour)
    ax.set_aspect("equal")
    ax.set_xlim((-0.01, 1.2))
    ax.axis("off")
    fig.set_size_inches(5, 1)
    fig.savefig("profile.svg", transparent=True, bbox_inches="tight")
    # plt.show()

if do["tors_spring"]:
    r_start = 0.5
    r_end = 0.25
    n_loops = 2
    angle_end = 150  # in deg
    res = 300

    r = np.linspace(r_start, r_end, res)
    angle = np.linspace(0, 2*n_loops*np.pi+np.deg2rad(angle_end), res)

    fig, ax = plt.subplots()
    ax.plot(r*np.cos(angle), r*np.sin(angle), color=colour, lw=linewidth)
    ax.plot([r_start*0.9, r_start*1.1], [0, 0], color=colour, lw=linewidth)
    ax.set_aspect("equal")
    ax.axis("off")
    fig.savefig("tors_spring.svg", transparent=True)
    # plt.show()

if do["tors_damper"]:
    add_ground = True
    body_angle_range = 30  # deg
    body_angle_end = 110  # deg
    plate_body_end_distance = 10  # deg
    body_width = 0.3
    damper_angle_end = 150  # deg
    res_angles = 30
    
    pusher_base = ([0.9, 1.1], [0, 0])

    pusher_angle = np.linspace(0, np.deg2rad(body_angle_end-plate_body_end_distance), res_angles)
    pusher = (np.cos(pusher_angle), np.sin(pusher_angle))

    connector_angle = np.linspace(np.deg2rad(body_angle_end), np.deg2rad(damper_angle_end), res_angles)
    connector = (np.cos(connector_angle), np.sin(connector_angle))

    angle_plate = np.deg2rad(body_angle_end-plate_body_end_distance)
    r_plate = np.asarray([-0.9, 0.9])*body_width/2
    plate = (np.cos([angle_plate, angle_plate])*r_plate+pusher[0][-1], 
             np.sin([angle_plate, angle_plate])*r_plate+pusher[1][-1])
    
    bae = np.deg2rad(body_angle_end)
    r_housing_end = np.asarray([-1, 1])*body_width/2
    housing_end = (np.cos([bae, bae])*r_housing_end+connector[0][0], 
                   np.sin([bae, bae])*r_housing_end+connector[1][0])
    
    angle_housing = np.linspace(np.deg2rad(body_angle_end-body_angle_range), np.deg2rad(body_angle_end), res_angles)
    upper_housing = ((1+body_width/2)*np.cos(angle_housing), (1+body_width/2)*np.sin(angle_housing))
    lower_housing = ((1-body_width/2)*np.cos(angle_housing), (1-body_width/2)*np.sin(angle_housing))

    fig, ax = plt.subplots()
    for line, linename in zip([pusher_base, pusher, connector, plate, housing_end, upper_housing, lower_housing],
                              ["pusher_base", "pusher", "connector", "plate", "housing_end", "upper_housing",
                               "lower_housing"]):
        ax.plot(*line, color=colour, lw=linewidth)
    ax.set_aspect("equal")
    ax.axis("off")
    fig.savefig("tors_damper.svg", transparent=True)
    # plt.show()

if do["Cn_vs_Cl"]:
    polar = pd.read_csv("../data/FFA_WA3_221/polars_new.dat", delim_whitespace=True)
    aoa = np.deg2rad(polar["alpha"])
    C_n = polar["C_l"]*np.cos(aoa)+polar["C_d"]*np.sin(aoa)

    fig, ax = plt.subplots()
    ax.axvline(-26.5, ymax=0.25, linestyle="--", lw=0.7, color="#212427")
    ax.axvline(29, ymax=0.75, linestyle="--", lw=0.7, color="#212427")
    ax.plot(polar["alpha"], polar["C_l"], label=r"$C_l$", color="#2E2585")
    ax.plot(polar["alpha"], C_n, label=r"$C_n$", color="#DCCD7D")
    ax.set_xlabel(r"$\alpha$ (\unit{\degree})")
    ax.set_ylabel(r"force coefficient (\unit{-})")
    ax.legend()

    ax.set_xticks([-180, -26.5, 0, 29, 180])
    ax.set_yticks([-1.5, 0, 1.5])
    ax.spines[['right', 'top']].set_visible(False)
    fig.savefig("Cn_vs_Cl.pdf")

