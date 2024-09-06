import matplotlib.pyplot as plt
import pandas as pd
import numpy as np 
import matplotlib as mpl
from scipy.interpolate import interp1d
import json
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm

_plot_backend_latex = True
# fig_size =  "textwidth"
# fig_size =  "half_textwidth"
fig_size =  "page_height"
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

if fig_size == "textwidth":
    plt.rcParams.update({
        "axes.labelsize": 14,
        "axes.titlesize": 14,
        "xtick.labelsize": 12,
        "ytick.labelsize": 12,
        "legend.fontsize": 12,
    })
elif fig_size == "half_textwidth":
    plt.rcParams.update({
        "axes.labelsize": 18,
        "axes.titlesize": 18,
        "xtick.labelsize": 16,
        "ytick.labelsize": 16,
        "legend.fontsize": 16,
    })
elif fig_size == "page_height":
    plt.rcParams.update({
        "axes.labelsize": 10,
        "axes.titlesize": 10,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "legend.fontsize": 8,
    })


def full_range_to_normalised(*args):
    return [[val/256 for val in vals] for vals in args]
_cmap_colours = ["#94CBEC", "#DCCD7D", "#2E2585"]


# palette names available: Paul Tol, SSP, Tableu10ColorBlind
_palette_name = "Paul Tol"  # used for all lines in plots (also such that have None as linstyle)
_fill_name = "Paul Tol"  # used for filling polygons
_c_all = {
    # https://www.nceas.ucsb.edu/sites/default/files/2022-06/Colorblind%20Safe%20Color%20Schemes.pdf
    "Paul Tol": full_range_to_normalised(*[  
        (148, 203, 236),  # sky blue, #94CBEC
        (194, 106, 119),  # skin-pink, #C26A77
        (51, 117, 56),  # medium green, #337538
        (93, 168, 153),  # something between green and blue, #5DA899
        (46, 37, 133),  # dark blue, #2E2585
        (221, 221, 211),  # light grey, #DDDDDD
        (220, 205, 125),  # light ocker, #DCCD7D
        (159, 74, 150),  # pink, #9F4A96
        (126, 41, 84),  # violet, #7E2954
        ]),
    "SSP": [ # https://www.simplifiedsciencepublishing.com/resources/best-color-palettes-for-scientific-figures-and-data-visualizations
        "#003a7d",  # dark blue
        "#008dff",  # medium blue
        "#ff73b6",  # pink
        "#c701ff",  # purple
        "#4ecb8d",  # green
        "#ff9d3a",  # orange
        "#f9e858",  # yellow
        "#d83034",  # red
    ],
    # https://tableaufriction.blogspot.com/2012/11/finally-you-can-use-tableau-data-colors.html
    "Tableu10ColorBlind": full_range_to_normalised(*[
        (0, 107, 164),  # blue, #006BA4
        (255, 128, 14),  # orange, #FF800E
        (171, 171, 171),  # light grey, #ABABAB
        (89, 89, 89),  # dark grey, #595959
        (95, 158, 209),  # sky blue, #5F9ED1
        (200, 82, 0),  # brown, #C85200
        (137, 137, 137),  # medium grey, #898989
        (163, 200, 236),  # light sky blue, #A3C8EC
        (207, 207, 207),  # very light grey, #CFCFCF
        (255, 188, 121),  # light orange, #FFBC79
        ])   
}
_colormap = LinearSegmentedColormap.from_list("Tol_muted_cmap", ["#DCCD7D", "#94CBEC", "#2E2585"], N=256)
def _fcolormap(n_levels):
    return LinearSegmentedColormap.from_list("Tol_muted_cmap", ["#DCCD7D", "#94CBEC", "#2E2585"], N=n_levels)

_c = _c_all[_palette_name]
_c_fill = _c_all[_fill_name]
_betterblack = "#212427"



do = {
    "profile": True,
    "tors_spring": False,
    "tors_damper": False,
    "Cn_vs_Cl": False,
    "Kirchhoff_HGM": True,
    "oF_vs_f_scaled": False
}

colour = "#212427"  # better black
better_black = "#212427"
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

if do["Kirchhoff_HGM"]:
    df_polar = pd.read_csv("../data/FFA_WA3_221/polars_new.dat", delim_whitespace=True)
    df_fully_sep = pd.read_csv("../data/FFA_WA3_221/preparation/BL_openFAST_Cl_disc/C_l_fs.dat")
    df_sep_point = pd.read_csv("../data/FFA_WA3_221/preparation/BL_openFAST_Cl_disc/f_l.dat")

    C_l = interp1d(np.deg2rad(df_polar["alpha"]), df_polar["C_l"])
    C_lfs = interp1d(df_fully_sep["alpha_fs"], df_fully_sep["C_l_fs"])
    f = interp1d(df_sep_point["alpha_l"], df_sep_point["f_l"])

    with open("../data/FFA_WA3_221/preparation/BL_openFAST_Cl_disc/aero_characteristics.json", "r") as aero:
        aero_characteristics = json.load(aero)
        C_lslope = aero_characteristics["C_l_visc_max_slope"]
        alpha_0 = aero_characteristics["C_l_visc_root"]

    aoas, fs = np.meshgrid(np.linspace(np.deg2rad(-26.5), np.deg2rad(29), 100),
                           np.linspace(0, 1, 100))
    
    Kirchhoff = C_lslope*(aoas-alpha_0)*((1+np.sqrt(fs))/2)**2
    HGM = C_lslope*(aoas-alpha_0)*fs+(1-fs)*C_lfs(aoas)

    # colors = ["#DCCD7D", "#94CBEC", "#2E2585"]
    cmap = LinearSegmentedColormap.from_list("Tol_muted_cmap", _cmap_colours, N=300)
    # Define the normalization for your data
    norm = TwoSlopeNorm(vmin=-19, vcenter=0, vmax=50)
        
    levels = 300
    fig, ax = plt.subplots()
    rel_diff = (HGM-Kirchhoff)/HGM*1e2
    cf = ax.contourf(np.rad2deg(aoas), fs, (HGM-Kirchhoff)/HGM*1e2, norm=norm, cmap=cmap, levels=levels)
    cbar = fig.colorbar(cf, ax=ax, label="% rel. difference between the Kirchhoff and HGM approach")
    ax.set_xlabel(r"$\alpha$ (\unit{\degree})")
    ax.set_ylabel(r"$f_l$ (\unit{-}) and $x_4$ (\unit{-})")
    ax.set_xticks([-26.5, 0, 29])
    ax.set_yticks([0, 1])
    cbar.ax.set_yticks([-19.7, 0, 50])
    cf.set_edgecolor("face")
    fig.savefig("Kirchhoff_vs_HGM.pdf")

if do["oF_vs_f_scaled"]:
    from scipy.interpolate import interp1d
    c_oF = "#337538"
    c_f_scaled = "#DCCD7D"

    inflow_speed = 5
    T_u_last = 0.5/inflow_speed
    dt = 0.01
    A1 = 0.165
    A2 = 0.335
    b1 = 0.0445
    b2 = 0.3
    tmp1 = np.exp(-dt*b1/T_u_last) 
    tmp2 = np.exp(-dt*b2/T_u_last)
    T_p = 1.5
    T_f = 6
    df_fl = pd.read_csv("../data/FFA_WA3_221/preparation/BL_openFAST_Cl_disc/f_l.dat")
    f_l = interp1d(df_fl["alpha_l"], df_fl["f_l"])

    # aoa_qs = np.r_[np.zeros(50), np.ones(300)]
    aoa_qs = np.r_[np.zeros(50), np.linspace(0, 1, 100), np.ones(850)]
    case_name = "fast"
    
    # aoa_qs = np.r_[np.zeros(50), np.linspace(0, 1, 850), np.ones(100)]
    # case_name = "slow"
    
    time = dt*np.arange(aoa_qs.size)
    alpha_eff = np.zeros_like(aoa_qs)
    C_lpot = np.zeros_like(aoa_qs)
    f_steady = np.ones_like(aoa_qs)
    x1 = np.zeros_like(aoa_qs)
    x2 = np.zeros_like(aoa_qs)
    x3 = np.zeros_like(aoa_qs)
    x4 = 0.9594992457447619*np.ones_like(aoa_qs)
    for i in range(1, aoa_qs.size):
        alpha_qs_avg = 0.5*(aoa_qs[i]+aoa_qs[i-1])
        x1[i] = x1[i-1]*tmp1+alpha_qs_avg*A1*(1-tmp1)  # this discretisation causes the failure in 
        x2[i] = x2[i-1]*tmp2+alpha_qs_avg*A2*(1-tmp2)  # reconstructing the static polar exactly

        alpha_eff[i] = aoa_qs[i]*(1-A1-A2)+x1[i]+x2[i]
        C_lpot[i-1] = 2*np.pi*alpha_eff[i]

        tmp3 = np.exp(-dt/(T_u_last*T_p))
        x3[i] = x3[i-1]*tmp3+0.5*(C_lpot[i]+C_lpot[i-1])*(1-tmp3)
        alpha_eq = x3[i]/(2*np.pi)

        tmp4 = np.exp(-dt/(T_u_last*T_f))
        f_steady[i] = f_l(alpha_eq)
        x4[i] = x4[i-1]*tmp4+0.5*(f_steady[i]+f_steady[i-1])*(1-tmp4)
        
    fig_all, ax_all = plt.subplots()
    ax_all.plot(time, np.rad2deg(aoa_qs), color=better_black, label=r"$\alpha_{\text{qs}}$")
    ax_all.plot(time, np.rad2deg(alpha_eff), color=c_oF, label=r"$\alpha_{\text{eff}}^{\text{openFAST}}$")
    ax_all_twin = ax_all.twinx()
    ax_all_twin.plot(time, x4, ls="--", color=c_oF, label=r"$x_4^{\text{openFAST}}$")

    fig_force, ax_force = plt.subplots()
    # ax_force.plot(time, alpha_eff*((1+np.sqrt(x4))/2)**2*x4, label="openFAST")
    ax_force.plot(time, 2*np.pi*alpha_eff*x4, color=c_oF, label="openFAST")
    label_oF = np.round((2*np.pi*alpha_eff*x4).max(), 1)

    alpha_eff = np.zeros_like(aoa_qs)
    C_lpot = np.zeros_like(aoa_qs)
    f_steady = np.ones_like(aoa_qs)
    x1 = np.zeros_like(aoa_qs)
    x2 = np.zeros_like(aoa_qs)
    x3 = np.zeros_like(aoa_qs)
    x4 = 0.9594992457447619*np.ones_like(aoa_qs)

    tmp_t = T_u_last/dt
    for i in range(1, aoa_qs.size):
        d_downwash = aoa_qs[i]*inflow_speed-aoa_qs[i-1]*inflow_speed
        x1[i] = x1[i-1]*tmp1+d_downwash*A1/b1*tmp_t*(1-tmp1)*x4[i-1]
        x2[i] = x2[i-1]*tmp2+d_downwash*A2/b2*tmp_t*(1-tmp2)*x4[i-1]

        alpha_eff[i] = aoa_qs[i]-(x1[i]+x2[i])/inflow_speed
        C_lpot[i] = 2*np.pi*alpha_eff[i]

        tmp3 = np.exp(-dt/(T_u_last*T_p))
        x3[i] = x3[i-1]*tmp3+0.5*(C_lpot[i]+C_lpot[i-1])*(1-tmp3)
        alpha_eq = x3[i]/(2*np.pi)

        tmp4 = np.exp(-dt/(T_u_last*T_f))
        f_steady[i] = f_l(alpha_eq)
        x4[i] = x4[i-1]*tmp4+0.5*(f_steady[i]+f_steady[i-1])*(1-tmp4)

    ax_all.plot(time, np.rad2deg(alpha_eff), color=c_f_scaled, label=r"$\alpha_{\text{eff}}^{f\text{-scaled}}$")
    ax_all_twin.plot(time, x4, ls="--", color=c_f_scaled, label=r"$x_4^{f\text{-scaled}}$")


    # ax_force.plot(time, alpha_eff*((1+np.sqrt(x4))/2)**2*x4, label=r"$f$-scaled")
    ax_force.plot(time, 2*np.pi*alpha_eff*x4, color=c_f_scaled, label=r"$f$-scaled")
    label_f_scaled = np.round((2*np.pi*alpha_eff*x4).max(), 1)
    ax_force.set_xlabel(r"time (\unit{\second})")
    
    ax_all.set_xlabel(r"time (\unit{\second})")
    ax_all.set_ylabel(r"angle of attack (\unit{\degree})")
    ax_all_twin.set_ylabel(r"$x_4$ (\unit{-})")
    ax_all_twin.legend(loc="center right")
    if case_name == "fast":
        ax_all.legend(loc="center right", bbox_to_anchor=(0.65, 0.5))
    elif case_name == "slow":
        ax_all.legend(loc="center left")
    ax_force.set_ylabel(r"$\frac{2\pi}{\unit{\radian}}\alpha_{\text{eff}}x_4$ (\unit{-})")
    ax_force.legend()
    ax_force.spines[['right', 'top']].set_visible(False)
    ax_all.spines[['top']].set_visible(False)
    ax_all_twin.spines[['top']].set_visible(False)
    ax_force.set_yticks([0, label_f_scaled, label_oF])
    ax_force.set_xticks([0, 10])
    ax_all.set_xticks([0, 10])
    ax_all.set_yticks([0, 60])
    ax_all_twin.set_yticks([0, 1])

    fig_all.savefig(f"oF_vs_f_scaled_aoa_and_x4_{case_name}.pdf", bbox_inches="tight")
    fig_force.savefig(f"oF_vs_f_scaled_attached_cl_{case_name}.pdf", bbox_inches="tight")