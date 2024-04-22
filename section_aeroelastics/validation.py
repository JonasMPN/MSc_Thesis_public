from calculations import AeroForce, Rotations
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from os import listdir
from os.path import join
from calculations import ThreeDOFsAirfoil, AeroForce
import numpy as np
from utils_plot import MosaicHandler
from defaults import DefaultsPlots
from helper_functions import Helper
import json
helper = Helper()


def run_BeddoesLeishman(
        dir_profile: str, period_res: int=100,
        A1: float=0.3, A2: float=0.7, b1: float=0.14, b2: float=0.53
):
    dir_unsteady_data = join(dir_profile, "unsteady")
    us_files = listdir(dir_unsteady_data)
    oscillations = []
    for file in us_files:
        split = file.split("_")  # files have to be called unsteady_'amplitude'_'mean'.txt
        oscillations.append((float(split[1]), float(split[2].split(".")[0])))
    oscillations = [(10, 14)]
    aero = AeroForce(dir_profile)
    aero._prepare("BL")

    inflow_speed = 90.3*0.3048
    chord = 0.457
    k = 0.094
    n_periods = 8
    omega = 2*k*inflow_speed/chord
    T = 2*np.pi/omega
    dt = T/period_res
    overall_res = n_periods*period_res
    for amplitude, mean in oscillations:
        t = dt*np.arange(overall_res)
        alpha = np.deg2rad(mean+amplitude*np.sin(omega*t))
        alpha_speed = np.deg2rad(mean+amplitude*omega*np.cos(omega*t))

        airfoil = ThreeDOFsAirfoil(dir_profile, t, None, None, chord, None, None, None, None, None, None)
        airfoil.pos = np.c_[np.zeros((overall_res, 2)), -alpha]
        airfoil.vel = np.c_[np.zeros((overall_res, 2)), -alpha_speed]
        airfoil.inflow = inflow_speed*np.c_[np.ones_like(t), np.zeros_like(t)]
        airfoil.inflow_angle = np.zeros_like(t)
        airfoil.dt = np.r_[t[1:]-t[:-1], t[1]]
        airfoil.density = 1
        airfoil.set_aero_calc("BL", A1=A1, A2=A2, b1=b1, b2=b2)

        aero._init_BL(airfoil, airfoil.pos[0, :], airfoil.vel[0, :], airfoil.inflow[0, :],
                      pitching_around=0.25, alpha_at=0.75)
        coeffs = np.zeros((t.size, 3))
        for i in range(overall_res):
            coeffs[i, :] = aero._BL(airfoil, i, A1=A1, A2=A2, b1=b1, b2=b2)
        save_to = join(dir_profile, "Beddoes_Leishman_validation", f"ampl_{int(amplitude)}_mean_{int(mean)}")
        airfoil.save(save_to)
        f_f_aero = join(save_to, "f_aero.dat")
        df = pd.read_csv(f_f_aero)
        df["C_d"] = -coeffs[:, 0]  # because C_t and C_d are defined in opposite directions
        df["C_l"] = coeffs[:, 1]
        df.to_csv(f_f_aero, index=None)
        


class ValidationPlotter(DefaultsPlots):
    def __init__(self) -> None:
        DefaultsPlots.__init__(self)

    def plot_preparation(
            self,
            dir_preparation: str,
            file_polar: str
            ):
        dir_plots = helper.create_dir(join(dir_preparation, "plots"))[0]
        df_section = pd.read_csv(join(dir_preparation, "sep_points.dat"))
        df_aerohor = pd.read_csv(join(dir_preparation, "f_lookup_1.dat"))
        df_polar = pd.read_csv(file_polar, delim_whitespace=True)
        alpha_0_aerohor = pd.read_csv(join(dir_preparation, "alpha_0.dat"))["alpha_0"].to_numpy()
        alpha_0_aerohor = np.deg2rad(alpha_0_aerohor)
        with open(join(dir_preparation, "alpha_0.json"), "r") as f:
            alpha_0_section = np.deg2rad(json.load(f)["alpha_0"])
            
        for f_type in ["f_t", "f_n"]:
            fig, ax = plt.subplots()
            ax.plot(df_aerohor["alpha_n"], df_aerohor[f"{f_type}"], **self.plot_settings[f"{f_type}_aerohor"])
            ax.plot(df_section["alpha"], df_section[f"{f_type}"], 
                    **self.plot_settings[f"{f_type}_section"])
            handler = MosaicHandler(fig, ax)
            handler.update(x_labels=r"$\alpha$ (°)", y_labels=rf"${{{f_type[0]}}}_{{{f_type[2]}}}$ (-)", legend=True)
            handler.save(join(dir_plots, f"{f_type}_model_comp.pdf"))
        
        rot = Rotations()
        coeffs_aerohor = np.c_[self._get_C_t(alpha_0_aerohor, df_aerohor["alpha_t"], df_aerohor["f_t"]),
                               self._get_C_n(alpha_0_aerohor, df_aerohor["alpha_n"], df_aerohor["f_n"])]
        rot_coeffs_aerohor = rot.rotate_2D(coeffs_aerohor, np.deg2rad(df_aerohor["alpha_n"].to_numpy()))
        
        coesffs_section = np.c_[self._get_C_t(alpha_0_section, df_section["alpha"], df_section["f_t"]),
                                self._get_C_n(alpha_0_section, df_section["alpha"], df_section["f_n"])]
        rot_coeffs_section = rot.rotate_2D(coesffs_section, np.deg2rad(df_section["alpha"].to_numpy()))

        for i, coeff_type in enumerate(["C_d", "C_l"]):
            sign = 1 if coeff_type == "C_l" else -1
            fig, ax = plt.subplots()
            ax.plot(df_polar["alpha"], df_polar[coeff_type], **self.plot_settings[f"{coeff_type}_meas"])
            ax.plot(df_aerohor["alpha_n"], sign*rot_coeffs_aerohor[:, i], 
                    **self.plot_settings[f"{coeff_type}_rec_aerohor"])
            ax.plot(df_section["alpha"], sign*rot_coeffs_section[:, i], 
                    **self.plot_settings[f"{coeff_type}_rec_section"])
            handler = MosaicHandler(fig, ax)
            handler.update(x_labels=r"$\alpha$ (°)", 
                           y_labels=rf"${{{coeff_type[0]}}}_{{{coeff_type[2]}}}$ (-)", legend=True)
            handler.save(join(dir_plots, f"{coeff_type}_model_comp.pdf"))

    @staticmethod
    def plot_model_comparison(dir_results: str):
        dir_plots = helper.create_dir(join(dir_results, "plots", "model_comp"))[0]
        df_aerohor = pd.read_csv(join(dir_results, "1_BL_sim_data.dat"))
        df_section_general = pd.read_csv(join(dir_results, "general.dat"))
        df_section = pd.read_csv(join(dir_results, "f_aero.dat"))

        aoas = [col for col in df_section if "alpha" in col]
        for aoa in aoas:
            df_section[aoa] = np.rad2deg(df_section[aoa])

        plot_model_compare_params = ["alpha_eff", "alpha_steady", "C_nc", "C_ni", "C_nsEq", "f_n", "f_t", "C_nf",
                                     "C_tf", "C_nv", "C_l", "C_d"]
        line_width = 1
        plt_settings = {
            "aerohor": {
                "color": "forestgreen", "lw": line_width, "label": "aerohor"
            },
            "section": {
                "color": "orangered", "lw": line_width, "label": "section"
            }
        }
        y_labels = {
            "alpha_eff": r"$\alpha_{\text{eff}}$ (°)",
            "alpha_steady": r"$\alpha_{\text{steady}}$ (°)",
            "C_nc": r"$C_{\text{nc}}$ (-)", 
            "C_ni": r"$C_{\text{ni}}$ (-)", 
            "C_nsEq": r"$C_{\text{nsEq}}$ (-)", 
            "C_nf": r"$C_{\text{nf}}$ (-)", 
            "C_tf": r"$C_{\text{tf}}$ (-)", 
            "C_nv": r"$C_{\text{nv}}$ (-)", 
            "C_d": r"$C_{\text{d}}$ (-)", 
            "C_l": r"$C_{\text{l}}$ (-)", 
            "f_n": r"$f_{\text{n}}$ (-)", 
            "f_t": r"$f_{\text{t}}$ (-)", 
        }
        t_aerohor = df_aerohor["t"]
        t_section = df_section_general["time"]
        for plot_param in plot_model_compare_params:
            fig, ax = plt.subplots()
            handler = MosaicHandler(fig, ax)
            ax.plot(t_aerohor, df_aerohor[plot_param], **plt_settings["aerohor"])
            ax.plot(t_section, df_section[plot_param], **plt_settings["section"])
            handler.update(x_labels="t (s)", y_labels=y_labels[plot_param], legend=True)
            handler.save(join(dir_plots, f"model_comp_{plot_param}.pdf"))
        
        # fig, ax = plt.subplots()
        # handler = MosaicHandler(fig, ax)
        # ax.plot(t_section[:-1], df_section["alpha_qs"][:-1], label="qs")
        # ax.plot(t_section[:-1], df_section["d_alpha_qs_dt"][:-1], label="dadt")
        # handler.update(x_labels="t (s)", legend=True)
        # handler.save(join(dir_plots, "extra_aoa.pdf"))

    def plot_meas_comparison(
        self,
        file_unsteady_data: str,
        dir_results: str,
        period_res: int,
        ):
        df_meas = pd.read_csv(file_unsteady_data)
        df_aerohor = pd.read_csv(join(dir_results, "1_BL_sim_data.dat"))
        df_section = pd.read_csv(join(dir_results, "f_aero.dat"))
        map_measurement = {"C_d": " Cdp", "C_l": " Cl"}
        dir_save = helper.create_dir(join(dir_results, "plots", "meas_comp"))[0]
        line_width = 1
        plt_settings = {
            "aerohor": {
                "color": "forestgreen", "lw": line_width, "label": "aerohor"
            },
            "section": {
                "color": "orangered", "lw": line_width, "label": "section"
            }
        }
        for coef in ["C_d", "C_l"]:
            fig, ax = plt.subplots()
            handler = MosaicHandler(fig, ax)
            ax.plot(df_meas[" AOA (deg)"], df_meas[map_measurement[coef]], **self.plot_settings[coef+"_meas"])
            ax.plot(df_aerohor["alpha_steady"][-period_res-1:], df_aerohor[coef][-period_res-1:], 
                    **plt_settings["section"])
            ax.plot(np.rad2deg(df_section["alpha_steady"][-period_res-1:]), df_section[coef][-period_res-1:], 
                    **plt_settings["aerohor"])
            handler.update(x_labels=r"$\alpha_{\text{steady}}$ (°)", y_labels=rf"${{{coef[0]}}}_{{{coef[2]}}}$ (-)",
                           legend=True)
            handler.save(join(dir_save, f"{coef}.pdf"))




    
    @staticmethod
    def _get_C_t(alpha_0: float, alpha: np.ndarray, f_t: np.ndarray, C_l_slope: float=2*np.pi):
        return C_l_slope*np.sin(np.deg2rad(alpha)-alpha_0)**2*np.sqrt(np.abs(f_t))*np.sign(f_t)

    @staticmethod
    def _get_C_n(alpha_0: float, alpha: np.ndarray, f_n: np.ndarray, C_l_slope: float=2*np.pi):
        return C_l_slope*np.sin(np.deg2rad(alpha)-alpha_0)*((1+np.sqrt(np.abs(f_n))*np.sign(f_n))/2)**2


if __name__ == "__main__":
    do = {
        "run_simulation": False,
        "plot_results": True
    }
    
    if do["run_simulation"]:
        run_BeddoesLeishman("data/OSU", period_res=100)

    if do["plot_results"]:
        plotter = ValidationPlotter()
        plotter.plot_preparation("data/OSU/Beddoes_Leishman_preparation", "data/OSU/polars.dat")
        plotter.plot_model_comparison("data/OSU/Beddoes_Leishman_validation/ampl_10_mean_14")
        plotter.plot_meas_comparison("data/OSU/unsteady/unsteady_10_14.txt",
                                     "data/OSU/Beddoes_Leishman_validation/ampl_10_mean_14",
                                     period_res=100)

