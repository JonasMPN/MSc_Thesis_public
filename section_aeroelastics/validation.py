from calculations import AeroForce, Rotations
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from os import listdir
from os.path import join
from calculations import ThreeDOFsAirfoil, AeroForce, TimeIntegration, StructForce
from post_calculations import PostCaluculations
import numpy as np
from utils_plot import MosaicHandler
from defaults import DefaultsPlots, DefaultStructure
from helper_functions import Helper
import json
helper = Helper()


def run_BeddoesLeishman(
        dir_profile: str, index_unsteady_data: int,
        k: float, inflow_speed: float, chord: float, amplitude: float, mean: float, 
        period_res: int=100,
        A1: float=0.3, A2: float=0.7, b1: float=0.14, b2: float=0.53
        ):
    aero = AeroForce(dir_profile)
    aero._prepare("BL")

    n_periods = 8
    overall_res = n_periods*period_res
    omega = 2*k*inflow_speed/chord
    T = 2*np.pi/omega
    dt = T/period_res
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
    airfoil.set_aero_calc("BL", A1=A1, A2=A2, b1=b1, b2=b2, pitching_around=0.25, alpha_at=0.75)

    aero._init_BL(airfoil, pitching_around=0.25, alpha_at=0.75)
    coeffs = np.zeros((t.size, 3))
    for i in range(overall_res):
        coeffs[i, :] = aero._BL(airfoil, i, A1=A1, A2=A2, b1=b1, b2=b2)
    
    dir_res = helper.create_dir(join(dir_profile, "Beddoes_Leishman_validation", str(index_unsteady_data)))[0]
    airfoil.save(dir_res)
    f_f_aero = join(dir_res, "f_aero.dat")
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
        # df_aerohor = pd.read_csv(join(dir_preparation, "f_lookup_1.dat"))
        df_polar = pd.read_csv(file_polar, delim_whitespace=True)
        # alpha_0_aerohor = pd.read_csv(join(dir_preparation, "alpha_0.dat"))["alpha_0"].to_numpy()
        # alpha_0_aerohor = np.deg2rad(alpha_0_aerohor)
        with open(join(dir_preparation, "alpha_0.json"), "r") as f:
            alpha_0_section = np.deg2rad(json.load(f)["alpha_0"])
            
        for f_type in ["f_t", "f_n"]:
            fig, ax = plt.subplots()
            # ax.plot(df_aerohor["alpha_n"], df_aerohor[f"{f_type}"], **self.plot_settings[f"{f_type}_aerohor"])
            ax.plot(df_section["alpha"], df_section[f"{f_type}"], 
                    **self.plot_settings[f"{f_type}_section"])
            handler = MosaicHandler(fig, ax)
            handler.update(x_labels=r"$\alpha$ (°)", y_labels=rf"${{{f_type[0]}}}_{{{f_type[2]}}}$ (-)", legend=True)
            handler.save(join(dir_plots, f"{f_type}_model_comp.pdf"))
        
        rot = Rotations()
        # coeffs_aerohor = np.c_[self._get_C_t(alpha_0_aerohor, df_aerohor["alpha_t"], df_aerohor["f_t"]),
        #                        self._get_C_n(alpha_0_aerohor, df_aerohor["alpha_n"], df_aerohor["f_n"])]
        # rot_coeffs_aerohor = rot.rotate_2D(coeffs_aerohor, np.deg2rad(df_aerohor["alpha_n"].to_numpy()))
        
        coesffs_section = np.c_[self._get_C_t(alpha_0_section, df_section["alpha"], df_section["f_t"]),
                                self._get_C_n(alpha_0_section, df_section["alpha"], df_section["f_n"])]
        rot_coeffs_section = rot.rotate_2D(coesffs_section, np.deg2rad(df_section["alpha"].to_numpy()))

        for i, coeff_type in enumerate(["C_d", "C_l"]):
            sign = 1 if coeff_type == "C_l" else -1
            fig, ax = plt.subplots()
            ax.plot(df_polar["alpha"], df_polar[coeff_type], **self.plot_settings[f"{coeff_type}_meas"])
            # ax.plot(df_aerohor["alpha_n"], sign*rot_coeffs_aerohor[:, i], 
            #         **self.plot_settings[f"{coeff_type}_rec_aerohor"])
            ax.plot(df_section["alpha"], sign*rot_coeffs_section[:, i], 
                    **self.plot_settings[f"{coeff_type}_rec_section"])
            handler = MosaicHandler(fig, ax)
            handler.update(x_labels=r"$\alpha$ (°)", 
                           y_labels=rf"${{{coeff_type[0]}}}_{{{coeff_type[2]}}}$ (-)", legend=True)
            handler.save(join(dir_plots, f"{coeff_type}_model_comp.pdf"))

    @staticmethod
    def plot_model_comparison(dir_results: str):
        dir_plots = helper.create_dir(join(dir_results, "plots", "model_comp"))[0]
        # df_aerohor = pd.read_csv(join(dir_results, "aerohor_res.dat"))
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
        # t_aerohor = df_aerohor["t"]
        t_section = df_section_general["time"]
        for plot_param in plot_model_compare_params:
            fig, ax = plt.subplots()
            handler = MosaicHandler(fig, ax)
            # ax.plot(t_aerohor, df_aerohor[plot_param], **plt_settings["aerohor"])
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
        df_meas = pd.read_csv(file_unsteady_data, delim_whitespace=True)
        # df_aerohor = pd.read_csv(join(dir_results, "aerohor_res.dat"))
        df_section = pd.read_csv(join(dir_results, "f_aero.dat"))
        # map_measurement = {"C_d": " Cdp", "C_l": " Cl"}
        map_measurement = {"alpha": "AOA", "C_l": "CL", "C_d": "CD"}
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
            ax.plot(df_meas[map_measurement["alpha"]], df_meas[map_measurement[coef]], 
                    **self.plot_settings[coef+"_meas"])
            # ax.plot(df_aerohor["alpha_steady"][-period_res-1:], df_aerohor[coef][-period_res-1:], 
            #         **plt_settings["section"])
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


class HHT_alpha_validator(DefaultsPlots, DefaultStructure):
    def __init__(self, dir_polar: str, dir_root_results: str) -> None:
        DefaultsPlots.__init__(self)
        DefaultStructure.__init__(self)
        self.dir_polar = dir_polar
        self.dir_root_results = dir_root_results
        self._struc_params = None
        self._response_wanted = None

    def set_structure(
            self,
            chord: float,
            mass: float=None,
            mom_inertia: float=None,
            stiffness_edge: float=None,
            stiffness_flap: float=None,
            stiffness_tors: float=None,
            damping_edge: float=None,
            damping_flap: float=None,
            damping_tors: float=None,
    ):
        skip = ["self", "chord", "mass", "mom_inertia"]
        struc_params = {param: val for param, val in locals().items() if param not in ["skip"]+skip}
        response_wanted = {i: [] for i in range(3)}
        for i, (param, val) in enumerate(struc_params.items()):
            if val == None:
                struc_params[param] = 1
                response_wanted[i%3].append(False)
            else:
                response_wanted[i%3].append(True)
        self._response_wanted = {i: any(val_give) for i, val_give in response_wanted.items()}
        if (self._response_wanted[0] or self._response_wanted[1]) and mass is None:
            raise ValueError("'mass' must be set when calculating the response in the edgewise or flapwise direction.")
        if self._response_wanted[2] and mom_inertia is None:
            raise ValueError("'mom_inertia' must be set when calculating the response in the torsional direction.")
        
        self._struc_params = struc_params|{"chord": chord, 
                                           "mass": mass if mass is not None else 1, 
                                           "mom_inertia": mom_inertia if mom_inertia is not None else 1}
        self._struct_force = StructForce(**struc_params)

    def simulate(
            self,
            case_name: str,
            alpha: float,
            time: np.ndarray,
            init_pos: np.ndarray,
            init_vel: np.ndarray,
            init_accel: np.ndarray,
            f_edge: np.ndarray=None,
            f_flap: np.ndarray=None,
            f_tors: np.ndarray=None
    ):
        use = ["f_edge", "f_flap", "f_tors"]
        force = {param: val for param, val in locals().items() if param in use}
        f_ext = np.zeros((time.size, 3))
        f_response_wanted = {i: True for i in range(3)}
        for i, (param, val) in enumerate(force.items()):
            if val is None:
                force[param] = 1
                f_response_wanted[i] = False
            else:
                f_ext[:, i] = val
        for i, direction in enumerate(["edgewise", "flapwise", "torsional"]):
            if self._response_wanted[i] and not f_response_wanted[i]:
                raise ValueError(f"{direction} force time series is missing.")
            elif not self._response_wanted[i] and f_response_wanted[i]:
                raise ValueError(f"{direction} structural parameter(s) is(are) missing.")
            
        dir_save = helper.create_dir(join(self.dir_root_results, case_name))[0]
        airfoil = ThreeDOFsAirfoil(dir_polar=self.dir_polar, time=time, **self._struc_params)
        airfoil.pos[0, :] = init_pos
        airfoil.vel[0, :] = init_vel
        airfoil.accel[0, :] = init_accel
        
        airfoil.aero = f_ext
        time_integrator = TimeIntegration()
        time_integrator._init_HHT_alpha(airfoil, alpha, t[1])
        airfoil.damp = np.zeros((t.size, 3))
        airfoil.stiff = np.zeros((t.size, 3))

        for i in range(t.size-1):
            pos, vel, accel = time_integrator._HHT_alpha(airfoil, i, t[1])
            # print(airfoil.pos)
            airfoil.pos[i+1, :] = pos
            airfoil.vel[i+1, :] = vel
            airfoil.accel[i+1, :] = accel

            #todo check coordinate sytsem under rotation (also check for force)
            airfoil.damp[i, :], airfoil.stiff[i, :] = self._struct_force._linear(airfoil, i)
            
        i = t.size-1
        airfoil.damp[i, :], airfoil.stiff[i, :] = self._struct_force._linear(airfoil, i)
        airfoil.inflow = np.zeros((time.size, 2))
        airfoil._save(dir_save)
        with open(join(dir_save, "section_data.json"), "w") as f:
            json.dump(self._struc_params, f, indent=4)

        post_calc = PostCaluculations(dir_save, "alpha_steady")
        post_calc.project_data()
        post_calc.work()
        post_calc.kinetic_energy()
        post_calc.potential_energy()

    def plot(
        self, 
        case_name: str, 
        solution: dict[str, np.ndarray], 
        directions_wanted: list=["edge", "flap", "tors"],
        error_type: str="absolute"):
        dir_res = join(self.dir_root_results, case_name, "plots")
        df_general = pd.read_csv(join(dir_res, self._dfl_filenames["general"])) 
        time = df_general["time"]
        df_f_aero = pd.read_csv(join(dir_res, self._dfl_filenames["f_aero"]))
        df_f_struct = pd.read_csv(join(dir_res, self._dfl_filenames["f_structural"]))
        df_work = pd.read_csv(join(dir_res, self._dfl_filenames["work"]))
        df_e_kin = pd.read_csv(join(dir_res, self._dfl_filenames["e_kin"]))
        df_e_pot = pd.read_csv(join(dir_res, self._dfl_filenames["e_pot"]))

        axs_labels_pos = [["pos", "aero"],
                          ["damp", "stiff"]] 
        fig, axs = plt.subplot_mosaic(axs_labels_pos, tight_layout=True)
        fig_err, axs_err = plt.subplot_mosaic(axs_labels_pos, tight_layout=True)
        y_labels = {"pos": "position (m), (°)", "aero": r"$f_{\text{external}}$ (N), (Nm)",
                    "damp": r"$f_{\text{damping}}$ (N), (Nm)", "stiff": r"$f_{\text{stiffness}}$ (N), (Nm)"}
        y_labels_err = {ax: f"{error_type} error {label[:label.find("(")-1]}" for ax, label in y_labels.items()}

        dfs = {"pos": df_general, "aero": df_f_aero, "damp": df_f_struct, "stiff": df_f_struct}
        legend = self._plot(axs, axs_err, dfs, time, directions_wanted, error_type)
        handler = MosaicHandler(fig, axs)
        handler.update(legend=legend, y_labels=y_labels, x_labels="time (s)")
        handler.save(join(dir_res, "pos.pdf"))
        
        handler_err = MosaicHandler(fig_err, axs_err)
        handler_err.update(legend=legend, y_labels=y_labels_err, x_labels="time (s)")
        handler_err.save(join(dir_res, "error_pos.pdf"))

        fig, axs = plt.subplot_mosaic([["total", "work"],
                                       ["e_kin", "e_pot"]], tight_layout=True)
        y_labels = {"total": "Energy (Nm)", "work": "Work (Nm)",
                   "e_kin": f"Kinetic energy (Nm)", "e_pot": "Potential energy (Nm)"}
        y_labels_err = {ax: f"{error_type} error {label[:label.find("(")-1]}" for ax, label in y_labels.items()}
        df_total = pd.concat([df_e_kin.sum(axis=1), df_e_pot.sum(axis=1)], keys=["e_kin", "e_pot"], axis=1)
        df_total["e_total"] = df_total.sum(axis=1)
        dfs = {"total": df_total, "work": df_work, "e_kin": df_e_kin, "pot": df_e_pot}
        directions_wanted += ["kin", "pot", "total"]
        legend = self._plot(axs, axs_err, dfs, time, directions_wanted, error_type)

        handler = MosaicHandler(fig, axs)
        handler.update(legend=legend, y_labels=y_labels, x_labels="time (s)")
        handler.save(join(dir_res, "energy.pdf"))

        handler_err = MosaicHandler(fig_err, axs_err)
        handler_err.update(legend=legend, y_labels=y_labels_err, x_labels="time (s)")
        handler_err.save(join(dir_res, "error_energy.pdf"))
    
    @staticmethod
    def _plot(
        axs: dict[str], 
        axs_err: dict[str], 
        dfs: dict[str, pd.DataFrame], 
        time: np.ndarray,
        directions_wanted: list[str],
        error_type: str="absolute"):
        legend = {}
        for (param_type, ax), ax_err in zip(axs.items(), axs_err.values()):
            df = dfs[param_type]
            for col in df.columns:
                correct_type = param_type in col
                col_of_interest = np.count_nonzero(df[col])
                correct_direction = any([direction in col for direction in directions_wanted])
                if not all([correct_type, col_of_interest, correct_direction]):
                    continue
                try:
                    solution[col]
                except KeyError:
                    raise KeyError(f"Solution missing for '{col}'. Add those to solution['{col}'].")
                label = col[col.rfind("_")+1:]
                ax.plot(time, df[col], label=label)
                ax.plot(time, solution[col], "k", linestyle="--", label="sol")
                match error_type:
                    case "absolute":
                            ax_err.plot(time, solution[col]-df[col], label=label)
                    case "relative":
                            ax_err.plot(time, (solution[col]-df[col])/solution[col], label=label)
                    case _:
                            raise ValueError(f"'error_type' can be 'absolute' or 'relative' but was {error_type}.")
                legend[param_type] = True
        return legend
    

if __name__ == "__main__":
    do = {
        "run_simulation": False,
        "plot_results": False,
        "calc_HHT_alpha_response": True,
        "plot_HHT_alpha_response": True
    }
    dir_airfoil = "data/FFA_WA3_221"
    dir_HHT_alpha_validation = "data/HHT_alpha_validation"
    HHT_alpha_case = "test"
    period_res = 100

    if do["run_simulation"]:
        df_cases = pd.read_csv(join(dir_airfoil, "unsteady", "info.dat"))
        for i, row in df_cases.iterrows():
            run_BeddoesLeishman(dir_airfoil, index_unsteady_data=i,
                                k=row["k"], inflow_speed=row["inflow_speed"], chord=row["chord"], 
                                amplitude=row["amplitude"], mean=row["mean"], period_res=period_res)

    if do["plot_results"]:
        plotter = ValidationPlotter()
        plotter.plot_preparation(join(dir_airfoil,"Beddoes_Leishman_preparation"), join(dir_airfoil, "polars.dat"))
        dir_validations = join(dir_airfoil, "Beddoes_Leishman_validation")
        df_cases = pd.read_csv(join(dir_airfoil, "unsteady", "info.dat"))
        for case_name in listdir(dir_validations):
            dir_case = join(dir_validations, case_name)
            plotter.plot_model_comparison(dir_case)
            plotter.plot_meas_comparison(join(dir_airfoil, "unsteady", df_cases["filename"].iat[int(case_name)]),
                                         dir_case, period_res=period_res)
        
    if do["calc_HHT_alpha_response"] or do["plot_HHT_alpha_response"]:
        T = 10
        dt = 0.1
        t = dt*np.arange(T/dt+1)
        alpha = 0.1

        init_pos = (1, 0, 0)
        init_vel = (0, 0, 0)
        init_accel = (0, 0, 0)

        force_edge = np.zeros_like(t)
        force_flap = None
        force_tors = None

        chord = 1
        mass = .1
        mom_inertia = None

        stiffness_edge = 1
        stiffness_flap = None
        stiffness_tors = None

        damping_edge = 0
        damping_flap = None
        damping_tors = None

    if do["calc_HHT_alpha_response"]:
        validator = HHT_alpha_validator(dir_airfoil, dir_HHT_alpha_validation)
        validator.set_structure(chord=chord,mass=mass,mom_inertia=mom_inertia,stiffness_edge=stiffness_edge,
                                stiffness_flap=stiffness_flap,stiffness_tors=stiffness_tors,damping_edge=damping_edge,
                                damping_flap=damping_flap,damping_tors=damping_tors)   
        validator.simulate(case_name=HHT_alpha_case, time=t, alpha=alpha, init_pos=init_pos,init_vel=init_vel,
                           init_accel=init_accel, f_edge=force_edge, f_flap=force_flap, f_tors=force_tors)
        
    if do["plot_HHT_alpha_response"]:
        validator = HHT_alpha_validator(dir_airfoil, dir_HHT_alpha_validation)
        
        solution = {
            "pos_edge": np.cos(np.sqrt(stiffness_edge/mass)*t),
            "stiff_edge": -np.cos(np.sqrt(stiffness_edge/mass)*t),
            # "e_total": 
        }
        validator.plot(case_name=HHT_alpha_case, solution=solution)