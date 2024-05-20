from utils_plot import Shapes, PlotPreparation, AnimationPreparation, MosaicHandler
from calculations import Rotations
from defaults import DefaultPlot, DefaultStructure
import pandas as pd
import numpy as np
import json
import matplotlib.animation as animation
import matplotlib.pyplot as plt
import matplotlib
from os.path import join
from os import listdir
from scipy.signal import find_peaks
from helper_functions import Helper
from typing import Callable
from ast import literal_eval
from seaborn import heatmap
helper = Helper()


class Plotter(DefaultStructure, DefaultPlot, PlotPreparation):
    """Utility class that plots results. Requires a directory to have a certain text files holding data about a 
    simulation. The name of these files is given in the parent class DefaultStructure.
    """
    
    def __init__(self, file_profile: str, dir_data: str, dir_plots: str, dt_res: float=None) -> None:
        """Initialises a plotter instance.

        :param file_profile: Path from the current working directory to a text file holding the profile coordinates. The
        columns must be (x, y). 
        :type file_profile: str
        :param dir_data: Path from the current working directory to the directory containing the simulation results.
        :type dir_data: str
        :param dir_plots: Path from the current working directory to the directory where the plots are saved.
        :type dir_plots: str
        """
        DefaultStructure.__init__(self)
        DefaultPlot.__init__(self)
        PlotPreparation.__init__(self)
        
        self.profile = pd.read_csv(file_profile, delim_whitespace=True).to_numpy()
        self.df_f_aero = pd.read_csv(join(dir_data, self._dfl_filenames["f_aero"]))
        self.df_f_structural = pd.read_csv(join(dir_data, self._dfl_filenames["f_structural"]))
        self.df_general = pd.read_csv(join(dir_data, self._dfl_filenames["general"]))
        self.df_power = pd.read_csv(join(dir_data, self._dfl_filenames["power"]))
        self.df_e_kin = pd.read_csv(join(dir_data, self._dfl_filenames["e_kin"]))
        self.df_e_pot = pd.read_csv(join(dir_data, self._dfl_filenames["e_pot"]))
        self.time = self.df_general["time"].to_numpy()
        with open(join(dir_data, self._dfl_filenames["section_data"]), "r") as f:
            self.section_data = json.load(f)

        self.dir_plots = dir_plots
        helper.create_dir(self.dir_plots)
        self.dt_res = dt_res
        self._rot = Rotations()

    def force(self, equal_y: tuple[str]=None, trailing_every: int=2):
        """Plots the history of the airfoil movement and different forces.

        :param equal_y: Whether the force axes should have equal y scaling, defaults to None
        :type equal_y: tuple[str], optional
        :param trailing_every: How many quarter-chord points to skip while plotting, defaults to 2
        :type trailing_every: int, optional
        :return: None
        :rtype: None
        """
        fig, axs, handler = self._prepare_force_plot(equal_y)
        
        # get final position of the profile. Displace it such that the qc is at (0, 0) at t=0.
        pos = self.df_general[["pos_x", "pos_y", "pos_tors"]].to_numpy()
        rot = self._rot.active_2D(pos[-1, 2])
        rot_profile = rot@(self.profile[:, :2]-np.asarray([self.section_data["chord"]/4, 0])).T
        rot_profile += np.asarray([[pos[-1, 0]], [pos[-1, 1]]])
        axs["profile"].plot(rot_profile[0, :], rot_profile[1, :], **self.plt_settings["profile"])
        axs["profile"].plot(pos[::trailing_every, 0], pos[::trailing_every, 1], **self.plt_settings["qc_trail"])
        
        # grab all angles of attack from the data
        aoas = self._get_aoas(self.df_f_aero)
        # prepare dictionary of dataframes to plot
        dfs = {"aoa": self.df_f_aero, "aero": self.df_f_aero, "damp": self.df_f_structural, 
               "stiff": self.df_f_structural}
        
        plot = {
            "aoa": aoas,
            "aero": ["aero_edge", "aero_flap", "aero_mom"],
            "damp": ["damp_edge", "damp_flap", "damp_tors"],
            "stiff": ["stiff_edge", "stiff_flap", "stiff_tors"]
        }
        def param_name(param: str):
            return param if "alpha" in param else param[param.rfind("_")+1:]  
        axs = self._plot_to_mosaic(axs, plot, dfs, param_name)
        handler.update(legend=True)
        fig.savefig(join(self.dir_plots, "forces.pdf"))

    def energy(self, equal_y: tuple[str]=None, trailing_every: int=2):
        """Plots the history of the airfoil movement and different energies/power.

        :param equal_y: Whether the force axes should have equal y scaling, defaults to None
        :type equal_y: tuple[str], optional
        :param trailing_every: How many quarter-chord points to skip while plotting, defaults to 2
        :type trailing_every: int, optional
        :return: None
        :rtype: None
        """
        fig, axs, handler = self._prepare_energy_plot(equal_y)
        
        # get final position of the profile. Displace it such that the qc is at (0, 0) at t=0.
        pos = self.df_general[["pos_x", "pos_y", "pos_tors"]].to_numpy()
        rot = self._rot.active_2D(pos[-1, 2])
        rot_profile = rot@(self.profile[:, :2]-np.asarray([self.section_data["chord"]/4, 0])).T
        rot_profile += np.asarray([[pos[-1, 0]], [pos[-1, 1]]])
        axs["profile"].plot(rot_profile[0, :], rot_profile[1, :], **self.plt_settings["profile"])
        axs["profile"].plot(pos[::trailing_every, 0], pos[::trailing_every, 1], **self.plt_settings["qc_trail"])
        
        df_total = pd.concat([self.df_e_kin.sum(axis=1), self.df_e_pot.sum(axis=1)], keys=["e_kin", "e_pot"], axis=1)
        df_total["e_total"] = df_total.sum(axis=1)
        dfs = {"total": df_total, "power": self.df_power, "kinetic": self.df_e_kin, 
               "potential": self.df_e_pot}
        
        plot = {
            "total": ["e_total", "e_kin", "e_pot"],
            "power": ["aero_drag", "aero_lift", "aero_mom", "damp_edge", "damp_flap", "damp_tors"],
            "kinetic": ["edge", "flap", "tors"],
            "potential": ["edge", "flap", "tors"]
        }        
        def param_name(param: str):
            return param if "damp" not in param and "stiff" not in param else param[param.rfind("_")+1:]
        axs = self._plot_to_mosaic(axs, plot, dfs, param_name)
        handler.update(legend=True)
        fig.savefig(join(self.dir_plots, "energy.pdf"))
    
    def Beddoes_Leishman(self, equal_y: tuple[str]=None, trailing_every: int=2):
        """Plots the history of the airfoil movement and different BL parameters.

        :param equal_y: Whether the force axes should have equal y scaling, defaults to None
        :type equal_y: tuple[str], optional
        :param trailing_every: How many quarter-chord points to skip while plotting, defaults to 2
        :type trailing_every: int, optional
        :return: None
        :rtype: None
        """
        coeffs = self._get_force_and_moment_coeffs(self.df_f_aero)
        fig, axs, handler = self._prepare_BL_plot([*coeffs.keys()], equal_y)
        
        # get final position of the profile. Displace it such that the qc is at (0, 0) at t=0.
        pos = self.df_general[["pos_x", "pos_y", "pos_tors"]].to_numpy()
        rot = self._rot.active_2D(pos[-1, 2])
        rot_profile = rot@(self.profile[:, :2]-np.asarray([self.section_data["chord"]/4, 0])).T
        rot_profile += np.asarray([[pos[-1, 0]], [pos[-1, 1]]])
        axs["profile"].plot(rot_profile[0, :], rot_profile[1, :], **self.plt_settings["profile"])
        axs["profile"].plot(pos[::trailing_every, 0], pos[::trailing_every, 1], **self.plt_settings["qc_trail"])
        
        # grab all angles of attack from the data
        aoas = self._get_aoas(self.df_f_aero)
        
        plot = {"aoa": aoas}|coeffs
        dfs = {ax: self.df_f_aero for ax in plot.keys()}
        def param_name(param: str):
            return param
        axs = self._plot_to_mosaic(axs, plot, dfs, param_name)
        handler.update(legend=True)
        fig.savefig(join(self.dir_plots, "BL.pdf"))

    def _plot_to_mosaic(
            self,
            axes: dict[str, matplotlib.axes.Axes],
            plot: dict[str, list[str]],
            data: dict[str, pd.DataFrame|dict],
            map_column_to_settings: Callable,
            apply: dict[str, Callable]={"aoa": np.rad2deg}) -> dict[str, matplotlib.axes.Axes]:
        apply_to_axs = apply.keys()
        for ax, cols in plot.items():
            for col in cols:
                try: 
                    self.plt_settings[map_column_to_settings(col)]
                except KeyError:
                    raise NotImplementedError(f"Default plot styles for '{col}' are missing.")
                if ax in apply_to_axs:
                    axes[ax].plot(self.time, apply[ax](data[ax][col]), **self.plt_settings[map_column_to_settings(col)])
                else:
                    axes[ax].plot(self.time, data[ax][col], **self.plt_settings[map_column_to_settings(col)])
        return axes


class Animator(DefaultStructure, Shapes, AnimationPreparation):
    """Utility class that animates the results, meaning it creates an animation of the time series of the
    position of the airfoil (plotted as the whole airfoil moving) and of the forces or energies. It also 
    animates the aerodynamic forces in the plot of the moving airfoil.
    """
    def __init__(self, file_profile: str, dir_data: str, dir_plots: str) -> None:
        """Initialises an animation instance.

        :param file_profile: Path from the current working directory to a text file holding the profile coordinates. The
        columns must be (x, y). 
        :type file_profile: str
        :param dir_data: Path from the current working directory to the directory containing the simulation results.
        :type dir_data: str
        :param dir_plots: Path from the current working directory to the directory where the plots are saved.
        :type dir_plots: str
        """
        DefaultStructure.__init__(self)
        DefaultPlot.__init__(self)
        Shapes.__init__(self)
        AnimationPreparation.__init__(self)
        
        self.profile = pd.read_csv(file_profile, delim_whitespace=True).to_numpy()
        self.df_f_aero = pd.read_csv(join(dir_data, self._dfl_filenames["f_aero"]))
        self.df_f_structural = pd.read_csv(join(dir_data, self._dfl_filenames["f_structural"]))
        self.df_general = pd.read_csv(join(dir_data, self._dfl_filenames["general"]))
        self.df_power = pd.read_csv(join(dir_data, self._dfl_filenames["power"]))
        self.df_e_kin = pd.read_csv(join(dir_data, self._dfl_filenames["e_kin"]))
        self.df_e_pot = pd.read_csv(join(dir_data, self._dfl_filenames["e_pot"]))
        self.time = self.df_general["time"].to_numpy()
        with open(join(dir_data, self._dfl_filenames["section_data"]), "r") as f:
            self.section_data = json.load(f)

        self.dir_plots = dir_plots
        helper.create_dir(self.dir_plots)
        self._rot = Rotations()
    
    def force(
            self,
            angle_lift: str,
            arrow_scale_forces: float=1,
            arrow_scale_moment: float=1,
            plot_qc_trailing_every: int=2,
            keep_qc_trailing: int=40,
            time_steps: int=0,
            frame_skip_timesteps: int=1):
        qc_pos = self.df_general[["pos_x", "pos_y"]].to_numpy()
        profile = np.zeros((self.time.size, *self.profile.shape))
        norm_moments = self.df_f_aero["aero_mom"]/np.abs(self.df_f_aero["aero_mom"]).max()
        mom_arrow_res = 40
        mom_arrow = np.zeros((self.time.size, mom_arrow_res+3, 2))
        prof = self.profile-np.c_[0.25*self.section_data["chord"], 0]
        for i, (angle, moment) in enumerate(zip(self.df_general["pos_tors"], norm_moments)):
            rot_mat = self._rot.active_2D(angle)
            profile[i, :, :] = (rot_mat@prof.T).T+qc_pos[i, :]

            trailing_edge = (rot_mat@np.c_[(0.75-arrow_scale_moment)*self.section_data["chord"], 0].T).T+qc_pos[i, :]
            moment_arrow = rot_mat@self.circle_arrow(180*moment)*arrow_scale_moment
            mom_arrow[i, :, :] = moment_arrow.T+trailing_edge.squeeze()
        
        ids = np.arange(self.time.size)-1
        trailing_idx_from = ids-(keep_qc_trailing+ids%plot_qc_trailing_every)
        trailing_idx_from[trailing_idx_from < 0] = 0
        
        angle_aero_to_xyz = (self.df_general["pos_tors"]-self.df_f_aero[angle_lift]).to_numpy()
        aero_force = np.c_[self.df_f_aero["aero_drag"], self.df_f_aero["aero_lift"], np.zeros(self.time.size)]
        force_arrows = self._rot.project_separate(aero_force, angle_aero_to_xyz)*arrow_scale_forces
        
        dfs = {"general": self.df_general, "f_aero": self.df_f_aero, "f_structural": self.df_f_structural}
        fig, plt_lines, plt_arrows, aoas = self._prepare_force_animation(dfs)

        data_lines = {linename: self.df_f_aero[linename] for linename in ["aero_edge", "aero_flap", "aero_mom"]} |\
                     {linename: self.df_f_aero[linename] for linename in aoas} |\
                     {linename: self.df_f_structural[linename] for linename in ["damp_edge", "damp_flap", "damp_tors"]+
                                                                            ["stiff_edge", "stiff_flap", "stiff_tors"]}
        data_lines = data_lines | {"qc_trail": qc_pos,
                                   "profile": profile, # data with dict[line: data_for_line]
                                   "mom": mom_arrow}
        data_arrows = {"drag": force_arrows[:, :2], "lift": force_arrows[:, 2:4]}
        
        def update(i: int):
            i = frame_skip_timesteps*i+1
            for linename, data in data_lines.items():
                match linename:
                    case "profile"|"mom":
                        plt_lines[linename].set_data(data[i, :, 0], data[i, :, 1])
                    case "qc_trail":
                        plt_lines[linename].set_data(data[trailing_idx_from[i]:i:plot_qc_trailing_every, 0], 
                                                     data[trailing_idx_from[i]:i:plot_qc_trailing_every, 1])
                    case _:
                        plt_lines[linename].set_data(self.time[:i], data[:i])
            for arrow_name, data in data_arrows.items():
                plt_arrows[arrow_name].set_data(x=data_lines["qc_trail"][i, 0], y=data_lines["qc_trail"][i, 1], 
                                                dx=data[i, 0], dy=data[i, 1])
            rtrn = [*plt_lines.values()]+[*plt_arrows.values()]
            return tuple(rtrn)
        
        frames = int((self.time.size-1)/frame_skip_timesteps) if time_steps==0 else int(time_steps/frame_skip_timesteps)
        ani = animation.FuncAnimation(fig=fig, func=update, frames=frames, interval=15, blit=True)
        writer = animation.FFMpegWriter(fps=30)
        ani.save(join(self.dir_plots, "animation_force.mp4"), writer=writer)

    def energy(
            self,
            angle_lift: str,
            arrow_scale_forces: float=1,
            arrow_scale_moment: float=1,
            plot_qc_trailing_every: int=2,
            keep_qc_trailing: int=40):
        qc_pos = self.df_general[["pos_x", "pos_y"]].to_numpy()
        profile = np.zeros((self.time.size, *self.profile.shape))
        norm_moments = self.df_f_aero["aero_mom"]/np.abs(self.df_f_aero["aero_mom"]).max()
        mom_arrow_res = 40
        mom_arrow = np.zeros((self.time.size, mom_arrow_res+3, 2))
        prof = self.profile-np.c_[0.25*self.section_data["chord"], 0]
        for i, (angle, moment) in enumerate(zip(self.df_general["pos_tors"], norm_moments)):
            rot_mat = self._rot.active_2D(angle)
            profile[i, :, :] = (rot_mat@prof.T).T+qc_pos[i, :]

            trailing_edge = (rot_mat@np.c_[(0.75-arrow_scale_moment)*self.section_data["chord"], 0].T).T+qc_pos[i, :]
            moment_arrow = rot_mat@self.circle_arrow(180*moment)*arrow_scale_moment
            mom_arrow[i, :, :] = moment_arrow.T+trailing_edge.squeeze()
        
        ids = np.arange(self.time.size)-1
        trailing_idx_from = ids-(keep_qc_trailing+ids%plot_qc_trailing_every)
        trailing_idx_from[trailing_idx_from < 0] = 0
        
        angle_aero_to_xyz = (self.df_general["pos_tors"]-self.df_f_aero[angle_lift]).to_numpy()
        aero_force = np.c_[self.df_f_aero["aero_drag"], self.df_f_aero["aero_lift"], np.zeros(self.time.size)]
        force_arrows = self._rot.project_separate(aero_force, angle_aero_to_xyz)*arrow_scale_forces
        
        df_total = pd.concat([self.df_e_kin.sum(axis=1), self.df_e_pot.sum(axis=1)], keys=["e_kin", "e_pot"], axis=1)
        df_total["e_total"] = df_total.sum(axis=1)
        dfs = {"general": self.df_general, "e_tot": df_total, "power": self.df_power, "e_kin": self.df_e_kin, 
               "e_pot": self.df_e_pot}
        fig, plt_lines, plt_arrows = self._prepare_energy_animation(dfs)

        data_lines = {linename: df_total[linename] for linename in ["e_total", "e_kin", "e_pot"]} |\
                     {linename: self.df_power[linename] for linename in ["aero_drag", "aero_lift", "aero_mom", 
                                                                        "damp_edge", "damp_flap", "damp_tors"]} |\
                     {linename: self.df_e_kin[linename[linename.rfind("_")+1:]] for linename in ["kin_edge",
                                                                                                 "kin_flap", "kin_tors"]} |\
                     {linename: self.df_e_pot[linename[linename.rfind("_")+1:]] for linename in ["pot_edge", 
                                                                                                 "pot_flap", "pot_tors"]} 
        data_lines = data_lines | {"qc_trail": qc_pos, # data with dict[line: data_for_line]
                                   "profile": profile, 
                                   "mom": mom_arrow}
        data_arrows = {"drag": force_arrows[:, :2], "lift": force_arrows[:, 2:4]}
        
        def update(i: int):
            i += 1
            for linename, data in data_lines.items():
                match linename:
                    case "profile"|"mom":
                        plt_lines[linename].set_data(data[i, :, 0], data[i, :, 1])
                    case "qc_trail":
                        plt_lines[linename].set_data(data[trailing_idx_from[i]:i:plot_qc_trailing_every, 0], 
                                                     data[trailing_idx_from[i]:i:plot_qc_trailing_every, 1])
                    case _:
                        plt_lines[linename].set_data(self.time[:i], data[:i])
            for arrow_name, data in data_arrows.items():
                plt_arrows[arrow_name].set_data(x=data_lines["qc_trail"][i, 0], y=data_lines["qc_trail"][i, 1], 
                                                dx=data[i, 0], dy=data[i, 1])
            rtrn = [*plt_lines.values()]+[*plt_arrows.values()]
            return tuple(rtrn)
        
        ani = animation.FuncAnimation(fig=fig, func=update, frames=self.time.size-1, interval=15, blit=True)
        writer = animation.FFMpegWriter(fps=30)
        ani.save(join(self.dir_plots, "animation_energy.mp4"), writer=writer)
        

class HHTalphaPlotter(DefaultStructure):
    def __init__(self) -> None:
        DefaultStructure.__init__(self)

    @staticmethod
    def sol_and_sim(root_dir: str, parameters: list[str]):
        dir_plot = helper.create_dir(join(root_dir, "plots"))[0]
        schemes = listdir(root_dir)
        schemes.pop(schemes.index("plots"))
        case_numbers = []
        for i, scheme in enumerate(schemes):
            case_numbers.append(listdir(join(root_dir, scheme)))
            if len(case_numbers) == 2:
                if not case_numbers[0] == case_numbers[1]:
                    raise ValueError(f"Scheme '{schemes[i-1]}' and '{scheme[i]}' do not have the same case numbers: \n"
                                     f"Scheme '{schemes[i-1]}': {case_numbers[0]}\n"
                                     f"Scheme '{schemes[i]}': {case_numbers[1]}\n")
                case_numbers.pop(0)
                
        info = {}
        map_scheme_to_label = {
            "HHT-alpha": "o.g.",
            "HHT-alpha-adapted": "adpt."
        }
        for case_number in case_numbers[0]:
            fig, axs = plt.subplots(1, 2, figsize=(10, 5), tight_layout=True)
            axs = {"response": axs[0], "rel_err": axs[1]}
            for i, scheme in enumerate(schemes):
                current_dir = join(root_dir, scheme, case_number)
                df_sol = pd.read_csv(join(current_dir, "analytical_sol.dat"))   
                df_sim = pd.read_csv(join(current_dir, "general.dat"))
                df_err = pd.read_csv(join(current_dir, "errors.dat"))
                
                peaks_sol = find_peaks(df_sol["solution"])[0]
                peaks_sim = find_peaks(df_sim["pos_x"])[0]
                if peaks_sim[1]/peaks_sol[1] < 0.7:
                    peaks_sim = peaks_sim[1:]
                n_peaks = min(peaks_sim.size, peaks_sol.size)

                i_last_peak_sim = peaks_sim[:n_peaks-1][-1]+1
                i_last_peak_sol = peaks_sol[:n_peaks-1][-1]+1
                if i == 0:
                    axs["response"].plot(df_sol["time"][:i_last_peak_sol], 
                                         df_sol["solution"][:i_last_peak_sol], "k--", label="sol")
                axs["response"].plot(df_sim["time"][:i_last_peak_sim], 
                                     df_sim["pos_x"][:i_last_peak_sim], 
                                     label=r"HHT-$\alpha$"+f" {map_scheme_to_label[scheme]}")
                for param in parameters:
                    axs["rel_err"].plot(df_err[f"rel_err_{param}"]*100, label=f"{param} {map_scheme_to_label[scheme]}")

                with open(join(current_dir, "info.json"), "r") as f:
                    tmp_info = json.load(f)|{"scheme": scheme}
                
                if info == {}:
                    info = {param: [] for param in tmp_info.keys()}
                else:
                    for param, value in tmp_info.items():
                        info[param].append(value)
            handler = MosaicHandler(fig, axs)
            x_labels = {"response": "time (s)", "rel_err": "oscillation number (-)"}
            y_labels = {"response": "position (m)", "rel_err": "relative error (%)"}
            handler.update(x_labels=x_labels, y_labels=y_labels, legend=True)
            handler.save(join(dir_plot, f"{case_number}.pdf"))
        pd.DataFrame(info).to_csv(join(dir_plot, "info.dat"), index=None)

    @staticmethod
    def rotation(root_dir: str):
        dir_plot = helper.create_dir(join(root_dir, "plots"))[0]
        schemes = listdir(root_dir)
        schemes.pop(schemes.index("plots"))
        case_numbers = []
        for i, scheme in enumerate(schemes):
            case_numbers.append(listdir(join(root_dir, scheme)))
            if len(case_numbers) == 2:
                if not case_numbers[0] == case_numbers[1]:
                    raise ValueError(f"Scheme '{schemes[i-1]}' and '{scheme[i]}' do not have the same case numbers: \n"
                                     f"Scheme '{schemes[i-1]}': {case_numbers[0]}\n"
                                     f"Scheme '{schemes[i]}': {case_numbers[1]}\n")
                case_numbers.pop(0)
                
        info = {}
        map_scheme_to_label = {
            "HHT-alpha": "o.g.",
            "HHT-alpha-adapted": "adpt."
        }
        for case_number in case_numbers[0]:
            fig, axs = plt.subplots(1, 2, figsize=(10, 5), tight_layout=True)
            axs = {"split": axs[0], "joined": axs[1]}
            for i, scheme in enumerate(schemes):
                current_dir = join(root_dir, scheme, case_number)
                df_sol = pd.read_csv(join(current_dir, "analytical_sol.dat"))
                df_sim = pd.read_csv(join(current_dir, "general.dat"))

                for direction in ["x", "y", "tors"]:
                    if i == 0:
                        axs["split"].plot(df_sol["time"], df_sol[f"solution_{direction}"], "k")
                    axs["split"].plot(df_sim["time"], df_sim[f"pos_{direction}"], "--", label=f"{direction} {scheme}")
                
                if i == 0:
                    axs["joined"].plot(df_sol[f"solution_x"], df_sol[f"solution_y"], "ok", ms=0.5, label="eE")
                axs["joined"].plot(df_sim["pos_x"], df_sim["pos_y"], "or", ms=0.5, label=f"{scheme}")

                # with open(join(current_dir, "info.json"), "r") as f:
                #     tmp_info = json.load(f)|{"scheme": scheme}
                
                # if info == {}:
                #     info = {param: [] for param in tmp_info.keys()}
                # else:
                #     for param, value in tmp_info.items():
                #         info[param].append(value)
            
            handler = MosaicHandler(fig, axs)
            x_labels = {"split": "time (s)", "joined": "x (m)"}
            y_labels = {"split": "position (m), (deg)", "joined": "x (m)"}
            handler.update(x_labels=x_labels, y_labels=y_labels, legend=True)
            handler.save(join(dir_plot, f"{case_number}.pdf"))
        # pd.DataFrame(info).to_csv(join(dir_plot, "info.dat"), index=None)

    def plot_case(
        self, 
        case_dir: str, 
        solution: dict[str, np.ndarray], 
        directions_wanted: list=["edge", "flap", "tors"],
        error_type: str="absolute"):
        dir_plots = join(case_dir, "plots")
        df_general = pd.read_csv(join(case_dir, self._dfl_filenames["general"])) 
        time = df_general["time"]
        df_f_aero = pd.read_csv(join(case_dir, self._dfl_filenames["f_aero"]))
        df_f_struct = pd.read_csv(join(case_dir, self._dfl_filenames["f_structural"]))
        df_power = pd.read_csv(join(case_dir, self._dfl_filenames["power"]))
        df_e_kin = pd.read_csv(join(case_dir, self._dfl_filenames["e_kin"]))
        df_e_pot = pd.read_csv(join(case_dir, self._dfl_filenames["e_pot"]))

        axs_labels_pos = [["pos", "aero"],
                          ["damp", "stiff"]] 
        fig, axs = plt.subplot_mosaic(axs_labels_pos, tight_layout=True)
        fig_err, axs_err = plt.subplot_mosaic(axs_labels_pos, tight_layout=True)
        y_labels = {"pos": "position (m), (°)", "aero": r"$f_{\text{external}}$ (N), (Nm)",
                    "damp": r"$f_{\text{damping}}$ (N), (Nm)", "stiff": r"$f_{\text{stiffness}}$ (N), (Nm)"}
        y_labels_err = {ax: f"{error_type} error {label[:label.find("(")-1]}" for ax, label in y_labels.items()}

        helper.create_dir(dir_plots)
        dfs = {"pos": df_general, "aero": df_f_aero, "damp": df_f_struct, "stiff": df_f_struct}
        legend = self._plot(axs, axs_err, dfs, solution, time, directions_wanted, error_type)
        handler = MosaicHandler(fig, axs)
        handler.update(legend=legend, y_labels=y_labels, x_labels="time (s)")
        handler.save(join(dir_plots, "pos.pdf"))
        
        handler_err = MosaicHandler(fig_err, axs_err)
        handler_err.update(legend=legend, y_labels=y_labels_err, x_labels="time (s)")
        handler_err.save(join(dir_plots, "error_pos.pdf"))

        fig, axs = plt.subplot_mosaic([["total", "power"],
                                       ["e_kin", "e_pot"]], tight_layout=True)
        y_labels = {"total": "Energy (Nm)", "power": "Work (Nm)",
                   "e_kin": f"Kinetic energy (Nm)", "e_pot": "Potential energy (Nm)"}
        y_labels_err = {ax: f"{error_type} error {label[:label.find("(")-1]}" for ax, label in y_labels.items()}
        df_total = pd.concat([df_e_kin.sum(axis=1), df_e_pot.sum(axis=1)], keys=["e_kin", "e_pot"], axis=1)
        df_total["e_total"] = df_total.sum(axis=1)
        dfs = {"total": df_total, "power": df_power, "e_kin": df_e_kin, "pot": df_e_pot}
        directions_wanted += ["kin", "pot", "total"]
        legend = self._plot(axs, axs_err, dfs, solution, time, directions_wanted, error_type)

        handler = MosaicHandler(fig, axs)
        handler.update(legend=legend, y_labels=y_labels, x_labels="time (s)")
        handler.save(join(dir_plots, "energy.pdf"))

        handler_err = MosaicHandler(fig_err, axs_err)
        handler_err.update(legend=legend, y_labels=y_labels_err, x_labels="time (s)")
        handler_err.save(join(dir_plots, "error_energy.pdf"))
        
    @staticmethod
    def _plot(
        axs: dict[str], 
        axs_err: dict[str], 
        dfs: dict[str, pd.DataFrame], 
        solution: dict[str, np.ndarray],
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
    

class BLValidationPlotter(DefaultPlot, Rotations):
    def __init__(self) -> None:
        DefaultPlot.__init__(self)

    def plot_preparation(
            self,
            dir_preparation: str,
            file_polar: str,
            ):
        dir_preparation = dir_preparation.replace("\\", "/")
        BL_scheme = dir_preparation.split("/")[-1]
        scheme_id = {
            "BL_chinese": 3,
            "BL_openFAST_Cl_disc": 4,
            "BL_openFAST_Cd": 5,
        }[BL_scheme]
        
        if scheme_id in [3]:
            self._BL_preparation(dir_preparation, file_polar, scheme_id)
        elif scheme_id in [4]:
            self._BL_openFAST_preparation(dir_preparation, file_polar, "data/FFA_WA3_221/general_openFAST")

    def _BL_preparation(
            self,
            dir_preparation: str,
            file_polar: str,
            scheme_id: int
            ):
        f_data = {}
        for file in listdir(dir_preparation):
            if "sep_points" not in file:
                continue
            file_name = file.split(".")[0]
            f_data[f"{file_name[-1]}"] = pd.read_csv(join(dir_preparation, file))
        
        dir_plots = helper.create_dir(join(dir_preparation, "plots"))[0]
        
        fig, ax = plt.subplots()
        alpha = {}  # for plots of coeffs
        f = {}
        for direction, df in f_data.items():
            ax.plot(df["alpha"], df[f"f_{direction}"], **self.plt_settings[f"f_{direction}"])

            alpha[direction] = np.deg2rad(df["alpha"].to_numpy()) # np needed for later project_2D().
            f[direction] = df[f"f_{direction}"].to_numpy()
        handler = MosaicHandler(fig, ax)
        handler.update(x_labels=r"$\alpha$ (°)", y_labels="separation point (-)", legend=True, x_lims=[-50, 50], 
                       y_lims=[-0.1, 1.1])
        handler.save(join(dir_plots, "sep_points.pdf"))

        with open(join(dir_preparation, "zero_data.json"), "r") as file:
            zero_data = json.load(file)

        coeffs = self._reconstruct_force_coeffs(scheme_id, np.deg2rad(zero_data["alpha_0_n"]),
                                                np.rad2deg(zero_data["C_n_slope"]), alpha, f)
        coeffs = np.c_[coeffs["C_t"], coeffs["C_n"]]
        alpha = alpha["n"]
        rot_coeffs = self.rotate_2D(coeffs, alpha)
        df_polar = pd.read_csv(file_polar, delim_whitespace=True)

        fig, ax = plt.subplots()
        ax.plot(df_polar["alpha"], df_polar["C_d"], **self.plt_settings["C_d_specify"])
        ax.plot(df_polar["alpha"], df_polar["C_l"], **self.plt_settings["C_l_specify"])
        ax.plot(np.rad2deg(alpha), -rot_coeffs[:, 0]+df_polar["C_d"].min(), **self.plt_settings["C_d_rec"])
        ax.plot(np.rad2deg(alpha), rot_coeffs[:, 1], **self.plt_settings["C_l_rec"])
        handler = MosaicHandler(fig, ax)
        handler.update(x_labels=r"$\alpha$ (°)", y_labels="Force coefficient (-)", x_lims=[-180, 180], y_lims=[-3, 3],
                       legend=True)
        handler.save(join(dir_plots, "coeffs.pdf"))

    def _BL_openFAST_preparation(
            self,
            dir_preparation: str,
            file_polar: str,
            dir_paper_data: str,
            model_of_paper: str="openFAST"
    ):
        df_polar = pd.read_csv(file_polar, delim_whitespace=True)
        df_C_fs_paper = pd.read_csv(join(dir_paper_data, "Cl_fs.dat"))
        df_f_paper = pd.read_csv(join(dir_paper_data, "f_s.dat"))
        for file in listdir(dir_preparation):
            if "C_fs" in file:
                df_C_fs = pd.read_csv(join(dir_preparation, file))
            elif "sep_points" in file:
                df_f = pd.read_csv(join(dir_preparation, file))

        dir_plots = helper.create_dir(join(dir_preparation, "plots"))[0]

        fig, ax = plt.subplots()
        ax.plot(df_f["alpha"], df_f["f_l"], **self.plt_settings["section"])
        ax.plot(df_f_paper["alpha"], df_f_paper["f_l"], **self.plt_settings[model_of_paper])
        handler = MosaicHandler(fig, ax)
        handler.update(x_labels=r"$\alpha$ (°)", y_labels=r"$f$ (-)", x_lims=[-50, 50], y_lims=[-0.1, 1.1],
                       grid=True, legend=True)
        handler.save(join(dir_plots, "sep_point.pdf"))

        fig, ax = plt.subplots()
        alpha_paper = df_C_fs_paper["alpha"].sort_values()
        ax.plot(df_polar["alpha"], df_polar["C_l"], **self.plt_settings["C_l_specify"])
        ax.plot(df_C_fs["alpha"], df_C_fs["C_fs"], **self.plt_settings["section"])
        ax.plot(alpha_paper.values, df_C_fs_paper["C_lfs"].loc[alpha_paper.index], **self.plt_settings[model_of_paper])
        handler = MosaicHandler(fig, ax)
        handler.update(x_labels=r"$\alpha$ (°)", y_labels=r"$C_{l\text{,fs}}$ (-)", x_lims=[-50, 50], 
                       y_lims=[-1.5, 1.85], legend=True)
        handler.save(join(dir_plots, "C_fully_sep.pdf"))
        
    def plot_meas_comparison(
        self,
        root_unsteady_data: str,
        files_unsteady_data: dict[str],
        dir_results: str,
        period_res: int,
        ):
        map_measurement = {"AOA": "alpha", "CL": "C_l", "CD": "C_d", "CM": "C_m"}
        data_meas = {coeff: {} for coeff in ["C_d", "C_l", "C_m"]}
        for param, file in files_unsteady_data.items():
            param = param if len(param.split("_")) == 1 else param.split("_")[0]
            delim_whitespace = False if param != "HAWC2" else True
            df = pd.read_csv(join(root_unsteady_data, file), delim_whitespace=delim_whitespace)
            for coeff in df.columns[1:]:
                data_meas[map_measurement[coeff]][param] = [df["AOA"], df[coeff]]
                
        # df_aerohor = pd.read_csv(join(dir_results, "aerohor_res.dat"))
        df_section = pd.read_csv(join(dir_results, "f_aero.dat"))
        
        dir_save = helper.create_dir(join(dir_results, "plots"))[0]
        for coeff in ["C_d", "C_l", "C_m"]:
            fig, ax = plt.subplots()
            handler = MosaicHandler(fig, ax)
            for tool, data in data_meas[coeff].items():
                ax.plot(data[0], data[1], **self.plt_settings[coeff+f"_{tool}"])
            # if coef != "C_m":
                # ax.plot(df_aerohor["alpha_steady"][-period_res-1:], df_aerohor[coef][-period_res-1:], 
                #         **self.plt_settings["aerohor"])
            ax.plot(np.rad2deg(df_section["alpha_steady"][-period_res-1:]), df_section[coeff][-period_res-1:], 
                    **self.plt_settings["section"])
            handler.update(x_labels=r"$\alpha_{\text{steady}}$ (°)", y_labels=rf"${{{coeff[0]}}}_{{{coeff[2]}}}$ (-)",
                           legend=True)
            handler.save(join(dir_save, f"{coeff}.pdf"))

    def plot_over_polar(
        self,
        file_polar: str,
        dir_results: str,
        period_res: int,
        case_data: pd.Series,
        alpha_buffer: float=3,
        ):
        df_polar = pd.read_csv(file_polar, delim_whitespace=True)
        # df_aerohor = pd.read_csv(join(dir_results, "aerohor_res.dat"))
        df_section = pd.read_csv(join(dir_results, "f_aero.dat"))
        
        dir_save = helper.create_dir(join(dir_results, "plots"))[0]

        alpha_min = case_data["mean"]-case_data["amplitude"]-alpha_buffer
        alpha_max = case_data["mean"]+case_data["amplitude"]+alpha_buffer
        df_sliced = df_polar.loc[(df_polar["alpha"] >= alpha_min) & (df_polar["alpha"] <= alpha_max)]
        for coef in ["C_d", "C_l", "C_m"]:
            fig, ax = plt.subplots()
            handler = MosaicHandler(fig, ax)
            ax.plot(df_sliced["alpha"], df_sliced[coef], **self.plt_settings[coef])
            # if coef != "C_m":
            #     ax.plot(df_aerohor["alpha_steady"][-period_res-1:], df_aerohor[coef][-period_res-1:], 
            #             **plt_settings["aerohor"])
            ax.plot(np.rad2deg(df_section["alpha_steady"][-period_res-1:]), df_section[coef][-period_res-1:], 
                    **self.plt_settings["section"])
            handler.update(x_labels=r"$\alpha_{\text{steady}}$ (°)", y_labels=rf"${{{coef[0]}}}_{{{coef[2]}}}$ (-)",
                           legend=True)
            handler.save(join(dir_save, f"{coef}.pdf"))

    @staticmethod
    def plot_model_comparison(dir_results: str):
        dir_plots = helper.create_dir(join(dir_results, "plots", "model_comp"))[0]
        df_aerohor = pd.read_csv(join(dir_results, "aerohor_res.dat"))
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
    
    @staticmethod
    def _reconstruct_force_coeffs(
        scheme_id: int, 
        alpha_0: float, 
        C_slope: float, 
        alpha: dict[str, np.ndarray], 
        f: dict[str, np.ndarray]):
        coeffs = {f"C_{direction}": None for direction in ["n", "t"]}
        if scheme_id==1 or scheme_id==3:
            alpha_n = alpha["n"]
            alpha_t = alpha["t"]
            f_n = f["n"]
            f_t = f["t"]
        elif scheme_id==2:
            alpha_n = alpha["n"]
            f_n = f["n"]

        match scheme_id:
            case 1:
                coeffs["C_n"] = C_slope*np.sin(alpha_n-alpha_0)*((1+np.sqrt(np.abs(f_n))*np.sign(f_n))/2)**2
                coeffs["C_t"] = C_slope*np.sin(alpha_t-alpha_0)**2*np.sqrt(np.abs(f_t))*np.sign(f_t)
            case 2:
                coeffs["C_n"] = C_slope*(alpha_n-alpha_0)*((1+np.sqrt(np.abs(f_n))*np.sign(f_n))/2)**2
                coeffs["C_t"] = C_slope*alpha_n**2*np.sqrt(np.abs(f_n))*np.sign(f_n)
            case 3:
                coeffs["C_n"] = C_slope*(alpha_n-alpha_0)*((1+np.sqrt(np.abs(f_n))*np.sign(f_n))/2)**2
                coeffs["C_t"] = C_slope*(alpha_t-alpha_0)*np.tan(alpha_t)*np.sqrt(np.abs(f_t))*np.sign(f_t)
        return coeffs

    
def combined_forced(root_dir: str):
    df_combinations = pd.read_csv(join(root_dir, "combinations.dat"))
    amplitudes = np.sort(df_combinations["amplitude"].unique())
    aoas = np.sort(df_combinations["alpha"].unique())

    period_aero_work = np.zeros((amplitudes.size, aoas.size))
    period_struct_work = np.zeros((amplitudes.size, aoas.size))
    convergence_aero = np.zeros((amplitudes.size, aoas.size))
    convergence_struct = np.zeros((amplitudes.size, aoas.size))

    ampl_to_ind = {ampl: i for i, ampl in enumerate(amplitudes)}
    aoa_to_ind = {aoa: i for i, aoa in enumerate(aoas)}
    for i, row in df_combinations.iterrows():
        ampl = row["amplitude"]
        aoa = row["alpha"]
        df = pd.read_csv(join(root_dir, str(i), "period_work.dat"))
        aero_work = df[["aero_drag", "aero_lift", "aero_mom"]].sum(axis=1).to_numpy()
        struct_work = df[["damp_edge", "damp_flap", "damp_tors"]].sum(axis=1).to_numpy()

        period_aero_work[ampl_to_ind[ampl], aoa_to_ind[aoa]] = aero_work[-1]
        period_struct_work[ampl_to_ind[ampl], aoa_to_ind[aoa]] = struct_work[-1]

        convergence_aero[ampl_to_ind[ampl], aoa_to_ind[aoa]] = (aero_work[-1]-aero_work[-2])/aero_work[-1]
        convergence_struct[ampl_to_ind[ampl], aoa_to_ind[aoa]] = (struct_work[-1]-struct_work[-2])/struct_work[-1]
        
    dir_plots = helper.create_dir(join(root_dir, "plots"))[0]
    fig, ax = plt.subplots()
    ax = heatmap(period_aero_work, xticklabels=np.round(aoas, 3), ax=ax, yticklabels=np.round(amplitudes, 3),
                 cbar_kws={"label": r"period work"}, cmap="RdYlGn", annot=True, fmt=".3g")
    handler = MosaicHandler(fig, ax)
    handler.update(x_labels="angle of attack (°)", y_labels="oscillation amplitude (m)")
    handler.save(join(dir_plots, "period_aero_work.pdf"))

    fig, ax = plt.subplots()
    ax = heatmap(convergence_aero, xticklabels=np.round(aoas, 3), ax=ax, yticklabels=np.round(amplitudes, 3),
                 cbar_kws={"label": r"convergence"}, cmap="RdYlGn", annot=True, fmt=".3g")
    handler = MosaicHandler(fig, ax)
    handler.update(x_labels="angle of attack (°)", y_labels="oscillation amplitude (m)")
    handler.save(join(dir_plots, "convergence_aero.pdf"))

    fig, ax = plt.subplots()
    ax = heatmap(period_struct_work, xticklabels=np.round(aoas, 3), ax=ax, yticklabels=np.round(amplitudes, 3),
                 cbar_kws={"label": r"period work"}, cmap="RdYlGn", annot=True, fmt=".3g")
    handler = MosaicHandler(fig, ax)
    handler.update(x_labels="angle of attack (°)", y_labels="oscillation amplitude (m)")
    handler.save(join(dir_plots, "period_struct_work.pdf"))

    fig, ax = plt.subplots()
    ax = heatmap(convergence_struct, xticklabels=np.round(aoas, 3), ax=ax, yticklabels=np.round(amplitudes, 3),
                 cbar_kws={"label": r"convergence"}, cmap="RdYlGn", annot=True, fmt=".3g")
    handler = MosaicHandler(fig, ax)
    handler.update(x_labels="angle of attack (°)", y_labels="oscillation amplitude (m)")
    handler.save(join(dir_plots, "convergence_struct.pdf"))
    
    fig, ax = plt.subplots()
    ax = heatmap(period_aero_work+period_struct_work, xticklabels=np.round(aoas, 3), ax=ax, 
                 yticklabels=np.round(amplitudes, 3), cbar_kws={"label": "total period work"}, cmap="RdYlGn", 
                 annot=True, fmt=".3g")
    handler = MosaicHandler(fig, ax)
    handler.update(x_labels="angle of attack (°)", y_labels="oscillation amplitude (m)")
    handler.save(join(dir_plots, "stability.pdf"))