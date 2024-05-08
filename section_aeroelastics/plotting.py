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
helper = Helper()


class Plotter(DefaultStructure, DefaultPlot, PlotPreparation):
    """Utility class that plots results. Requires a directory to have a certain text files holding data about a 
    simulation. The name of these files is given in the parent class DefaultStructure.
    """
    
    def __init__(self, file_profile: str, dir_data: str, dir_plots: str) -> None:
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
        self.df_work = pd.read_csv(join(dir_data, self._dfl_filenames["work"]))
        self.df_e_kin = pd.read_csv(join(dir_data, self._dfl_filenames["e_kin"]))
        self.df_e_pot = pd.read_csv(join(dir_data, self._dfl_filenames["e_pot"]))
        self.time = self.df_general["time"].to_numpy()
        with open(join(dir_data, self._dfl_filenames["section_data"]), "r") as f:
            self.section_data = json.load(f)

        self.dir_plots = dir_plots
        helper.create_dir(self.dir_plots)
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
        aoas, df_aoas = self._get_aoas(self.df_f_aero)
        # prepare dictionary of dataframes to plot
        dfs = {"aoa": df_aoas, "aero": self.df_f_aero, "damp": self.df_f_structural, 
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
        """Plots the history of the airfoil movement and different energies/work.

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
        dfs = {"total": df_total, "work": self.df_work, "kinetic": self.df_e_kin, 
               "potential": self.df_e_pot}
        
        plot = {
            "total": ["e_total", "e_kin", "e_pot"],
            "work": ["aero_drag", "aero_lift", "aero_mom", "damp_edge", "damp_flap", "damp_tors"],
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
        fig, axs, handler = self._prepare_BL_plot(equal_y)
        
        # get final position of the profile. Displace it such that the qc is at (0, 0) at t=0.
        pos = self.df_general[["pos_x", "pos_y", "pos_tors"]].to_numpy()
        rot = self._rot.active_2D(pos[-1, 2])
        rot_profile = rot@(self.profile[:, :2]-np.asarray([self.section_data["chord"]/4, 0])).T
        rot_profile += np.asarray([[pos[-1, 0]], [pos[-1, 1]]])
        axs["profile"].plot(rot_profile[0, :], rot_profile[1, :], **self.plt_settings["profile"])
        axs["profile"].plot(pos[::trailing_every, 0], pos[::trailing_every, 1], **self.plt_settings["qc_trail"])
        
        # grab all angles of attack from the data
        aoas, df_aoas = self._get_aoas(self.df_f_aero)
        # prepare dictionary of dataframes to plot
        dfs = {ax_label: self.df_f_aero for ax_label in axs.keys() if ax_label not in ["profile", "aoa"]}
        dfs["aoa"] = df_aoas
        
        plot = {
            "aoa": aoas,
            "C_n": ["C_nc", "C_ni", "C_npot", "C_nsEq", "C_nf", "C_nv_instant", "C_nv"],
            "C_t": ["C_tpot", "C_tf"],
            "C_m": ["C_mqs", "C_mnc"]
        }        
        def param_name(param: str):
            return param
        axs = self._plot_to_mosaic(axs, plot, dfs, param_name)
        handler.update(legend=True)
        fig.savefig(join(self.dir_plots, "BL.pdf"))

    def _plot_to_mosaic(
            self,
            axes: dict[str, matplotlib.axes.Axes],
            plot: dict[str, list[str]],
            data: dict[str, pd.DataFrame],
            map_column_to_settings: Callable) -> dict[str, matplotlib.axes.Axes]:
        for ax, cols in plot.items():
            time = self.time if ax != "work" else self.time[:-1]
            for col in cols:
                try: 
                    self.plt_settings[map_column_to_settings(col)]
                except KeyError:
                    raise NotImplementedError(f"Default plot styles for '{col}' are missing.")
                axes[ax].plot(time, data[ax][col].to_numpy(), **self.plt_settings[map_column_to_settings(col)])
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
        self.df_work = pd.read_csv(join(dir_data, self._dfl_filenames["work"]))
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
        
        dfs = {"general": self.df_general, "f_aero": self.df_f_aero, "f_structural": self.df_f_structural}
        fig, plt_lines, plt_arrows, df_aoas = self._prepare_force_animation(dfs)

        data_lines = {linename: self.df_f_aero[linename] for linename in ["aero_edge", "aero_flap", "aero_mom"]} |\
                     {linename: df_aoas[linename] for linename in df_aoas.columns} |\
                     {linename: self.df_f_structural[linename] for linename in ["damp_edge", "damp_flap", "damp_tors"]+
                                                                            ["stiff_edge", "stiff_flap", "stiff_tors"]}
        data_lines = data_lines | {"qc_trail": qc_pos,
                                   "profile": profile, # data with dict[line: data_for_line]
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
        dfs = {"general": self.df_general, "e_tot": df_total, "work": self.df_work, "e_kin": self.df_e_kin, 
               "e_pot": self.df_e_pot}
        fig, plt_lines, plt_arrows = self._prepare_energy_animation(dfs)

        data_lines = {linename: df_total[linename] for linename in ["e_total", "e_kin", "e_pot"]} |\
                     {linename: self.df_work[linename] for linename in ["aero_drag", "aero_lift", "aero_mom", 
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
        df_work = pd.read_csv(join(case_dir, self._dfl_filenames["work"]))
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

        fig, axs = plt.subplot_mosaic([["total", "work"],
                                       ["e_kin", "e_pot"]], tight_layout=True)
        y_labels = {"total": "Energy (Nm)", "work": "Work (Nm)",
                   "e_kin": f"Kinetic energy (Nm)", "e_pot": "Potential energy (Nm)"}
        y_labels_err = {ax: f"{error_type} error {label[:label.find("(")-1]}" for ax, label in y_labels.items()}
        df_total = pd.concat([df_e_kin.sum(axis=1), df_e_pot.sum(axis=1)], keys=["e_kin", "e_pot"], axis=1)
        df_total["e_total"] = df_total.sum(axis=1)
        dfs = {"total": df_total, "work": df_work, "e_kin": df_e_kin, "pot": df_e_pot}
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
    

class BLValidationPlotter(DefaultPlot):
    def __init__(self) -> None:
        DefaultPlot.__init__(self)

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
            zero_lift_data = json.load(f)
            alpha_0_section = np.deg2rad(zero_lift_data["alpha_0"])
            dC_l_dalpha = np.rad2deg(zero_lift_data["dC_l_dalpha"])
            
        for f_type in ["f_t", "f_n"]:
            fig, ax = plt.subplots()
            ax.plot(df_aerohor[f"alpha_{f_type.split("_")[-1]}"], df_aerohor[f"{f_type}"], 
                    **self.plt_settings[f"{f_type}_aerohor"])
            ax.plot(df_section["alpha"], df_section[f"{f_type}"], 
                    **self.plt_settings[f"{f_type}_section"])
            handler = MosaicHandler(fig, ax)
            handler.update(x_labels=r"$\alpha$ (°)", y_labels=rf"${{{f_type[0]}}}_{{{f_type[2]}}}$ (-)", legend=True,
                           x_lims=[-40, 40], y_lims=[-1.2, 1.2])
            ax.grid()
            handler.save(join(dir_plots, f"{f_type}_model_comp.pdf"))
        
        rot = Rotations()
        coeffs_aerohor = np.c_[self._get_C_t(alpha_0_aerohor, df_aerohor["alpha_t"], df_aerohor["f_t"], 2*np.pi),
                               self._get_C_n(alpha_0_aerohor, df_aerohor["alpha_n"], df_aerohor["f_n"], 2*np.pi)]
        rot_coeffs_aerohor = rot.rotate_2D(coeffs_aerohor, np.deg2rad(df_aerohor["alpha_n"].to_numpy()))
        
        coesffs_section = np.c_[self._get_C_t(alpha_0_section, df_section["alpha"], df_section["f_t"], dC_l_dalpha),
                                self._get_C_n(alpha_0_section, df_section["alpha"], df_section["f_n"], dC_l_dalpha)]
        rot_coeffs_section = rot.rotate_2D(coesffs_section, np.deg2rad(df_section["alpha"].to_numpy()))
        rot_coeffs_section -= np.asarray([0.00663685061, 0])

        for i, coeff_type in enumerate(["C_d", "C_l"]):
            sign = 1 if coeff_type == "C_l" else -1
            fig, ax = plt.subplots()
            ax.plot(df_polar["alpha"], df_polar[coeff_type], **self.plt_settings[f"{coeff_type}_HAWC2"])
            ax.plot(df_aerohor["alpha_n"], sign*rot_coeffs_aerohor[:, i], 
                    **self.plt_settings[f"{coeff_type}_rec_aerohor"])
            ax.plot(df_section["alpha"], sign*rot_coeffs_section[:, i], 
                    **self.plt_settings[f"{coeff_type}_rec_section"])
            handler = MosaicHandler(fig, ax)
            handler.update(x_labels=r"$\alpha$ (°)", 
                           y_labels=rf"${{{coeff_type[0]}}}_{{{coeff_type[2]}}}$ (-)", legend=True,
                           x_lims=[-10, 10], y_lims=[-0.1, 0.2])
            handler.save(join(dir_plots, f"{coeff_type}_model_comp.pdf"))

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

    def plot_meas_comparison(
        self,
        file_unsteady_data: str,
        dir_results: str,
        period_res: int,
        ):
        df_meas = pd.read_csv(file_unsteady_data, delim_whitespace=True)
        df_aerohor = pd.read_csv(join(dir_results, "aerohor_res.dat"))
        df_section = pd.read_csv(join(dir_results, "f_aero.dat"))
        
        map_measurement = {"alpha": "AOA", "C_l": "CL", "C_d": "CD", "C_m": "CM"}
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
        if "unsteady_10_20_k1.dat" in file_unsteady_data:
            df_cl = pd.read_csv("data/FFA_WA3_221/unsteady/openFAST_1.dat")
            df_cd = pd.read_csv("data/FFA_WA3_221/unsteady/openFAST_2.dat")
            dfs = {"C_l": df_cl, "C_d": df_cd}
        for coef in ["C_d", "C_l", "C_m"]:
            fig, ax = plt.subplots()
            handler = MosaicHandler(fig, ax)
            ax.plot(df_meas[map_measurement["alpha"]], df_meas[map_measurement[coef]], 
                    **self.plt_settings[coef+"_HAWC2"])
            if "unsteady_10_20_k1.dat" in file_unsteady_data:
                if coef != "C_m":
                    ax.plot(dfs[coef]["AOA"], dfs[coef][map_measurement[coef]], "ko", ms=2,
                             label="openFAST")
            if coef != "C_m":
                ax.plot(df_aerohor["alpha_steady"][-period_res-1:], df_aerohor[coef][-period_res-1:], 
                        **plt_settings["aerohor"])
            ax.plot(np.rad2deg(df_section["alpha_steady"][-period_res-1:]), df_section[coef][-period_res-1:], 
                    **plt_settings["section"])
            handler.update(x_labels=r"$\alpha_{\text{steady}}$ (°)", y_labels=rf"${{{coef[0]}}}_{{{coef[2]}}}$ (-)",
                           legend=True)
            handler.save(join(dir_save, f"{coef}.pdf"))

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
        
        dir_save = helper.create_dir(join(dir_results, "plots", "polar_comp"))[0]
        line_width = 1
        plt_settings = {
            "aerohor": {
                "color": "forestgreen", "lw": line_width, "label": "aerohor"
            },
            "section": {
                "color": "orangered", "lw": line_width, "label": "section"
            }
        }

        alpha_min = case_data["mean"]-case_data["amplitude"]-alpha_buffer
        alpha_max = case_data["mean"]+case_data["amplitude"]+alpha_buffer
        ids_alpha = np.logical_and(df_polar["alpha"].to_numpy()>=alpha_min, df_polar["alpha"].to_numpy()<=alpha_max)
        df_sliced = df_polar.loc[(df_polar["alpha"] >= alpha_min) & (df_polar["alpha"] <= alpha_max)]

        for coef in ["C_d", "C_l", "C_m"]:
            fig, ax = plt.subplots()
            handler = MosaicHandler(fig, ax)
            ax.plot(df_sliced["alpha"], df_sliced[coef], 
                    **self.plt_settings[coef])
            # if coef != "C_m":
            #     ax.plot(df_aerohor["alpha_steady"][-period_res-1:], df_aerohor[coef][-period_res-1:], 
            #             **plt_settings["aerohor"])
            ax.plot(np.rad2deg(df_section["alpha_steady"][-period_res-1:]), df_section[coef][-period_res-1:], 
                    **plt_settings["section"])
            handler.update(x_labels=r"$\alpha_{\text{steady}}$ (°)", y_labels=rf"${{{coef[0]}}}_{{{coef[2]}}}$ (-)",
                           legend=True)
            handler.save(join(dir_save, f"{coef}.pdf"))


    @staticmethod
    def _get_C_t(alpha_0: float, alpha: np.ndarray, f_t: np.ndarray, C_l_slope: float):
        return C_l_slope*np.sin(np.deg2rad(alpha)-alpha_0)**2*np.sqrt(np.abs(f_t))*np.sign(f_t)

    @staticmethod
    def _get_C_n(alpha_0: float, alpha: np.ndarray, f_n: np.ndarray, C_l_slope: float):
        return C_l_slope*np.sin(np.deg2rad(alpha)-alpha_0)*((1+np.sqrt(np.abs(f_n))*np.sign(f_n))/2)**2
