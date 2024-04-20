from utils_plot import Shapes, PlotPreparation, AnimationPreparation
from calculations import Rotations
from defaults import DefaultsPlots, DefaultStructure
import pandas as pd
import numpy as np
import json
import matplotlib.animation as animation
import matplotlib
from os.path import join
from helper_functions import Helper
from typing import Callable
helper = Helper()


class Plotter(DefaultStructure, DefaultsPlots, PlotPreparation):
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
        DefaultsPlots.__init__(self)
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
        rot = self._rot.passive_2D(pos[-1, 2])
        rot_profile = rot@(self.profile[:, :2]-np.asarray([self.section_data["chord"]/4, 0])).T
        rot_profile += np.asarray([[pos[-1, 0]], [pos[-1, 1]]])
        axs["profile"].plot(rot_profile[0, :], rot_profile[1, :], **self.plot_settings["profile"])
        axs["profile"].plot(pos[::trailing_every, 0], pos[::trailing_every, 1], **self.plot_settings["qc_trail"])
        
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
        rot = self._rot.passive_2D(pos[-1, 2])
        rot_profile = rot@(self.profile[:, :2]-np.asarray([self.section_data["chord"]/4, 0])).T
        rot_profile += np.asarray([[pos[-1, 0]], [pos[-1, 1]]])
        axs["profile"].plot(rot_profile[0, :], rot_profile[1, :], **self.plot_settings["profile"])
        axs["profile"].plot(pos[::trailing_every, 0], pos[::trailing_every, 1], **self.plot_settings["qc_trail"])
        
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
                    self.plot_settings[map_column_to_settings(col)]
                except KeyError:
                    raise NotImplementedError(f"Default plot styles for '{col}' are missing.")
                axes[ax].plot(time, data[ax][col].to_numpy(), **self.plot_settings[map_column_to_settings(col)])
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
        DefaultsPlots.__init__(self)
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
            rot_mat = self._rot.acitve_2D(angle)
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
            rot_mat = self._rot.acitve_2D(angle)
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
        




        