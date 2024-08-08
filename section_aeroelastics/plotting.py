from utils_plot import Shapes, PlotPreparation, AnimationPreparation, PlotHandler, get_colourbar
from calculations import Rotations
from defaults import DefaultPlot, DefaultStructure
import pandas as pd
import numpy as np
import json
import matplotlib.animation as animation
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
import matplotlib
from os.path import join
from os import listdir
from scipy.signal import find_peaks
from scipy.interpolate import interp1d
from helper_functions import Helper
from typing import Callable
from ast import literal_eval
from seaborn import heatmap
from itertools import product
helper = Helper()


class Plotter(DefaultStructure, DefaultPlot, PlotPreparation):
    """Utility class that plots results. Requires a directory to have a certain text files holding data about a 
    simulation. The name of these files is given in the parent class DefaultStructure.
    """
    
    def __init__(self, file_profile: str, dir_data: str, dir_plots: str, structure_coordinate_system: str,
                 dt_res: float=500) -> None:
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
        self.profile *= self.section_data["chord"]

        self.dir_plots = dir_plots
        helper.create_dir(self.dir_plots)
        self.dt_res = dt_res
        self._rot = Rotations()
        if structure_coordinate_system not in ["ef", "xy"]:
            raise ValueError(f"Unsupported 'structure_coordinate_system'={structure_coordinate_system}. Supported are "
                             " 'ef' and 'xy'.")
        self._cs_struc = structure_coordinate_system

    def force(self, equal_y: tuple[str]=None, trailing_every: int=40, time_frame: tuple[float, float]=None):
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
        use_ids = None
        if time_frame is not None:
            use_ids = np.logical_and(self.time >= time_frame[0], self.time <= time_frame[1])
        pos = self.df_general[["pos_x", "pos_y", "pos_tors"]].to_numpy()
        pos = pos if use_ids is None else pos[use_ids, :]
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
        
        plot = {"aoa": aoas}
        # if self._cs_struc == "ef":
        if True:
            plot["aero"] = ["aero_edge", "aero_flap", "aero_mom"]
            plot["damp"] = ["damp_edge", "damp_flap", "damp_tors"]
            plot["stiff"] = ["stiff_edge", "stiff_flap", "stiff_tors"]
        elif self._cs_struc == "xy":
            plot["aero"] = ["aero_x", "aero_y", "aero_mom"]
            plot["damp"] = ["damp_x", "damp_y", "damp_tors"]
            plot["stiff"] = ["stiff_x", "stiff_y", "stiff_tors"]

        def param_name(param: str):
            return param
        
        axs = self._plot_to_mosaic(axs, plot, dfs, param_name, time_frame=time_frame)
        handler.update(legend=True)
        fig.savefig(join(self.dir_plots, "forces.pdf"))
                    
    def force_fill(self, equal_y: tuple[str]=None, trailing_every: int=40, alpha: int=0.2, peak_distance: int=400,
                   time_frame: tuple[float, float]=None):
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
        
        plot = {"aoa": aoas}
        # if self._cs_struc == "ef":
        if True:
            plot["aero"] = ["aero_edge", "aero_flap", "aero_mom"]
            plot["damp"] = ["damp_edge", "damp_flap", "damp_tors"]
            plot["stiff"] = ["stiff_edge", "stiff_flap", "stiff_tors"]
        elif self._cs_struc == "xy":
            plot["aero"] = ["aero_x", "aero_y", "aero_mom"]
            plot["damp"] = ["damp_x", "damp_y", "damp_tors"]
            plot["stiff"] = ["stiff_x", "stiff_y", "stiff_tors"]

        def param_name(param: str):
            return param
        
        n_data_points = pos.shape[0]
        skip_points = max(int(n_data_points/self.dt_res), 1)
        axs = self._plot_and_fill_to_mosaic(axs, plot, dfs, param_name, alpha=alpha, peak_distance=peak_distance,
                                            time_frame=time_frame)
        handler.update(legend=True)
        fig.savefig(join(self.dir_plots, "forces_fill.pdf"))

    def energy(self, equal_y: tuple[str]=None, trailing_every: int=40, time_frame: tuple[float, float]=None):
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
        use_ids = None
        if time_frame is not None:
            use_ids = np.logical_and(self.time >= time_frame[0], self.time <= time_frame[1])
        pos = self.df_general[["pos_x", "pos_y", "pos_tors"]].to_numpy()
        pos = pos if use_ids is None else pos[use_ids, :]
        rot = self._rot.active_2D(pos[-1, 2])
        rot_profile = rot@(self.profile[:, :2]-np.asarray([self.section_data["chord"]/4, 0])).T
        rot_profile += np.asarray([[pos[-1, 0]], [pos[-1, 1]]])
        axs["profile"].plot(rot_profile[0, :], rot_profile[1, :], **self.plt_settings["profile"])
        axs["profile"].plot(pos[::trailing_every, 0], pos[::trailing_every, 1], **self.plt_settings["qc_trail"])
        
        df_total = pd.concat([self.df_e_kin["total"], self.df_e_pot["total"]], keys=["e_kin", "e_pot"], axis=1)
        df_total["e_total"] = df_total.sum(axis=1)
        dfs = {"total": df_total, "power": self.df_power, "kinetic": self.df_e_kin, 
               "potential": self.df_e_pot}
        
        plot = {
            "total": ["e_total", "e_kin", "e_pot"],
            "power": ["aero_drag", "aero_lift", "aero_mom"]
        }
        # if self._cs_struc == "ef":
        #     plot["power"] += ["damp_edge", "damp_flap", "damp_tors"]
        #     plot["kinetic"] = ["edge", "flap", "tors"]
        #     plot["potential"] = ["edge", "flap", "tors"]
        # elif self._cs_struc == "xy":
        #     plot["power"] += ["damp_x", "damp_y", "damp_tors"]
        #     plot["kinetic"] = ["x", "y", "tors"]
        #     plot["potential"] = ["x", "y", "tors"]

        plot["power"] += ["damp_edge", "damp_flap", "damp_tors"]
        plot["kinetic"] = ["edge", "flap", "tors"]
        
        if self._cs_struc == "ef":
            plot["potential"] = ["edge", "flap", "tors"]
        elif self._cs_struc == "xy":
            plot["potential"] = ["x", "y", "tors"]

        def param_name(param: str):
            return param 
        
        n_data_points = pos.shape[0]
        axs = self._plot_to_mosaic(axs, plot, dfs, param_name, time_frame=time_frame)
        handler.update(legend=True)
        fig.savefig(join(self.dir_plots, "energy.pdf"))
    
    def energy_fill(self, equal_y: tuple[str]=None, trailing_every: int=40, alpha: int=0.2, peak_distance: int=400,
                    time_frame: tuple[float, float]=None):
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
            "power": ["aero_drag", "aero_lift", "aero_mom"]
        }
        # if self._cs_struc == "ef":
        #     plot["power"] += ["damp_edge", "damp_flap", "damp_tors"]
        #     plot["kinetic"] = ["edge", "flap", "tors"]
        #     plot["potential"] = ["edge", "flap", "tors"]
        # elif self._cs_struc == "xy":
        #     plot["power"] += ["damp_x", "damp_y", "damp_tors"]
        #     plot["kinetic"] = ["x", "y", "tors"]
        #     plot["potential"] = ["x", "y", "tors"]

        plot["power"] += ["damp_edge", "damp_flap", "damp_tors"]
        plot["kinetic"] = ["edge", "flap", "tors"]
        
        if self._cs_struc == "ef":
            plot["potential"] = ["edge", "flap", "tors"]
        elif self._cs_struc == "xy":
            plot["potential"] = ["x", "y", "tors"]

        def param_name(param: str):
            return param 
        
        n_data_points = pos.shape[0]
        skip_points = max(int(n_data_points/self.dt_res), 1)
        axs = self._plot_and_fill_to_mosaic(axs, plot, dfs, param_name, alpha=alpha, peak_distance=peak_distance,
                                            time_frame=time_frame)
        handler.update(legend=True)
        fig.savefig(join(self.dir_plots, "energy_fill.pdf"))

    def Beddoes_Leishman(self, equal_y: tuple[str]=None, trailing_every: int=40, time_frame: tuple[float, float]=None):
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
        use_ids = None
        if time_frame is not None:
            use_ids = np.logical_and(self.time >= time_frame[0], self.time <= time_frame[1])
        pos = self.df_general[["pos_x", "pos_y", "pos_tors"]].to_numpy()
        pos = pos if use_ids is None else pos[use_ids, :]
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
        
        n_data_points = pos.shape[0]
        axs = self._plot_to_mosaic(axs, plot, dfs, param_name, time_frame=time_frame)
        handler.update(legend=True)
        fig.savefig(join(self.dir_plots, "BL.pdf"))

    def Beddoes_Leishman_fill(self, equal_y: tuple[str]=None, trailing_every: int=40, alpha: int=0.2, 
                              peak_distance: int=400, time_frame: tuple[float, float]=None):
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
        
        n_data_points = pos.shape[0]
        skip_points = max(int(n_data_points/self.dt_res), 1)
        axs = self._plot_and_fill_to_mosaic(axs, plot, dfs, param_name, alpha=alpha, peak_distance=peak_distance,
                                            time_frame=time_frame)
        handler.update(legend=True)
        fig.savefig(join(self.dir_plots, "BL_fill.pdf"))

    def damping(self, 
                alpha_thresholds: tuple[float, list], 
                time_frame: tuple[float, float]=None,
                polar: tuple[str, list[float]|int]=None,
                colours: list[str]=["chocolate", "grey", "orangered", "forestgreen", "royalblue", "crimson"]):
        use_ids = np.arange(self.time.size)
        time = self.time
        if time_frame is not None:
            time_frame = (time_frame[0], min(time_frame[1], time[-1]))
            use_ids = np.logical_and(self.time >= time_frame[0], self.time <= time_frame[1])
            time = time[use_ids]
        else:
            time_frame = [0, time[-1]]

        pos_x = self.df_general["pos_x"].iloc[use_ids].to_numpy().flatten()
        alpha_th = self.df_f_aero[alpha_thresholds[0]].to_numpy()

        ppeaks = find_peaks(pos_x)[0]
        npeaks = find_peaks(-pos_x)[0]
        max_aoa = np.rad2deg(alpha_th[ppeaks[-2]:ppeaks[-1]]).max()

        thresholds = alpha_thresholds[1]  # deg
        thresholds = np.deg2rad(thresholds[::-1])
        alpha_above_threshold = {alpha: [] for alpha in thresholds}
        above_set = {alpha: False for alpha in thresholds}
        start_above = {alpha: 0 for alpha in thresholds}
        end_above = {alpha: 0 for alpha in thresholds}
        for peak_id, (osci_begin, osci_end) in enumerate(zip(ppeaks[:-1], ppeaks[1:])):
            for i_threshold, alpha_threshold in enumerate(thresholds):
                if np.any(alpha_th[osci_begin:osci_end] > alpha_threshold):
                    if not above_set[alpha_threshold]:
                        start_above[alpha_threshold] = peak_id
                        above_set[alpha_threshold] = True
                        for at in thresholds[i_threshold+1:]:
                            end_above[at] = peak_id
                        continue
                else:
                    end_above[alpha_threshold] = peak_id
            for alpha_threshold in thresholds:
                if start_above[alpha_threshold] < end_above[alpha_threshold] and above_set[alpha_threshold]:
                    alpha_above_threshold[alpha_threshold].append((start_above[alpha_threshold], 
                                                                   end_above[alpha_threshold]))
                    above_set[alpha_threshold] = False

        for alpha_threshold in thresholds:
            if start_above[alpha_threshold] > end_above[alpha_threshold]:
                alpha_above_threshold[alpha_threshold].append((start_above[alpha_threshold], peak_id))
                break
            
        damping_ratio = []
        amplitude = [(pos_x[ppeaks[0]]-pos_x[npeaks[0]])/2]
        damping_ratio = []
        for i, (idx_ppeak, idx_npeak) in enumerate(zip(ppeaks[1:], npeaks[1:]), 1):
            # mean = pos_x[ppeaks[max(0, i-4)]:ppeaks[min(ppeaks.size-1, i+4)]].mean()
            # ampl = pos_x[idx_ppeak]-mean
            # amplitude.append(ampl)
            # log_dec = np.log((pos_x[ppeaks[i-1]]-mean)/(pos_x[idx_ppeak]-mean))
            amplitude.append((pos_x[idx_ppeak]-pos_x[idx_npeak])/2)
            log_dec = np.log(amplitude[i-1]/amplitude[i])
            damping_ratio.append(log_dec/(2*np.pi))
        amplitude = np.asarray(amplitude)  # for consistency in the following code
        
        if polar is not None:
            df_polar = pd.read_csv(polar[0], delim_whitespace=True)
            if "alpha" not in df_polar:
                df_polar = pd.read_csv(polar[0])
            C_lus = self.df_f_aero["C_lus"]
            C_dus = self.df_f_aero["C_dus"]
            alpha_eff = self.df_f_aero["alpha_eff"]
            if isinstance(polar[1], list):
                polar_ids = [(idx, idx+1) for idx in ppeaks[polar[1]]]
            else:
                ampl_min, ampl_max = min(amplitude), max(amplitude)
                if polar[1] >= len(colours):
                    raise ValueError(f"Missing {polar[1]-len(colours)} 'colours'. Add more to the method call.")
                ampls = np.exp(np.linspace(np.log(ampl_min), np.log(ampl_max), polar[1]))
                polar_ids = []
                for ampl in ampls:
                    idx_greater = (amplitude>=ampl).argmax()
                    idx_smaller = (amplitude<=ampl).argmax()
                    idx = max(idx_greater, idx_smaller)
                    polar_ids.append((idx, idx+1))
                    
        fig, ax = plt.subplots()
        ax.semilogx(amplitude[:-1], damping_ratio, linewidth=2)
        if polar is not None:
            for colour_idx, polar_idx in enumerate(polar_ids):
                ax.semilogx(amplitude[polar_idx[0]], damping_ratio[polar_idx[0]], marker="o", color=colours[colour_idx])
        
        colours_AoA_range = ["darkorange", "orangered"]
        colour_mapping = {alpha_threshold: colours_AoA_range[i] for i, alpha_threshold in enumerate(thresholds[::-1])}
        for alpha_threshold in thresholds:
            for begin_above, end_above in alpha_above_threshold[alpha_threshold]:
                ax.axvspan(amplitude[begin_above], amplitude[end_above], color=colour_mapping[alpha_threshold], 
                           alpha=0.3)
        ax.grid(which="both")
                
        handler = PlotHandler(fig, ax)
        handler.update(x_labels="oscillation amplitude (m)", 
                       y_labels=r"normal ($\approx$edgewise) damping ratio (-)")
        thresholds = np.rad2deg(thresholds[::-1])
        plt.title(rf"orange $\left(\alpha_{{eff}}>{np.round(thresholds[0], 1)}^{{\circ}}\right)$, "
                rf"red $\left(\alpha_{{eff}}>{np.round(thresholds[1], 1)}^{{\circ}}\right)$"
                "\n"
                rf"last period: $\alpha_{{eff,\text{{max}}}}\approx{np.round(max_aoa,1)}^{{\circ}}$")
        handler.save(join(self.dir_plots, "damping.pdf"))

        if polar is not None:
            for coeff, coeff_name in zip([C_lus, C_dus], ["C_l", "C_d"]):
                fig, ax = plt.subplots()
                for colour_idx, (peak_idx_begin, peak_idx_end) in enumerate(polar_ids):
                    begin, end = ppeaks[peak_idx_begin], ppeaks[peak_idx_end]
                    ax.plot(np.rad2deg(alpha_eff.iloc[begin:end+1]), coeff.iloc[begin:end+1], color=colours[colour_idx])

                alpha_lim = ax.get_xlim()
                alpha_polar = df_polar["alpha"].to_numpy().flatten()
                alpha_ids = np.logical_and(alpha_polar>=alpha_lim[0], alpha_polar<=alpha_lim[1])
                ax.plot(alpha_polar[alpha_ids], df_polar[coeff_name].iloc[alpha_ids], "--k")
                handler = PlotHandler(fig, ax)
                handler.update(x_labels=r"$\alpha$ (°)", y_labels=rf"${coeff_name}$ (-)")
                handler.save(join(self.dir_plots, f"{coeff_name}_loops.pdf"))


    def couple_timeseries(self, cmap="viridis_r", linewidth: float=1.2, skip_points: int=1):
        df_polar = pd.read_csv("data/FFA_WA3_221/polars_new.dat", delim_whitespace=True)
        a = (self.df_f_aero, "f_aero", "alpha_steady", r"$\alpha$ (deg)")
        b = (self.df_f_aero, "f_aero", "alpha_eff", r"$\alpha_{\text{eff}}$ (deg)")
        c = (self.df_f_aero, "f_aero", "C_lus", r"$C_{l,\text{us}}$ (-)")
        d = (self.df_f_aero, "f_aero", "C_dus", r"$C_{d,\text{us}}$ (-)")
        e = (self.df_f_aero, "f_aero", "f_steady", r"$f$ (-)")
        f = (self.df_f_aero, "f_aero", "x4", r"$x_4$ (-)")
        from_f_aero = [(self.df_f_aero, "f_aero", "aero_edge", r"$f_{\text{edge}}^{\text{aero}}$ (N)"), 
                       (self.df_f_aero, "f_aero", "aero_flap", r"$f_{\text{flap}}^{\text{aero}}$ (N)"), 
                       (self.df_f_aero, "f_aero", "aero_mom", r"$f_{\text{moment}}^{\text{aero}}$ (Nm)"),
                       ]
        from_power = [(self.df_power, "power", "aero_drag", r"$P_{\text{drag}}^{\text{aero}}$ (Nm/s)"),
                      (self.df_power, "power", "aero_lift", r"$P_{\text{lift}}^{\text{aero}}$ (Nm/s)"),
                      (self.df_power, "power", "aero_mom", r"$P_{\text{moment}}^{\text{aero}}$ (Nm/s)")]
        from_energy = [(self.df_e_kin, "e_kin", "total", r"$E_{\text{total}}^{\text{kin}}$ (Nm)"),
                       (self.df_e_pot, "e_pot", "total", r"$E_{\text{total}}^{\text{pot}}$ (Nm)")]
        from_general = [(self.df_general, "general", "pos_edge", r"$x_{\text{edge}}$ (m)"),
                        (self.df_general, "general", "vel_edge_xy", r"$u_{\text{edge}}$ (m/s)"),
                        (self.df_general, "general", "pos_flap", r"$x_{\text{flap}}$ (m)"),
                        (self.df_general, "general", "vel_flap_xy", r"$u_{\text{flap}}$ (m/s)"),
                        (self.df_general, "general", "pos_tors", r"$x_{\text{torsion}}$ (deg)")]
        couples = [(a, b), (c, b), (d, b), (f, b), (b, from_general[0])]
        # couples = [(from_general[1], from_general[0]), (from_general[2], from_general[0]), (from_general[2], from_general[1])]
        # couples = [(b, from_general[1])]
        # couples = [p for p in product(from_f_aero, from_general)] + [p for p in product(from_power, from_general)]
        # couples = [(from_general[1], from_general[0]), (from_general[2], from_general[0]), 
        #             (from_general[2], from_general[1])]
        # couples += [(from_energy[0], from_energy[1])]
        dir_plots_coupled_root = helper.create_dir(join(self.dir_plots, "coupled"))[0]
        time = self.df_general["time"].to_numpy()
        # because this column is used in the loop
        self.df_general["pos_tors"]= np.rad2deg(self.df_general["pos_tors"])
        for (df_specific, ds, col_specific, label_specific), (df_general, dg, col_general, label_general) in couples:
            dir_plots = helper.create_dir(join(dir_plots_coupled_root, f"{dg}_{ds}"))[0]
            save_to = join(dir_plots, f"{col_specific}_{col_general}.pdf")
            val_general = df_general[col_general].to_numpy()
            # because power values have one values fewer
            val_general = val_general if "P_" not in label_specific else val_general[:-1]
            val_specifc = df_specific[col_specific].to_numpy()
            
            val_general = np.rad2deg(val_general)
            # val_specifc = np.rad2deg(val_specifc)
            add = None
            if col_specific == "C_lus" and "alpha" in col_general:
                add = (df_polar["alpha"], df_polar["C_l"])
            elif col_specific == "C_dus" and "alpha" in col_general:
                add = (df_polar["alpha"], df_polar["C_d"])
            self.coupled_timeseries(time, val_general, val_specifc,
                                    label_general, label_specific, cmap=cmap, linewidth=linewidth, save_to=save_to, 
                                    skip_points=skip_points, add=add)
        self.df_general["pos_tors"]= np.deg2rad(self.df_general["pos_tors"])  # change the column back to rad
    
    @staticmethod
    def coupled_timeseries(time: np.ndarray, val1: np.ndarray, val2: np.ndarray,
                           x_label: str, y_label: str, grid: bool=False, cmap="viridis_r", linewidth: float=1.2,
                           save_to: str=None, skip_points: int=1, add: tuple[list, list]=None):
        fig, ax = plt.subplots()

        points = np.asarray([val1[::skip_points], val2[::skip_points]]).T.reshape(-1, 1, 2)
        segments = np.concatenate([points[:-1], points[1:]], axis=1)

        norm = plt.Normalize(0, time[-1])
        lc = LineCollection(segments, cmap=cmap, norm=norm, linewidth=linewidth)
        lc.set_array(time[::skip_points])
        line = ax.add_collection(lc)
        ax.autoscale()

        cbar = fig.colorbar(line, ax=ax, label="time (s)")
        ticks = cbar.get_ticks().tolist()
        if time[-1] not in ticks:
            ticks.append(time[-1])
            ticks = sorted(ticks)
        cbar.set_ticks(ticks)

        if add is not None:
            min_x = val1[::skip_points].min()
            max_x = val1[::skip_points].max()
            x = add[0] if isinstance(add[0], np.ndarray) else np.asarray(add[0])
            y = add[1] if isinstance(add[1], np.ndarray) else np.asarray(add[1])
            used_ids = np.logical_and(x>=min_x, x<=max_x)
            ax.plot(x[used_ids], y[used_ids], "k", label="polar")
        
        handler = PlotHandler(fig, ax)
        handler.update(x_labels=x_label, y_labels=y_label, grid=grid)
        if save_to is not None:
            handler.save(save_to)
        return fig, ax, handler

    def _plot_to_mosaic(
            self,
            axes: dict[str, matplotlib.axes.Axes],
            plot: dict[str, list[str]],
            data: dict[str, pd.DataFrame|dict],
            map_column_to_settings: Callable,
            apply: dict[str, Callable]={"aoa": np.rad2deg},
            time_frame: tuple[float, float]=None) -> dict[str, matplotlib.axes.Axes]:
        if time_frame is not None:
            time_ids = np.logical_and(self.time >= time_frame[0], self.time <= time_frame[1])
            base_time = self.time[time_ids]
            n_data_points = time_ids.sum()
        else:
            base_time = self.time
            n_data_points = self.time.size
            
        skip_points = max(int(n_data_points/self.dt_res), 1)
        apply_to_axs = apply.keys()
        for ax, cols in plot.items():
            for col in cols:
                try: 
                    self.plt_settings[map_column_to_settings(col)]
                except KeyError:
                    raise NotImplementedError(f"Default plot styles for '{col}' are missing.")
                vals = data[ax][col] if ax not in apply_to_axs else apply[ax](data[ax][col])
                time = base_time
                time = base_time if ax != "power" else base_time[:-1]
                use_ids = time_ids if ax != "power" else time_ids[:-1]
                vals = vals if time_frame is None else vals[use_ids]                
                axes[ax].plot(time[::skip_points], vals[::skip_points],
                              **self.plt_settings[map_column_to_settings(col)])
        return axes
    
    def _plot_and_fill_to_mosaic(
            self,
            axes: dict[str, matplotlib.axes.Axes],
            plot: dict[str, list[str]],
            data: dict[str, pd.DataFrame|dict],
            map_column_to_settings: Callable,
            apply: dict[str, Callable]={"aoa": np.rad2deg},
            alpha: float=0.2,
            peak_distance: int=400,
            time_frame: tuple[float, float]=None) -> dict[str, matplotlib.axes.Axes]:
        if time_frame is not None:
            time_ids = np.logical_and(self.time >= time_frame[0], self.time <= time_frame[1])
            base_time = self.time[time_ids]
        else:
            time_ids = np.arange(self.time.size)
            base_time = self.time
        apply_to_axs = apply.keys()
        for ax, cols in plot.items():
            for col in cols:
                try: 
                    self.plt_settings[map_column_to_settings(col)]
                except KeyError:
                    raise NotImplementedError(f"Default plot styles for '{col}' are missing.")
                vals = data[ax][col] if ax not in apply_to_axs else apply[ax](data[ax][col])
                time = base_time if ax != "power" else base_time[:-1]
                use_ids = time_ids if ax != "power" else time_ids[:-1]
                vals = vals if time_frame is None else vals[use_ids]
                

                ids_max = find_peaks(vals, distance=peak_distance)[0]
                ids_min = find_peaks(-vals, distance=peak_distance)[0]

                vals_max = vals[ids_max]
                vals_min = vals[ids_min]
                t_max = time[ids_max]
                t_min = time[ids_min]

                fill_time = np.unique(np.sort(np.r_[t_max, t_min]))
                vals_max_interp = interp1d(t_max, vals_max, fill_value="extrapolate")
                vals_min_interp = interp1d(t_min, vals_min, fill_value="extrapolate")
                plt_settings_no_label = self.plt_settings[map_column_to_settings(col)].copy()
                plt_settings_no_label.pop("label")
                axes[ax].plot(t_max, vals_max, **self.plt_settings[map_column_to_settings(col)])
                axes[ax].plot(t_min, vals_min, **plt_settings_no_label)
                axes[ax].fill_between(fill_time, vals_max_interp(fill_time), vals_min_interp(fill_time), alpha=alpha, 
                                      facecolor=plt_settings_no_label["color"])
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
        
        self.df_f_aero = pd.read_csv(join(dir_data, self._dfl_filenames["f_aero"]))
        self.df_f_structural = pd.read_csv(join(dir_data, self._dfl_filenames["f_structural"]))
        self.df_general = pd.read_csv(join(dir_data, self._dfl_filenames["general"]))
        self.df_power = pd.read_csv(join(dir_data, self._dfl_filenames["power"]))
        self.df_e_kin = pd.read_csv(join(dir_data, self._dfl_filenames["e_kin"]))
        self.df_e_pot = pd.read_csv(join(dir_data, self._dfl_filenames["e_pot"]))
        self.time = self.df_general["time"].to_numpy()
        with open(join(dir_data, self._dfl_filenames["section_data"]), "r") as f:
            self.section_data = json.load(f)
        
        self._chord = self.section_data["chord"]
        AnimationPreparation.__init__(self, self.section_data["chord"])
        self.profile = pd.read_csv(file_profile, delim_whitespace=True).to_numpy()*self._chord
        self.dir_plots = dir_plots
        helper.create_dir(self.dir_plots)
        self._rot = Rotations()
    
    def force(
            self,
            angle_lift: str,
            arrow_scale_forces: float=None,
            arrow_scale_moment: float=None,
            plot_qc_trailing_every: int=2,
            keep_qc_trailing: int=40,
            time_frame: tuple[float, float]=None,
            dt_per_frame: float=None):
        use_ids = np.arange(self.time.size)
        time = self.time
        if time_frame is not None:
            time_frame = (time_frame[0], min(time_frame[1], time[-1]))
            use_ids = np.logical_and(self.time >= time_frame[0], self.time <= time_frame[1])
            time = time[use_ids]
        else:
            time_frame = [0, time[-1]]

        qc_pos = self.df_general[["pos_x", "pos_y"]].to_numpy()[use_ids, :]
        profile = np.zeros((time.size, *self.profile.shape))
        norm_moments = self.df_f_aero["aero_mom"].iloc[use_ids]/np.abs(self.df_f_aero["aero_mom"].iloc[use_ids]).max()
        mom_arrow_res = 40
        mom_arrow = np.zeros((time.size, mom_arrow_res+3, 2))
        prof = self.profile-np.c_[0.25*self._chord, 0]
        for i, (angle, moment) in enumerate(zip(self.df_general["pos_tors"].iloc[use_ids], norm_moments)):
            rot_mat = self._rot.active_2D(angle)
            profile[i, :, :] = (rot_mat@prof.T).T+qc_pos[i, :]

            trailing_edge = (rot_mat@np.c_[(0.75-arrow_scale_moment)*self._chord, 0].T).T+qc_pos[i, :]
            moment_arrow = rot_mat@self.circle_arrow(180*moment)*arrow_scale_moment
            mom_arrow[i, :, :] = moment_arrow.T+trailing_edge.squeeze()
        
        tmp = np.arange(time.size)-1
        trailing_idx_from = tmp-(keep_qc_trailing+tmp%plot_qc_trailing_every)
        trailing_idx_from[trailing_idx_from < 0] = 0
        
        angle_aero_to_xyz = -(self.df_general["pos_tors"].iloc[use_ids]+
                              self.df_f_aero[angle_lift].iloc[use_ids]).to_numpy()
        aero_force = np.c_[self.df_f_aero["aero_drag"].iloc[use_ids], 
                           self.df_f_aero["aero_lift"].iloc[use_ids], 
                           np.zeros(time.size)]
        force_arrows = self._rot.project_separate(aero_force, angle_aero_to_xyz)*arrow_scale_forces
        
        dfs = {"general": self.df_general.iloc[use_ids], 
               "f_aero": self.df_f_aero.iloc[use_ids],
               "f_structural": self.df_f_structural.iloc[use_ids]}
        
        fig, plt_lines, plt_arrows, aoas = self._prepare_force_animation(dfs, time_frame)

        #todo code rad2deg conversion nicer (also in _prepare_force_animation()!)
        data_lines = {linename: self.df_f_aero[linename].iloc[use_ids].to_numpy() for linename in 
                      ["aero_edge", "aero_flap", "aero_mom"]} |\
                     {linename: np.rad2deg(self.df_f_aero[linename].iloc[use_ids].to_numpy()) for linename in aoas} |\
                     {linename: self.df_f_structural[linename].iloc[use_ids].to_numpy() for linename in 
                      ["damp_edge", "damp_flap", "damp_tors", "stiff_edge", "stiff_flap", "stiff_tors"]}
        data_lines = data_lines | {"qc_trail": qc_pos,
                                   "profile": profile, # data with dict[line: data_for_line]
                                   "mom": mom_arrow}
        data_arrows = {"drag": force_arrows[:, :2], "lift": force_arrows[:, 2:4]}

        dt_sim = self.time[1]
        dt_per_frame = dt_sim if dt_per_frame is None else min(dt_per_frame, dt_sim)
        n_frames = int((time_frame[1]-time_frame[0])/dt_per_frame)
        ids_frames = [int(i*dt_per_frame/dt_sim) for i in range(n_frames)]
        
        def update(j: int):
            i = ids_frames[j]
            for linename, data in data_lines.items():
                match linename:
                    case "profile"|"mom":
                        plt_lines[linename].set_data(data[i, :, 0], data[i, :, 1])
                    case "qc_trail":
                        plt_lines[linename].set_data(data[trailing_idx_from[i]:i:plot_qc_trailing_every, 0], 
                                                     data[trailing_idx_from[i]:i:plot_qc_trailing_every, 1])
                    case _:
                        plt_lines[linename].set_data(time[:i], data[:i])
            for arrow_name, data in data_arrows.items():
                plt_arrows[arrow_name].set_data(x=data_lines["qc_trail"][i, 0], y=data_lines["qc_trail"][i, 1], 
                                                dx=data[i, 0], dy=data[i, 1])
            rtrn = [*plt_lines.values()]+[*plt_arrows.values()]
            if not (j+1)%100:
                print(f"Animation until frame number {j+1} of {n_frames} ({np.round((j+1)/n_frames*1e2, 2)}%) done.")
            return tuple(rtrn)
        
        ani = animation.FuncAnimation(fig=fig, func=update, frames=n_frames, interval=15, blit=True)
        writer = animation.FFMpegWriter(fps=30)
        ani.save(join(self.dir_plots, "animation_force.mp4"), writer=writer)

    def energy(
            self,
            angle_lift: str,
            arrow_scale_forces: float=1,
            arrow_scale_moment: float=1,
            plot_qc_trailing_every: int=2,
            keep_qc_trailing: int=40,
            time_frame: tuple[float, float]=None,
            dt_per_frame: float=None):
        use_ids = np.arange(self.time.size)
        time = self.time
        if time_frame is not None:
            time_frame = (time_frame[0], min(time_frame[1], time[-1]))
            use_ids = np.logical_and(self.time >= time_frame[0], self.time <= time_frame[1])
            time = time[use_ids]
        else:
            time_frame = [0, time[-1]]

        qc_pos = self.df_general[["pos_x", "pos_y"]].iloc[use_ids].to_numpy()
        profile = np.zeros((time.size, *self.profile.shape))
        norm_moments = self.df_f_aero["aero_mom"].iloc[use_ids]/np.abs(self.df_f_aero["aero_mom"].iloc[use_ids]).max()
        mom_arrow_res = 40
        mom_arrow = np.zeros((time.size, mom_arrow_res+3, 2))
        prof = self.profile-np.c_[0.25*self._chord, 0]
        for i, (angle, moment) in enumerate(zip(self.df_general["pos_tors"].iloc[use_ids], norm_moments)):
            rot_mat = self._rot.active_2D(angle)
            profile[i, :, :] = (rot_mat@prof.T).T+qc_pos[i, :]

            trailing_edge = (rot_mat@np.c_[(0.75-arrow_scale_moment)*self._chord, 0].T).T+qc_pos[i, :]
            moment_arrow = rot_mat@self.circle_arrow(180*moment)*arrow_scale_moment
            mom_arrow[i, :, :] = moment_arrow.T+trailing_edge.squeeze()
        
        ids = np.arange(time.size)-1
        trailing_idx_from = ids-(keep_qc_trailing+ids%plot_qc_trailing_every)
        trailing_idx_from[trailing_idx_from < 0] = 0
        
        angle_aero_to_xyz = (self.df_general["pos_tors"]-self.df_f_aero[angle_lift]).iloc[use_ids].to_numpy()
        aero_force = np.c_[self.df_f_aero["aero_drag"].iloc[use_ids], 
                           self.df_f_aero["aero_lift"].iloc[use_ids], 
                           np.zeros(time.size)]
        force_arrows = self._rot.project_separate(aero_force, angle_aero_to_xyz)*arrow_scale_forces
        
        df_total = pd.concat([self.df_e_kin["total"].iloc[use_ids], 
                              self.df_e_pot["total"].iloc[use_ids]], keys=["e_kin", "e_pot"], axis=1)
        df_total["e_total"] = df_total.sum(axis=1)
        dfs = {"general": self.df_general.iloc[use_ids], 
               "e_tot": df_total, 
               "power": self.df_power.iloc[use_ids[:-1]], 
               "e_kin": self.df_e_kin.iloc[use_ids], 
               "e_pot": self.df_e_pot.iloc[use_ids]}
        fig, plt_lines, plt_arrows = self._prepare_energy_animation(dfs, time_frame)

        data_lines = {linename: df_total[linename] for linename in ["e_total", "e_kin", "e_pot"]} |\
                     {linename: self.df_power[linename].iloc[use_ids[:-1]].to_numpy() for linename in 
                      ["aero_drag", "aero_lift", "aero_mom", "damp_edge", "damp_flap", "damp_tors"]} |\
                     {linename: self.df_e_kin[linename[linename.rfind("_")+1:]].iloc[use_ids].to_numpy() for linename in ["kin_edge","kin_flap", "kin_tors"]} |\
                     {linename: self.df_e_pot[linename[linename.rfind("_")+1:]].iloc[use_ids].to_numpy() for linename in ["pot_x", "pot_y", "pot_tors"]} 
        data_lines = data_lines | {"qc_trail": qc_pos, # data with dict[line: data_for_line]
                                   "profile": profile, 
                                   "mom": mom_arrow}
        data_arrows = {"drag": force_arrows[:, :2], "lift": force_arrows[:, 2:4]}
        
        dt_sim = self.time[1]
        dt_per_frame = dt_sim if dt_per_frame is None else min(dt_per_frame, dt_sim)
        n_frames = int((time_frame[1]-time_frame[0])/dt_per_frame)
        ids_frames = [int(i*dt_per_frame/dt_sim) for i in range(n_frames)]
        
        def update(j: int):
            i = ids_frames[j]
            for linename, data in data_lines.items():
                match linename:
                    case "profile"|"mom":
                        plt_lines[linename].set_data(data[i, :, 0], data[i, :, 1])
                    case "qc_trail":
                        plt_lines[linename].set_data(data[trailing_idx_from[i]:i:plot_qc_trailing_every, 0], 
                                                     data[trailing_idx_from[i]:i:plot_qc_trailing_every, 1])
                    case _:
                        plt_lines[linename].set_data(time[:i], data[:i])
            for arrow_name, data in data_arrows.items():
                plt_arrows[arrow_name].set_data(x=data_lines["qc_trail"][i, 0], y=data_lines["qc_trail"][i, 1], 
                                                dx=data[i, 0], dy=data[i, 1])
            rtrn = [*plt_lines.values()]+[*plt_arrows.values()]
            if not (j+1)%100:
                print(f"Animation until frame number {j+1} of {n_frames} ({np.round((j+1)/n_frames*1e2, 2)}%) done.")
            return tuple(rtrn)
        
        ani = animation.FuncAnimation(fig=fig, func=update, frames=n_frames, interval=15, blit=True)
        writer = animation.FFMpegWriter(fps=30)
        ani.save(join(self.dir_plots, "animation_energy.mp4"), writer=writer)

    def Beddoes_Leishman(
            self,
            angle_lift: str,
            arrow_scale_forces: float=1,
            arrow_scale_moment: float=1,
            plot_qc_trailing_every: int=2,
            keep_qc_trailing: int=40,
            time_frame: tuple[float, float]=None,
            dt_per_frame: float=None):
        use_ids = np.arange(self.time.size)
        time = self.time
        if time_frame is not None:
            time_frame = (time_frame[0], min(time_frame[1], time[-1]))
            use_ids = np.logical_and(self.time >= time_frame[0], self.time <= time_frame[1])
            time = time[use_ids]
        else:
            time_frame = [0, time[-1]]

        qc_pos = self.df_general[["pos_x", "pos_y"]].iloc[use_ids].to_numpy()
        profile = np.zeros((time.size, *self.profile.shape))
        norm_moments = self.df_f_aero["aero_mom"].iloc[use_ids]/np.abs(self.df_f_aero["aero_mom"].iloc[use_ids]).max()
        mom_arrow_res = 40
        mom_arrow = np.zeros((time.size, mom_arrow_res+3, 2))
        prof = self.profile-np.c_[0.25*self._chord, 0]
        for i, (angle, moment) in enumerate(zip(self.df_general["pos_tors"].iloc[use_ids], norm_moments)):
            rot_mat = self._rot.active_2D(angle)
            profile[i, :, :] = (rot_mat@prof.T).T+qc_pos[i, :]

            trailing_edge = (rot_mat@np.c_[(0.75-arrow_scale_moment)*self._chord, 0].T).T+qc_pos[i, :]
            moment_arrow = rot_mat@self.circle_arrow(180*moment)*arrow_scale_moment
            mom_arrow[i, :, :] = moment_arrow.T+trailing_edge.squeeze()
        
        ids = np.arange(time.size)-1
        trailing_idx_from = ids-(keep_qc_trailing+ids%plot_qc_trailing_every)
        trailing_idx_from[trailing_idx_from < 0] = 0
        
        angle_aero_to_xyz = (self.df_general["pos_tors"]-self.df_f_aero[angle_lift]).iloc[use_ids].to_numpy()
        aero_force = np.c_[self.df_f_aero["aero_drag"].iloc[use_ids], 
                           self.df_f_aero["aero_lift"].iloc[use_ids], 
                           np.zeros(time.size)]
        force_arrows = self._rot.project_separate(aero_force, angle_aero_to_xyz)*arrow_scale_forces
        
        df_total = pd.concat([self.df_e_kin.iloc[use_ids].sum(axis=1), 
                              self.df_e_pot.iloc[use_ids].sum(axis=1)], keys=["e_kin", "e_pot"], axis=1)
        df_total["e_total"] = df_total.sum(axis=1)
        dfs = {"general": self.df_general.iloc[use_ids], "f_aero": self.df_f_aero.iloc[use_ids]}
        fig, plt_lines, plt_arrows, aoas, coeffs = self._prepare_BL_animation(dfs, time_frame)

        coeff_names = [name for names in coeffs.values() for name in names]
        data_lines = {linename: self.df_f_aero[linename].iloc[use_ids].to_numpy() for linename in coeff_names}
        data_lines = data_lines|{linename: np.rad2deg(self.df_f_aero[linename].iloc[use_ids].to_numpy()) for linename in aoas}
        data_lines = data_lines | {"qc_trail": qc_pos, # data with dict[line: data_for_line]
                                   "profile": profile, 
                                   "mom": mom_arrow}
        data_arrows = {"drag": force_arrows[:, :2], "lift": force_arrows[:, 2:4]}
        
        dt_sim = self.time[1]
        dt_per_frame = dt_sim if dt_per_frame is None else min(dt_per_frame, dt_sim)
        n_frames = int((time_frame[1]-time_frame[0])/dt_per_frame)
        ids_frames = [int(i*dt_per_frame/dt_sim) for i in range(n_frames)]
        
        def update(j: int):
            i = ids_frames[j]
            for linename, data in data_lines.items():
                match linename:
                    case "profile"|"mom":
                        plt_lines[linename].set_data(data[i, :, 0], data[i, :, 1])
                    case "qc_trail":
                        plt_lines[linename].set_data(data[trailing_idx_from[i]:i:plot_qc_trailing_every, 0], 
                                                     data[trailing_idx_from[i]:i:plot_qc_trailing_every, 1])
                    case _:
                        plt_lines[linename].set_data(time[:i], data[:i])
            for arrow_name, data in data_arrows.items():
                plt_arrows[arrow_name].set_data(x=data_lines["qc_trail"][i, 0], y=data_lines["qc_trail"][i, 1], 
                                                dx=data[i, 0], dy=data[i, 1])
            rtrn = [*plt_lines.values()]+[*plt_arrows.values()]
            if not (j+1)%100:
                print(f"Animation until frame number {j+1} of {n_frames} ({np.round((j+1)/n_frames*1e2, 2)}%) done.")
            return tuple(rtrn)
                
        ani = animation.FuncAnimation(fig=fig, func=update, frames=n_frames, interval=15, blit=True)
        writer = animation.FFMpegWriter(fps=30)
        ani.save(join(self.dir_plots, "animation_BL.mp4"), writer=writer)


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
            "HHT-alpha-xy": "o.g.",
            "HHT-alpha-xy-adapted": "adpt."
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
            handler = PlotHandler(fig, axs)
            x_labels = {"response": "time (s)", "rel_err": "oscillation number (-)"}
            y_labels = {"response": "position (m)", "rel_err": "relative error (%)"}
            handler.update(x_labels=x_labels, y_labels=y_labels, legend=True)
            handler.save(join(dir_plot, f"{case_number}.pdf"))
        pd.DataFrame(info).to_csv(join(dir_plot, "info.dat"), index=None)

    @staticmethod
    def rotation(root_dir: str, case_id: int=None):
        dir_plot = helper.create_dir(join(root_dir, "plots"))[0]
        schemes = listdir(root_dir)
        schemes.pop(schemes.index("plots"))
        if case_id is not None:
            case_numbers = [[str(case_id)]]
        else:
            case_numbers = []
            for i, scheme in enumerate(schemes):
                case_numbers.append(listdir(join(root_dir, scheme)))
                if len(case_numbers) == 2:
                    if not case_numbers[0] == case_numbers[1]:
                        raise ValueError(f"Scheme '{schemes[i-1]}' and '{scheme[i]}' do not have the same case "
                                         f"numbers: \n "
                                         f"Scheme '{schemes[i-1]}': {case_numbers[0]}\n"
                                         f"Scheme '{schemes[i]}': {case_numbers[1]}\n")
                    case_numbers.pop(0)
                
        info = {}
        map_scheme_to_label = {
            "HHT-alpha-xy": "o.g.",
            "HHT-alpha-xy-adapted": "adpt."
        }
        x_labels = {"pos": {"split": "time (s)", "joined": "x (m)"},
                    "vel": {"split": "time (s)", "joined": "u (m/s)"}}
        y_labels = {"pos": {"split": "position (m), (deg)", "joined": "y (m)"},
                    "vel": {"split": "velocity (m/s), (deg/s)", "joined": "v (m/s)"}}
        for case_number in case_numbers[0]:
            for param in ["pos", "vel"]:
                fig, axs = plt.subplots(1, 2, figsize=(10, 5), tight_layout=True)
                axs = {"split": axs[0], "joined": axs[1]}
                for i, scheme in enumerate(schemes):
                    current_dir = join(root_dir, scheme, case_number)
                    df_sol = pd.read_csv(join(current_dir, "analytical_sol.dat"))
                    df_sim = pd.read_csv(join(current_dir, "general.dat"))

                    for direction in ["x", "y", "tors"]:
                        if i == 0:
                            axs["split"].plot(df_sol["time"], df_sol[f"{param}_{direction}"], "k")
                        axs["split"].plot(df_sim["time"], df_sim[f"{param}_{direction}"], "--", 
                                          label=f"{param} {direction} {scheme}")
                    
                    if i == 0:
                        axs["joined"].plot(df_sol[f"{param}_x"], df_sol[f"{param}_y"], "k", ms=0.5, label="eE")
                    axs["joined"].plot(df_sim[f"{param}_x"], df_sim[f"{param}_y"], "or", ms=0.5, label=f"{scheme}")

                    # with open(join(current_dir, "info.json"), "r") as f:
                    #     tmp_info = json.load(f)|{"scheme": scheme}
                    
                    # if info == {}:
                    #     info = {param: [] for param in tmp_info.keys()}
                    # else:
                    #     for param, value in tmp_info.items():
                    #         info[param].append(value)
                
                handler = PlotHandler(fig, axs)
                handler.update(x_labels=x_labels[param], y_labels=y_labels[param], legend=True)
                handler.save(join(dir_plot, f"{case_number}_{param}.pdf"))
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
        handler = PlotHandler(fig, axs)
        handler.update(legend=legend, y_labels=y_labels, x_labels="time (s)")
        handler.save(join(dir_plots, "pos.pdf"))
        
        handler_err = PlotHandler(fig_err, axs_err)
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

        handler = PlotHandler(fig, axs)
        handler.update(legend=legend, y_labels=y_labels, x_labels="time (s)")
        handler.save(join(dir_plots, "energy.pdf"))

        handler_err = PlotHandler(fig_err, axs_err)
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
        
        if any([specifier in BL_scheme for specifier in ["IAG", "AEROHOR"]]):
            self._BL_preparation(dir_preparation, file_polar, BL_scheme)
        elif any([specifier in BL_scheme for specifier in ["openFAST"]]):
            self._BL_openFAST_preparation(dir_preparation, file_polar, "data/FFA_WA3_221/general_openFAST")

    def _BL_preparation(
            self,
            dir_preparation: str,
            file_polar: str,
            BL_scheme: str
            ):
        df_polar = pd.read_csv(file_polar, delim_whitespace=True)
        if "alpha" not in df_polar:
            df_polar = pd.read_csv(file_polar)

        f_data = {}
        for file in listdir(dir_preparation):
            if "f_" not in file:
                continue
            file_name = file.split(".")[0]
            f_data[f"{file_name[-1]}"] = pd.read_csv(join(dir_preparation, file))
        
        dir_plots = helper.create_dir(join(dir_preparation, "plots"))[0]
        
        fig, ax = plt.subplots()
        alpha = {}  # for plots of coeffs
        f = {}
        x_lims = [-50, 50]
        for direction, df in f_data.items():
            alpha[direction] = df[f"alpha_{direction}"].to_numpy().flatten()
            raoa = np.rad2deg(df[f"alpha_{direction}"])
            x_lims = [max(x_lims[0], raoa.min()), min(x_lims[1], raoa.max())]
            ax.plot(raoa, df[f"f_{direction}"], **self.plt_settings[f"f_{direction}"])
            f[direction] = df[f"f_{direction}"].to_numpy()

        handler = PlotHandler(fig, ax)
        handler.update(x_labels=r"$\alpha$ (°)", y_labels="separation point (-)", legend=True, x_lims=x_lims, 
                       y_lims=[-0.1, 1.1])
        handler.save(join(dir_plots, "sep_points.pdf"))

        with open(join(dir_preparation, "aero_characteristics.json"), "r") as file:
            aero_characteristics = json.load(file)
            
        coeffs = self._reconstruct_force_coeffs(BL_scheme, aero_characteristics, alpha, f)
        if "C_t" in coeffs and "C_n" in coeffs:
            coeffs = np.c_[coeffs["C_t"], coeffs["C_n"]]
        elif "C_n" in coeffs:
            C_l = interp1d(np.deg2rad(df_polar["alpha"]), df_polar["C_l"])
            C_d = interp1d(np.deg2rad(df_polar["alpha"]), df_polar["C_d"])
            C_t = np.sin(alpha["n"])*C_l(alpha["n"])-np.cos(alpha["n"])*C_d(alpha["n"])
            coeffs = np.c_[C_t, coeffs["C_n"]]
        alpha = alpha["n"]
        rot_coeffs = self.rotate_2D(coeffs, alpha)

        fig, ax = plt.subplots()
        ax.plot(df_polar["alpha"], df_polar["C_d"], **self.plt_settings["C_d_specify"])
        ax.plot(df_polar["alpha"], df_polar["C_l"], **self.plt_settings["C_l_specify"])
        if "AEROHOR" in BL_scheme:
            ax.plot(np.rad2deg(alpha), -rot_coeffs[:, 0]+df_polar["C_d"].min(), **self.plt_settings["C_d_rec"])
        else:
            ax.plot(np.rad2deg(alpha), -rot_coeffs[:, 0], **self.plt_settings["C_d_rec"])
        ax.plot(np.rad2deg(alpha), rot_coeffs[:, 1], **self.plt_settings["C_l_rec"])

        handler = PlotHandler(fig, ax)
        handler.update(x_labels=r"$\alpha$ (°)", y_labels="Force coefficient (-)", legend=True)
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
            if "_fs" in file:
                df_C_fs = pd.read_csv(join(dir_preparation, file))
                alpha_fs = df_C_fs["alpha_fs"].to_numpy().flatten()
                C_col = [col for col in df_C_fs if "C" in col][0]
                C_fs = df_C_fs[C_col].to_numpy().flatten()
            elif "f_" in file:
                df_f = pd.read_csv(join(dir_preparation, file))
                cols = df_f.columns
                alpha_f = [col for col in cols if "alpha" in col][0]

        aoa_f = df_f[alpha_f].to_numpy()
        ids_C_fs = np.logical_and(alpha_fs>=aoa_f[1], alpha_fs<=aoa_f[-2])
        alpha_fs = np.rad2deg(alpha_fs[ids_C_fs])
        C_fs = C_fs[ids_C_fs]

        dir_plots = helper.create_dir(join(dir_preparation, "plots"))[0]

        fig, ax = plt.subplots()
        ax.plot(np.rad2deg(df_f[alpha_f]), df_f["f_l"], **self.plt_settings["section"])
        ax.plot(df_f_paper["alpha"], df_f_paper["f_l"], **self.plt_settings[model_of_paper])
        handler = PlotHandler(fig, ax)
        handler.update(x_labels=r"$\alpha$ (°)", y_labels=r"$f$ (-)", x_lims=[-50, 50], y_lims=[-0.1, 1.1],
                       grid=True, legend=True)
        handler.save(join(dir_plots, "sep_point.pdf"))

        fig, ax = plt.subplots()
        alpha_paper = df_C_fs_paper["alpha"].sort_values()
        ax.plot(df_polar["alpha"], df_polar["C_l"], **self.plt_settings["C_l_specify"])
        ax.plot(alpha_fs, C_fs, **self.plt_settings["section"])
        ax.plot(alpha_paper.values, df_C_fs_paper["C_lfs"].loc[alpha_paper.index], **self.plt_settings[model_of_paper])
        handler = PlotHandler(fig, ax)
        handler.update(x_labels=r"$\alpha$ (°)", y_labels=r"$C_{l\text{,fs}}$ (-)", x_lims=[-50, 50], 
                       y_lims=[-1.5, 1.85], legend=True)
        handler.save(join(dir_plots, "C_fully_sep.pdf"))
        
        
    def plot_meas_comparison(
        self,
        root_unsteady_data: str,
        files_unsteady_data: dict[str],
        dir_results: str,
        period_res: int,
        coeffs_polar: dict[str, np.ndarray]=None
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
            handler = PlotHandler(fig, ax)
            if coeff in coeffs_polar:
                ax.plot(*coeffs_polar[coeff], **self.plt_settings[coeff])
            for tool, data in data_meas[coeff].items():
                ax.plot(data[0], data[1], **self.plt_settings[coeff+f"_{tool}"])
                # ax.plot(data[0], data[1], **self.plt_settings[coeff])
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
        if "alpha" not in df_polar:
            df_polar = pd.read_csv(file_polar)
        # df_aerohor = pd.read_csv(join(dir_results, "aerohor_res.dat"))
        df_section = pd.read_csv(join(dir_results, "f_aero.dat"))
        
        dir_save = helper.create_dir(join(dir_results, "plots"))[0]

        alpha_min = case_data["mean"]-case_data["amplitude"]-alpha_buffer
        alpha_max = case_data["mean"]+case_data["amplitude"]+alpha_buffer
        # print(alpha_min, alpha_max)
        df_sliced = df_polar.loc[(df_polar["alpha"] >= alpha_min) & (df_polar["alpha"] <= alpha_max)]
        
        for coef in ["C_d", "C_l", "C_m"]:
            fig, ax = plt.subplots()
            handler = PlotHandler(fig, ax)
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
            handler = PlotHandler(fig, ax)
            ax.plot(t_aerohor, df_aerohor[plot_param], **plt_settings["aerohor"])
            ax.plot(t_section, df_section[plot_param], **plt_settings["section"])
            handler.update(x_labels="t (s)", y_labels=y_labels[plot_param], legend=True)
            handler.save(join(dir_plots, f"model_comp_{plot_param}.pdf"))
    
    @staticmethod
    def _reconstruct_force_coeffs(
        BL_scheme: str, 
        aero_characteristics: dict,
        alpha: dict[str, np.ndarray], 
        f: dict[str, np.ndarray]):
        coeffs = {f"C_{direction}": None for direction in alpha}
        if "AEROHOR" in BL_scheme:
            C_slope = aero_characteristics["C_n_inv_max_slope"]
            alpha_0 = aero_characteristics["C_n_inv_root"]
            alpha_n = alpha["n"]
            alpha_t = alpha["t"]
            f_n = f["n"]
            f_t = f["t"]
        elif "IAG" in BL_scheme:
            C_slope = aero_characteristics["C_n_inv_max_slope"]
            alpha_0 = aero_characteristics["C_n_visc_root"]
            alpha_n = alpha["n"]
            f_n = f["n"]
        elif "openFAST" in BL_scheme:
            alpha_0 = aero_characteristics["C_l_visc_root"]
            C_slope = aero_characteristics["C_l_visc_max_slope"]
            alpha_l = alpha["l"]
            f_l = f["l"]
        else:
            raise NotImplementedError(f"Reconstructing force coefficients for BL scheme {BL_scheme}.")

        if "AEROHOR" in BL_scheme:
                coeffs["C_n"] = C_slope*(alpha_n-alpha_0)*((1+np.sqrt(np.abs(f_n))*np.sign(f_n))/2)**2
                coeffs["C_t"] = C_slope*(alpha_t-alpha_0)*np.tan(alpha_t)*np.sqrt(np.abs(f_t))*np.sign(f_t)
        elif "IAG" in BL_scheme:
                coeffs["C_n"] = C_slope*(alpha_n-alpha_0)*((1+np.sqrt(f_n))/2)**2
        elif "openFAST" in BL_scheme:
                coeffs["C_l"] = C_slope*(alpha_l-alpha_0)*((1+np.sqrt(f_l))/2)**2
        return coeffs

    
def combined_forced(root_dir: str):
    df_combinations = pd.read_csv(join(root_dir, "combinations.dat"))
    amplitudes = np.sort(df_combinations["amplitude"].unique())[::-1]
    aoas = np.sort(df_combinations["alpha"].unique())

    period_aero_work = np.zeros((amplitudes.size, aoas.size))
    norm_period_aero_work = np.zeros((amplitudes.size, aoas.size))
    period_struct_work = np.zeros((amplitudes.size, aoas.size))
    convergence_aero = np.zeros((amplitudes.size, aoas.size))
    convergence_struct = np.zeros((amplitudes.size, aoas.size))

    ampl_to_ind = {ampl: i for i, ampl in enumerate(amplitudes)}
    aoa_to_ind = {aoa: i for i, aoa in enumerate(aoas)}
    for i, row in df_combinations.iterrows():
        ampl = row["amplitude"]
        aoa = row["alpha"] 
        dir_current = join(root_dir, str(i))
        df = pd.read_csv(join(dir_current, "period_work.dat"))

        e_kin_total = pd.read_csv(join(dir_current, "e_kin.dat"), usecols=["total"]).to_numpy().flatten()
        e_pot_total = pd.read_csv(join(dir_current, "e_kin.dat"), usecols=["total"]).to_numpy().flatten()
        e_total = e_kin_total+e_pot_total
        peaks_e = find_peaks(e_total)[0]
        mean_e = e_total[peaks_e[-2]:peaks_e[-1]].mean()

        aero_work = df[["aero_drag", "aero_lift", "aero_mom"]].sum(axis=1).to_numpy()
        struct_work = df[["damp_edge", "damp_flap", "damp_tors"]].sum(axis=1).to_numpy()

        period_aero_work[ampl_to_ind[ampl], aoa_to_ind[aoa]] = aero_work[-1]
        norm_period_aero_work[ampl_to_ind[ampl], aoa_to_ind[aoa]] = aero_work[-1]/mean_e
        period_struct_work[ampl_to_ind[ampl], aoa_to_ind[aoa]] = struct_work[-1]

        convergence_aero[ampl_to_ind[ampl], aoa_to_ind[aoa]] = (aero_work[-1]-aero_work[-2])/aero_work[-1]
        convergence_struct[ampl_to_ind[ampl], aoa_to_ind[aoa]] = (struct_work[-1]-struct_work[-2])/struct_work[-1]

    dir_plots = helper.create_dir(join(root_dir, "plots"))[0]
    fig, ax = plt.subplots()
    cmap, norm = get_colourbar(period_aero_work)
    ax = heatmap(period_aero_work, xticklabels=np.round(aoas, 3), ax=ax, yticklabels=np.round(amplitudes, 3),
                 cbar_kws={"label": r"$W_{\text{aero}}$ per period (Nm/s)"}, cmap=cmap, norm=norm, annot=True, 
                 fmt=".3g")
    cbar = ax.collections[0].colorbar
    stab_min = period_aero_work.min()
    stab_max = period_aero_work.max()
    if stab_max > 0:
        cbar.set_ticks([stab_min, 0, stab_max])
        cbar.set_ticklabels([np.round(stab_min, 3), 0, np.round(stab_max, 3)])
    else:
        cbar.set_ticks([stab_min, 0])
        cbar.set_ticklabels([np.round(stab_min, 3), 0])
    cbar.minorticks_off()
    handler = PlotHandler(fig, ax)
    handler.update(x_labels="angle of attack (°)", y_labels="oscillation amplitude factor (-)")
    handler.save(join(dir_plots, "period_aero_work.pdf"))

    dir_plots = helper.create_dir(join(root_dir, "plots"))[0]
    fig, ax = plt.subplots()
    cmap, norm = get_colourbar(norm_period_aero_work)
    ax = heatmap(norm_period_aero_work, xticklabels=np.round(aoas, 3), ax=ax, yticklabels=np.round(amplitudes, 3),
                 cbar_kws={"label": r"$W_{\text{aero}}/\bar{E}_{\text{total}}$ per period (-)"}, cmap=cmap, 
                 norm=norm, annot=True, fmt=".3g")
    cbar = ax.collections[0].colorbar
    stab_min = norm_period_aero_work.min()
    stab_max = norm_period_aero_work.max()
    if stab_max > 0:
        cbar.set_ticks([stab_min, 0, stab_max])
        cbar.set_ticklabels([np.round(stab_min, 3), 0, np.round(stab_max, 3)])
    else:
        cbar.set_ticks([stab_min, 0])
        cbar.set_ticklabels([np.round(stab_min, 3), 0])
    cbar.minorticks_off()
    handler = PlotHandler(fig, ax)
    handler.update(x_labels="angle of attack (°)", y_labels="oscillation amplitude factor (-)")
    handler.save(join(dir_plots, "normalised_period_aero_work.pdf"))

    fig, ax = plt.subplots()
    cmap, norm = get_colourbar(convergence_aero)
    ax = heatmap(convergence_aero, xticklabels=np.round(aoas, 3), ax=ax, yticklabels=np.round(amplitudes, 3),
                 cbar_kws={"label": r"convergence"}, cmap=cmap, norm=norm, annot=True, fmt=".3g")
    cbar = ax.collections[0].colorbar
    stab_min = convergence_aero.min()
    stab_max = convergence_aero.max()
    cbar.set_ticks([stab_min, 0, stab_max])
    cbar.set_ticklabels([np.round(stab_min, 3), 0, np.round(stab_max, 3)])
    if stab_max > 0:
        cbar.set_ticks([stab_min, 0, stab_max])
        cbar.set_ticklabels([np.round(stab_min, 3), 0, np.round(stab_max, 3)])
    else:
        cbar.set_ticks([stab_min, 0])
        cbar.set_ticklabels([np.round(stab_min, 3), 0])
    cbar.minorticks_off()
    handler = PlotHandler(fig, ax)
    handler.update(x_labels="angle of attack (°)", y_labels="oscillation amplitude factor (-)")
    handler.save(join(dir_plots, "convergence_aero.pdf"))

    fig, ax = plt.subplots()
    cmap, norm = get_colourbar(period_struct_work)
    ax = heatmap(period_struct_work, xticklabels=np.round(aoas, 3), ax=ax, yticklabels=np.round(amplitudes, 3),
                 cbar_kws={"label": r"$W_{\text{struct}}$ work per period (Nm/s)"}, cmap=cmap, norm=norm, annot=True, 
                 fmt=".3g")
    cbar = ax.collections[0].colorbar
    stab_min = period_struct_work.min()
    stab_max = period_struct_work.max()
    cbar.set_ticks([stab_min, 0, stab_max])
    cbar.set_ticklabels([np.round(stab_min, 3), 0, np.round(stab_max, 3)])
    if stab_max > 0:
        cbar.set_ticks([stab_min, 0, stab_max])
        cbar.set_ticklabels([np.round(stab_min, 3), 0, np.round(stab_max, 3)])
    else:
        cbar.set_ticks([stab_min, 0])
        cbar.set_ticklabels([np.round(stab_min, 3), 0])
    cbar.minorticks_off()
    handler = PlotHandler(fig, ax)
    handler.update(x_labels="angle of attack (°)", y_labels="oscillation amplitude factor (-)")
    handler.save(join(dir_plots, "period_struct_work.pdf"))

    fig, ax = plt.subplots()
    cmap, norm = get_colourbar(convergence_struct)
    ax = heatmap(convergence_struct, xticklabels=np.round(aoas, 3), ax=ax, yticklabels=np.round(amplitudes, 3),
                 cbar_kws={"label": r"convergence"}, cmap=cmap, norm=norm, annot=True, fmt=".3g")
    cbar = ax.collections[0].colorbar
    stab_min = convergence_struct.min()
    stab_max = convergence_struct.max()
    cbar.set_ticks([stab_min, 0, stab_max])
    cbar.set_ticklabels([np.round(stab_min, 3), 0, np.round(stab_max, 3)])
    if stab_max > 0:
        cbar.set_ticks([stab_min, 0, stab_max])
        cbar.set_ticklabels([np.round(stab_min, 3), 0, np.round(stab_max, 3)])
    else:
        cbar.set_ticks([stab_min, 0])
        cbar.set_ticklabels([np.round(stab_min, 3), 0])
    cbar.minorticks_off()
    handler = PlotHandler(fig, ax)
    handler.update(x_labels="angle of attack (°)", y_labels="oscillation amplitude factor (-)")
    handler.save(join(dir_plots, "convergence_struct.pdf"))
    
    fig, ax = plt.subplots()
    stability = period_aero_work+period_struct_work
    cmap, norm = get_colourbar(stability)
    ax = heatmap(stability, xticklabels=np.round(aoas, 3), ax=ax, 
                 yticklabels=np.round(amplitudes, 3), cbar_kws={"label": "total period work (Nm/s)"}, cmap=cmap, 
                 norm=norm, annot=True, fmt=".3g")
    cbar = ax.collections[0].colorbar
    stab_min = stability.min()
    stab_max = stability.max()
    cbar.set_ticks([stab_min, 0, stab_max])
    cbar.set_ticklabels([np.round(stab_min, 3), 0, np.round(stab_max, 3)])
    if stab_max > 0:
        cbar.set_ticks([stab_min, 0, stab_max])
        cbar.set_ticklabels([np.round(stab_min, 3), 0, np.round(stab_max, 3)])
    else:
        cbar.set_ticks([stab_min, 0])
        cbar.set_ticklabels([np.round(stab_min, 3), 0])
    cbar.minorticks_off()
    handler = PlotHandler(fig, ax)
    handler.update(x_labels="angle of attack (°)", y_labels="oscillation amplitude factor (-)")
    handler.save(join(dir_plots, "stability.pdf"))


def combined_LOC_amplitude(
        root_dir: str,
        add_resolution: bool=True
):
    df_combinations = pd.read_csv(join(root_dir, "combinations.dat"))
    velocities = np.sort(df_combinations["velocity"].unique())
    aoas = np.sort(df_combinations["alpha"].unique())[::-1]

    amplitudes = np.zeros((aoas.size, velocities.size))
    convergence = np.zeros((aoas.size, velocities.size))
    aoa_types = ["alpha_qs", "alpha_eff"]
    max_aoa = {aoa_type: np.zeros((aoas.size, velocities.size)) for aoa_type in aoa_types}

    vel_to_ind = {vel: i for i, vel in enumerate(velocities)}
    aoa_to_ind = {aoa: i for i, aoa in enumerate(aoas)}
    for i, row in df_combinations.iterrows():
        vel = row["velocity"]
        aoa = row["alpha"] 
        dir_current = join(root_dir, str(i))
        pos_x = pd.read_csv(join(dir_current, "general.dat"), usecols=["pos_x"]).to_numpy().flatten()

        ppeaks = find_peaks(pos_x)[0]
        npeaks = find_peaks(-pos_x)[0]
        
        ampl_last = np.abs(pos_x[ppeaks[-1]]-pos_x[npeaks[-1]])/2
        amplitudes[aoa_to_ind[aoa], vel_to_ind[vel]] = abs(ampl_last)

        ampl_second_to_last = np.abs(pos_x[ppeaks[-2]]-pos_x[npeaks[-2]])/2
        converg = (ampl_last-ampl_second_to_last)/(ampl_second_to_last)
        convergence[aoa_to_ind[aoa], vel_to_ind[vel]] = converg

        df_aero = pd.read_csv(join(dir_current, "f_aero.dat"))
        start = ppeaks[-2]
        end = ppeaks[-1]
        for aoa_type in aoa_types:
            max_aoa[aoa_type][aoa_to_ind[aoa], vel_to_ind[vel]] = df_aero[aoa_type].iloc[start:end].abs().max()


    dir_plots = helper.create_dir(join(root_dir, "plots"))[0]
    levels = 20
    v, a = np.meshgrid(velocities, aoas)
    points = np.c_[(v.ravel(), a.ravel())]

    fig, ax = plt.subplots()
    cf = ax.contourf(v, a, amplitudes, cmap=plt.get_cmap("OrRd"), levels=levels)
    fig.colorbar(cf, ax=ax, label="LCO amplitude (m)")
    handler = PlotHandler(fig, ax)
    handler.update(x_labels="velocity (m/s)", y_labels="angle of attack (°)")
    if add_resolution:
        ax.plot(points[:, 0], points[:, 1], "ok", ms=0.4)
    handler.save(join(dir_plots, "LCO_amplitude_contourf.pdf"), close=False)

    fig, ax = plt.subplots()
    cf = ax.contourf(v, a, convergence*1e2, cmap=plt.get_cmap("RdYlGn_r"), levels=levels)
    fig.colorbar(cf, ax=ax, label="% rel. change in amplitude (-)")
    handler = PlotHandler(fig, ax)
    handler.update(x_labels="velocity (m/s)", y_labels="angle of attack (°)")
    if add_resolution:
        ax.plot(points[:, 0], points[:, 1], "ok", ms=0.4)
    handler.save(join(dir_plots, "LCO_amplitude_convergence_contourf.pdf"))
    
    aoa_label = {"alpha_qs": r"$\alpha_{qs}$", "alpha_eff": r"$\alpha_{eff}$"}
    for aoa_type in aoa_types:
        fig, ax = plt.subplots()
        cf = ax.contourf(v, a, np.rad2deg(max_aoa[aoa_type]), cmap=plt.get_cmap("OrRd"), levels=levels)
        fig.colorbar(cf, ax=ax, label="max "+ aoa_label[aoa_type]+ " during last period (°)")
        handler = PlotHandler(fig, ax)
        handler.update(x_labels="velocity (m/s)", y_labels="angle of attack (°)")
        if add_resolution:
            ax.plot(points[:, 0], points[:, 1], "ok", ms=0.4)
        handler.save(join(dir_plots, f"max_{aoa_type}.pdf"))


    # fig, ax = plt.subplots()
    # cmap, norm = get_colourbar(amplitudes)
    # ax = heatmap(amplitudes, xticklabels=np.round(velocities, 3), ax=ax, yticklabels=np.round(aoas, 3),
    #              cbar_kws={"label": "LCO amplitude (m)"}, cmap=cmap, norm=norm, annot=True, fmt=".3g")
    # cbar = ax.collections[0].colorbar
    # stab_min = amplitudes.min()
    # stab_max = amplitudes.max()
    # cbar.set_ticks([0, stab_max])
    # cbar.set_ticklabels([0, np.round(stab_max, 3)])
    # cbar.minorticks_off()
    # handler = PlotHandler(fig, ax)
    # handler.update(x_labels="velocity (m/s)", y_labels="angle of attack (°)")
    # handler.save(join(dir_plots, "LCO_amplitude_heat_map.pdf"))


    # dir_plots = helper.create_dir(join(root_dir, "plots"))[0]
    # fig, ax = plt.subplots()
    # cmap, norm = get_colourbar(amplitudes)
    # ax = heatmap(convergence, xticklabels=np.round(velocities, 3), ax=ax, yticklabels=np.round(aoas, 3),
    #              cbar_kws={"label": "rel. change in amplitude (-)"}, cmap=cmap, norm=norm, annot=True, fmt=".3g")
    # cbar = ax.collections[0].colorbar
    # stab_min = convergence.min()
    # stab_max = convergence.max()
    # if stab_max > 0:
    #     cbar.set_ticks([stab_min, 0, stab_max])
    #     cbar.set_ticklabels([np.round(stab_min, 3), 0, np.round(stab_max, 3)])
    # else:
    #     cbar.set_ticks([stab_min, 0])
    #     cbar.set_ticklabels([np.round(stab_min, 3), 0])
    # cbar.minorticks_off()
    # handler = PlotHandler(fig, ax)
    # handler.update(x_labels="velocity (m/s)", y_labels="angle of attack (°)")
    # handler.save(join(dir_plots, "LCO_amplitude_convergence.pdf"))
