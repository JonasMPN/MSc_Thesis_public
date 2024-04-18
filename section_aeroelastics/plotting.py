import matplotlib.axes
import matplotlib.figure
from utils_plot import MosaicHandler, Shapes, PlotPreparation
from calculations import Rotations
from defaults import DefaultsPlots, DefaultStructure
import pandas as pd
import numpy as np
import json
import matplotlib.pyplot as plt
import matplotlib
from os.path import join
from helper_functions import Helper
from copy import copy
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
        rot = self._rot.passive_2D(pos[-1, 2])
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

    def _plot_to_mosaic(
            self,
            axes: dict[str: matplotlib.axes.Axes],
            plot: dict[str: list[str]],
            data: dict[str: pd.DataFrame],
            map_column_to_settings: callable) -> dict[str: matplotlib.axes.Axes]:

        for ax, cols in plot.items():
            time = self.time if ax != "work" else self.time[:-1]
            for col in cols:
                try: 
                    self.plt_settings[map_column_to_settings(col)]
                except KeyError:
                    raise NotImplementedError(f"Default plot styles for '{col}' are missing.")
                axes[ax].plot(time, data[ax][col].to_numpy(), **self.plt_settings[map_column_to_settings(col)])
        return axes


class Animator(DefaultStructure, DefaultsPlots, Shapes, PlotPreparation):
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
    
    def force(self):
        fig, ax, handler = self._prepare_force_plot()
        x_lims_from = {
        "profile": [self.df_general["pos_x"]-.25, self.df_general["pos_x"]+1],
        "aoa": self.time,
        "aero": self.time,
        "damp": self.time,
        "stiff": self.time,
        }
        aoas, df_aoas = self._get_aoas(self.df_f_aero)
        y_lims_from = {
            "profile": [self.df_general["pos_y"][:, 1]-.4, self.df_general["pos_y"][:, 1]+0.3],
            "aoa": [df_aoas[col] for col in df_aoas.columns],
            "aero": [self.df_f_aero["drag"], self.df_f_aero["lift"], self.df_f_aero["moment"]],
            "damp": [self.df_f_structural["f_struct_damping_edge"], self.df_f_structural["f_struct_damping_flap"], 
                        self.df_f_structural["f_struct_damping_tors"]],
            "stiff": [self.df_f_structural["f_struct_stiff_edge"], self.df_f_structural["f_struct_stiff_flap"], 
                        self.df_f_structural["f_struct_stiff_tors"]],
        }
        ax, fig = handler.update(x_lims_from=x_lims_from, y_lims_from=y_lims_from)

        