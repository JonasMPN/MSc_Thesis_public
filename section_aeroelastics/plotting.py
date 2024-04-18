from utils_plot import MosaicHandler, Shapes
from calculations import Rotations
from defaults import DefaultsPlots, DefaultStructure
import pandas as pd
import numpy as np
import json
import matplotlib.pyplot as plt
from os.path import join, isdir
from helper_functions import Helper
from copy import copy
helper = Helper()

class Plotter(DefaultStructure, DefaultsPlots):
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
        fig, axs = plt.subplot_mosaic([["profile", "aoa", "aero"], 
                                       ["profile", "stiff", "damp"]], figsize=(10, 5), tight_layout=True)
        handler = MosaicHandler(fig, axs)
        x_labels = {
            "profile": "normal (m)",
            "aoa": "t (s)",
            "aero": "t (s)",
            "damp": "t (s)",
            "stiff": "t (s)",
        }
        y_labels = {
            "profile": "tangential (m)",
            "aoa": "angle of attack (°)",
            "aero": "aero (N) or (Nm)",
            "damp": "struct. damping, (N) or (Nm)",
            "stiff": "struct. stiffness, (N) or (Nm)",
        }
        aspect = {
            "profile": "equal"
        }
        fig, axs = handler.update(x_labels=x_labels, y_labels=y_labels, aspect=aspect)
        
        # get final position of the profile. Displace it such that the qc is at (0, 0) at t=0.
        pos = self.df_general[["pos_x", "pos_y", "pos_tors"]].to_numpy()
        rot = self._rot.passive_2D(pos[-1, 2])
        rot_profile = rot@(self.profile[:, :2]-np.asarray([self.section_data["chord"]/4, 0])).T
        rot_profile += np.asarray([[pos[-1, 0]], [pos[-1, 1]]])
        axs["profile"].plot(rot_profile[0, :], rot_profile[1, :], **self.plt_settings["profile"])
        axs["profile"].plot(pos[::trailing_every, 0], pos[::trailing_every, 1], **self.plt_settings["qc_trail"])
        
        # grab all angles of attack from the data
        aoas = [column for column in self.df_f_aero.columns if "alpha" in column]
        aoas.pop(aoas.index("alpha_steady"))
        aoas = ["alpha_steady"] + aoas  # plot "alpha_steady" first
        dfs_aoas = self.df_f_aero[aoas]
        dfs_aoas = dfs_aoas.apply(np.rad2deg)

        # prepare dictionary of dataframes to plot
        dfs = {"aoa": dfs_aoas, "aero": self.df_f_aero, "damp": self.df_f_structural, 
               "stiff": self.df_f_structural}
        
        plot = {
            "aoa": aoas,
            "aero": ["aero_edge", "aero_flap", "aero_mom"],
            "damp": ["damp_edge", "damp_flap", "damp_tors"],
            "stiff": ["stiff_edge", "stiff_flap", "stiff_tors"]
        }
        def param_name(param: str):
            return param if "alpha" in param else param[param.rfind("_")+1:]  
        for ax, cols in plot.items():
            for col in cols:
                try: 
                    self.plt_settings[param_name(col)]
                except KeyError:
                    raise NotImplementedError(f"Default plot styles for '{col}' are missing.")
                axs[ax].plot(self.time, dfs[ax][col].to_numpy(), **self.plt_settings[param_name(col)])
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
        fig, axs = plt.subplot_mosaic([["profile", "total", "work"],
                                       ["profile", "kinetic", "potential"]], figsize=(10, 5), tight_layout=True)
        handler = MosaicHandler(fig, axs)
        x_labels = {
            "profile": "normal (m)",
            "total": "t (s)",
            "work": "t (s)",
            "kinetic": "t (s)",
            "potential": "t (s)",
        }
        y_labels = {
            "profile": "tangential (m)",
            "total": "energy (Nm)",
            "work": "work (Nm)",
            "kinetic": "kinetic energy (Nm)",
            "potential": "potential energy (Nm)",
        }
        aspect = {
            "profile": "equal"
        }
        fig, axs = handler.update(x_labels=x_labels, y_labels=y_labels, aspect=aspect)
        
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
        for ax, cols in plot.items():
            t = self.time if ax != "work" else self.time[:-1]
            for col in cols:
                axs[ax].plot(t, dfs[ax][col].to_numpy(), **self.plt_settings[param_name(col)])
        handler.update(legend=True)
        fig.savefig(join(self.dir_plots, "energy.pdf"))



# class Animator:
#             x_lims_from = {
#             "profile": [self.df_general["pos_x"]-.25, self.df_general["pos_x"]+1],
#             "aoa": self.time,
#             "aero": self.time,
#             "damp": self.time,
#             "stiff": self.time,
#         }
#         aoa_to_plot = [aoa for aoa in self.df_f_aero.columns if "alpha" in aoa]
#         y_lims_from = {
#             "profile": [self.df_general["pos_y"][:, 1]-.4, self.df_general["pos_y"][:, 1]+0.3],
#             "aoa": [np.rad2deg(self.df_["aero"][aoa]) for aoa in aoa_to_plot],
#             "aero": [self.df_f_aero["drag"], self.df_f_aero["lift"], self.df_f_aero["moment"]],
#             "damp": [self.df_f_structural["f_struct_damping_edge"], self.df_f_structural["f_struct_damping_flap"], 
#                      self.df_f_structural["f_struct_damping_tors"]],
#             "stiff": [self.df_f_structural["f_struct_stiff_edge"], self.df_f_structural["f_struct_stiff_flap"], 
#                      self.df_f_structural["f_struct_stiff_tors"]],
#         }