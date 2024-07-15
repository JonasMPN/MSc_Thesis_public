import matplotlib.axes
import matplotlib.figure
from defaults import DefaultPlot
import numpy as np
import pandas as pd
from numpy.typing import ArrayLike
from typing import Callable
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, BoundaryNorm, to_rgba
from copy import deepcopy, copy
from helper_functions import Helper
from os.path import join

class PlotHandler(Helper):
    _default = object()

    _empty_call = "empty_call"
    _skip_call = "skip_call"
    _special_calls = [_empty_call, _skip_call]
    
    def __init__(self, figure, axes):
        self.fig = figure
        if isinstance(axes, dict):
            pass
        elif isinstance(axes, matplotlib.axes.Axes):
            axes = {"_": axes}
        elif isinstance(axes, np.ndarray):
            axes = {i: ax for i, ax in enumerate(axes.flatten())}
        else:
            raise NotImplementedError(f"Handling of 'axes' of type '{type(axes)}' is not supported. Supported types "
                                      "are 'dict', 'matplotlib.axes.Axes.', and 'np.ndarray'.")
        
        self.axs = axes 
        self._ax_labels = self.axs.keys()

    def update(
            self,
            x_labels: str | dict[str, str]=_default,
            y_labels: str | dict[str, str]=_default,
            x_lims: tuple | dict[str, tuple]=_default,
            y_lims: tuple | dict[str, tuple]=_default,
            x_lims_from: tuple | dict=_default,
            y_lims_from: tuple | dict=_default,
            legend: bool | dict[bool]=_default,
            aspect: str | float | dict[str, str] | dict[str, float] =_default,
            equal_x_lim: tuple[str] = _default,
            equal_y_lim: tuple[str] = _default,
            grid: bool | dict[str, bool] = _default,
            scale_limits: float=1,
            ) -> tuple[matplotlib.figure.Figure, matplotlib.axes.Axes]:
        """Function to automate and simplify the annotation of axes created with plt.subplot_mosaic(). Except for 'axs',
        all arguments can be a single value or a dictionary of values. The cases are:
        - single value: all axes will receive this value
        - dictionary: define a value for each 'axs' individually by using the 'axs's label as key of the dictionary.

        :param axs: axes created by matplotlib.pyplot.subplot_mosaic()
        :type axs: dict["str", matplotlib.pyplot.Axis]
        :param x_labels: x labels, _defaults to None
        :type x_labels: str | dict[str], optional
        :param y_labels: y labels, _defaults to None
        :type y_labels: str | dict[str], optional
        :param x_lims: x limits, _defaults to None
        :type x_lims: tuple | dict[tuple], optional
        :param y_lims: y limits, _defaults to None
        :type y_lims: tuple | dict[tuple], optional
        :param x_lims_from: set the x limits based on a list of values such that all values are displayed, _defaults to None
        :type x_lims_from: str | dict, optional
        :param y_lims_from: set the y limits based on a list of values such that all values are displayed, _defaults to None
        :type y_lims_from: str | dict, optional
        :param legend: whether or not to add a legend, _defaults to None
        :type legend: bool | dict[bool], optional
        :param aspect: aspect ratio of axis
        :type aspect: str | float | dict[str] | dict[float]
        :param equal_y_lim: Set axes that are going to have the same y limits. The limits are set to the limits of the 
        largest y limits of the axes.
        :type: tuple[str]
        """
        skip = ["skip", "self", "scale_limits", "x_lims_from", "y_lims_from", "equal_x_lim", "equal_y_lim"]
        lcs = {k: v for k,v in locals().items() if k not in skip}
        
        self._check_input_exculsivity(x_lims, y_lims, x_lims_from, y_lims_from)
        if x_lims_from != self._default:
            lcs["x_lims"] = self._get_limits_from(x_lims_from, scale_limits)
        if y_lims_from != self._default:
            lcs["y_lims"] = self._get_limits_from(y_lims_from, scale_limits)
        if equal_x_lim != self._default:
            raise NotImplementedError
        if equal_y_lim != self._default:
            raise NotImplementedError

        # self._check_options_completeness(self._ax_labels, **lcs)  # not used/wanted
        # filter for parameters that the user wants to update
        is_set = deepcopy({param: values for param, values in lcs.items() if values not in [self._default, None]})
        # deepcopy needed because of the mutable input parameters which are changed in the next two lines
        is_set = self._treat_special_options(**is_set)
        filled = self._handle_options(self._ax_labels, **is_set)
        map_params_to_axs_methods = {
            "x_labels": "set_xlabel",
            "y_labels": "set_ylabel",
            "x_lims": "set_xlim",
            "y_lims": "set_ylim",
            "legend": "legend",
            "aspect": "set_aspect",
            "grid": "grid"
        }
        self._handle_axes(**{map_params_to_axs_methods[param]: values for param, values in filled.items()})
        return self.fig, self.axs
    
    def save(self, save_as: str, close: bool=True):
        save_as = save_as.replace("\\", "/")
        self.create_dir(join(*save_as.split("/")[:-1]))
        self.fig.savefig(save_as)
        if close:
            plt.close(self.fig)
        
    def _handle_axes(
            self,
            set_xlabel: dict[str]=_default,
            set_ylabel: dict[str]=_default,
            set_xlim: dict[tuple]=_default,
            set_ylim: dict[tuple]=_default,
            legend: dict[bool]=_default,
            set_aspect: dict[str] | dict[tuple[float]]=_default,
            grid: dict[str, bool]=_default
            ) -> None:
        """Core function that update the axes. The keyword arguments of this function are the names of class methods of
        pyplot axes. This function goes through all keyword arguments filtering for those that have been set by the 
        user and then applies the method corresponding to the keyword argument to the axis.

        :param set_xlabel: x labels for the axes, given as {axis name: label}, defaults to _default
        :type set_xlabel: dict[str], optional
        :param set_ylabel: y labels for the axes, given as {axis name: label}, defaults to _default
        :type set_ylabel: dict[str], optional
        :param set_xlim: x limits for the axes, given as {axis name: (min, max)}, defaults to _default
        :type set_xlim: dict[tuple], optional
        :param set_ylim: y limits for the axes, given as {axis name: (min, max)}, defaults to _default
        :type set_ylim: dict[tuple], optional
        :param legend: Whether or not to show a legend for the axes, given as {axis name: bool}, defaults to _default
        :type legend: dict[str], optional
        :param set_aspect: Aspect ratio for the axes, given as {axis name: ratio}, defaults to _default
        :type set_aspect: dict[str] | dict[float], optional
        :raises NotImplementedError: This error should not occur. Otherwise, update_to is an unexpected value.
        """
        lcs = {k: v for k,v in locals().items() if k != "self"}
        options_used = {method: values for method, values in lcs.items() if values != self._default}
        for method, values in options_used.items():
            for ax_label, update_to in values.items():
                update = getattr(self.axs[ax_label], method)
                if update_to not in [self._empty_call, self._skip_call]:  # special calls are found as strings
                    update(*update_to)  # the ->*<-update_to is the reason for turning the values to tuples in 
                    # _fill_options()
                elif update_to == self._empty_call:
                    # some axes methods toggle a setting on by an empty call; example: ax.legend()
                    update()
                elif update_to == self._skip_call:
                    # likewise, not toggling the same setting as in the self._empty_call case works by omitting the 
                    # call altogether.
                    continue
                else:
                    raise NotImplementedError

    def _check_input_exculsivity(self, x_lims, y_lims, x_lims_from, y_lims_from) -> None:
        """The limits cannot be simultaneously set to a specific range and be expected to be set by reference values. 
        The "lims" and "lims_from are mutually exclusive.

        :param x_lims: The value it was set to in update()
        :type x_lims: _type_
        :param y_lims: The value it was set to in update()  
        :type y_lims: _type_
        :param x_lims_from: The value it was set to in update()
        :type x_lims_from: _type_
        :param y_lims_from: The value it was set to in update()
        :type y_lims_from: _type_
        :raises ValueError: When a "lims" and "lims_from" are set simultaneously.
        """
        if x_lims != self._default and x_lims_from != self._default:
            raise ValueError("x limits cannot be set and calculated at the same time.")
        if y_lims != self._default and y_lims_from != self._default:
            raise ValueError("y limits cannot be set and calculated at the same time.")
        
    def _treat_special_options(self, **kwargs) -> dict:
        """This function serves as treating settings that are toggled not by a function call with input arguments but by
        simply calling that function without input arguments. Example: ax.legend().
        In the update() call, the user specifies whether they want the setting to be toggled by True or False. This is
        here converted from the boolean value into one of the special calls defined in the class attributes.

        :raises ValueError: Error for coder; occurs when PlotHandler is not updated sufficiently to handle new
        special cases.
        :return: Returns the treated settings.  
        :rtype: dict
        """
        map_params = {
            "legend": {True: self._empty_call, False: self._skip_call},
            "grid": {True: self._empty_call, False: self._skip_call}
        }
        implemented_special_options = map_params.keys()
        for param, values in copy(kwargs).items():
            if param not in implemented_special_options:
                if param in map_params.keys():
                    raise ValueError(f"For coder: update 'map_params' a few lines above to include '{param}'.")
                continue
            if not isinstance(values, dict):
                kwargs[param] = map_params[param][values]
            else:
                for axs_label, value in values.items():
                    kwargs[param][axs_label] = map_params[param][value]
        return kwargs  

    def _handle_options(self, ax_labels: tuple[str], **kwargs) -> dict:
        """This method serves three functions:
        - if all axes should have the same value for a certain setting, define this setting for all axes
        - turns setting values that are not a tuple into a tuple (needed for ._handle_axes(), read there)
        - 

        :param ax_labels: _description_
        :type ax_labels: tuple[str]
        :raises TypeError: _description_
        :return: _description_
        :rtype: _type_
        """
        filled = {}
        for param, values in kwargs.items():  # values is either a dict with values for each axis label or just a value
            # that should be used for all axes
            if not isinstance(values, dict):
                values = {ax_label: values for ax_label in ax_labels}
            
            for ax_label, value in copy(values).items():  
                if isinstance(value, tuple): # turn all values into tuples; needed for ._handle_axes()
                    continue  
                if value in self._special_calls: 
                    # skip transformation of the special calls into tuples; they must remain strings. Needed for for 
                    # ._handle_axes().
                    continue
                try:
                    if isinstance(value, str):  # because tuple() on a string separates each character
                        raise TypeError
                    value = tuple(value)
                except TypeError:
                    value = (value,)
                values[ax_label] = value
            filled[param] = values
        return filled
    
    @staticmethod
    def _get_limits_from(limits_from: tuple | dict[tuple], scale: float) -> tuple[float, float] | dict[tuple[float, float]]:
        """Loops through all values and finds the overall max and min values. Returns those for each axis.

        :param limits_from: tuple containing iterables of values at each index or a dictionary with such a tuple for 
        each axes (key).
        :type limits_from: tuple | dict[tuple]
        :return: Returns the axis limits for for the given data or the axis limits for each axis for the given data.
        :rtype: tuple[float, float] | dict[tuple[float, float]]
        """
        if not isinstance(limits_from, dict):
            try: 
                limits_from[0][0]
            except IndexError:
                limits_from = [limits_from]
            limits_from = {"tmp": limits_from}
        limits = {}
        for ax_label, param_values in limits_from.items():
            try:
                param_values[0][0]
            except (IndexError, TypeError):
                param_values = [param_values]
            lim_max = param_values[0][0]
            lim_min = param_values[0][0]
            for values in param_values:
                lim_max = max(lim_max, max(values))
                lim_min = min(lim_min, min(values))
            mean = (lim_min+lim_max)/2
            limits[ax_label] = ((lim_min-mean)*scale+mean, (lim_max-mean)*scale+mean)
        return limits if "tmp" not in limits.keys() else (limits["tmp"][0], limits["tmp"][1])     
    
    @staticmethod
    def _get_equal_limits(*args):
        pass

    @staticmethod
    def _check_options_completeness(ax_labels: tuple[str], **kwargs):
        """Method that checks whether all axis have received a value for a certain setting. Loops over all available 
        settings.

        :param ax_labels: Mosaic axes labels.
        :type ax_labels: tuple[str]
        :raises ValueError: If a certain setting does not specify values for all axes.
        """
        for param, arg in kwargs.items():
            if isinstance(arg, dict):
                specified_labels = arg.keys()
                for ax_label in ax_labels:
                    if ax_label not in specified_labels:
                        raise ValueError(f"Parameter '{param}' is missing an entry for axis '{ax_label}'.")


class Shapes:
    """Class implementing shapes for pyplots that pyplots doesn't natively support.
    """
    @staticmethod
    def circle_arrow(
        angle: float, 
        head_angle: float=40, 
        head_length: float=0.5, 
        res: float=40) -> np.ndarray:
        """Creates an arrow that spans along a (semi)circle. The arrow and arrowhead are merely lines. Returns
        a (res, 2) numpy.ndarray with the (x, y) values of the arrow.

        :param angle: The angle the arrow spans in degree. Hence, angl=pi creates an arrow along a semicircle.
        :type angle: float
        :param head_angle: Angle in degree that the head spans, defaults to 40
        :type head_angle: float, optional
        :param head_length: Length of each head line, defaults to 0.5
        :type head_length: float, optional
        :param res: Number of points used for the arrow body (the line along the circle), defaults to 40
        :type res: int, optional
        :return: (x, y) coordinates to plot the arrow.
        :rtype: np.ndarrays
        """
        angle = np.deg2rad(angle)
        head_angle = np.deg2rad(head_angle)
        alpha_qs = np.linspace(0, angle, res)
        body_x = np.cos(alpha_qs)
        body_y = np.sin(alpha_qs)

        angle *= (1-head_length/8)
        head_x = [np.sin(angle-head_angle/2)*head_length, 0, np.sin(angle+head_angle/2)*head_length]
        head_x = np.asarray(head_x)+body_x[-1] if angle >= 0 else -np.asarray(head_x)+body_x[-1]

        head_y = [-np.cos(angle-head_angle/2)*head_length, 0, -np.cos(angle+head_angle/2)*head_length]
        head_y = np.asarray(head_y)+body_y[-1] if angle >= 0 else -np.asarray(head_y)+body_y[-1]
        
        return np.c_[np.r_[body_x, head_x], np.r_[body_y, head_y]].T


class PlotPreparation:
    @staticmethod
    def _prepare_force_plot(
        equal_y: tuple[str]=None) -> tuple[matplotlib.figure.Figure, matplotlib.axes.Axes, PlotHandler]:
        """Prepares the pyplot figure and axes and a PlotHandler instance for a force plot.

        :param equal_y: Sets the axes that will have the same y limits, defaults to None
        :type equal_y: tuple[str], optional
        :return: A pyplot figure, axes and an instance of PlotHandler for the same figure and axes
        :rtype: tuple[matplotlib.figure.Figure, matplotlib.axes.Axes, PlotHandler]
        """
        fig, axs = plt.subplot_mosaic([["profile", "aoa", "aero"], 
                                       ["profile", "stiff", "damp"]], figsize=(10, 5), tight_layout=True, dpi=300)
        handler = PlotHandler(fig, axs)
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
        return *handler.update(x_labels=x_labels, y_labels=y_labels, aspect=aspect), handler

    @staticmethod
    def _prepare_energy_plot(
            equal_y: tuple[str]=None
            ) -> tuple[matplotlib.figure.Figure, matplotlib.axes.Axes, PlotHandler]:
        """Prepares the pyplot figure and axes and a PlotHandler instance for a energy plot.

        :param equal_y: Sets the axes that will have the same y limits, defaults to None
        :type equal_y: tuple[str], optional
        :return: A pyplot figure, axes and an instance of PlotHandler for the same figure and axes
        :rtype: tuple[matplotlib.figure.Figure, matplotlib.axes.Axes, PlotHandler]
        """
        fig, axs = plt.subplot_mosaic([["profile", "total", "power"],
                                       ["profile", "kinetic", "potential"]], figsize=(10, 5), tight_layout=True,
                                      dpi=300)
        handler = PlotHandler(fig, axs)
        x_labels = {
            "profile": "normal (m)",
            "total": "t (s)",
            "power": "t (s)",
            "kinetic": "t (s)",
            "potential": "t (s)",
        }
        y_labels = {
            "profile": "tangential (m)",
            "total": "energy (Nm)",
            "power": "power (Nm)",
            "kinetic": "kinetic energy (Nm)",
            "potential": "potential energy (Nm)",
        }
        aspect = {
            "profile": "equal"
        }
        return *handler.update(x_labels=x_labels, y_labels=y_labels, aspect=aspect), handler
    
    @staticmethod
    def _prepare_BL_plot(
            coeffs: list[str],
            equal_y: tuple[str]=None,
            ) -> tuple[matplotlib.figure.Figure, matplotlib.axes.Axes, PlotHandler]:
        """Prepares the pyplot figure and axes and a PlotHandler instance for a energy plot.

        :param equal_y: Sets the axes that will have the same y limits, defaults to None
        :type equal_y: tuple[str], optional
        :return: A pyplot figure, axes and an instance of PlotHandler for the same figure and axes
        :rtype: tuple[matplotlib.figure.Figure, matplotlib.axes.Axes, PlotHandler]
        """
        fig, axs = plt.subplot_mosaic([["profile", "aoa", coeffs[0]],
                                       ["profile", coeffs[1], "C_m"]], figsize=(10, 5), tight_layout=True,
                                      dpi=300)
        handler = PlotHandler(fig, axs)
        x_labels = {
            "profile": "normal (m)",
            "aoa": "t (s)",
            coeffs[0]: "t (s)",
            coeffs[1]: "t (s)",
            "C_m": "t (s)",
        }
        y_labels = {
            "profile": "tangential (m)",
            "aoa": "angle of attack (°)",
            coeffs[0]: f"contribution to ${coeffs[0]}$ (-)",
            coeffs[1]: f"contribution to ${coeffs[1]}$ (-)",
            "C_m": f"contribution to $C_m$ (-)"
        }
        aspect = {
            "profile": "equal"
        }
        return *handler.update(x_labels=x_labels, y_labels=y_labels, aspect=aspect), handler
    
    @staticmethod
    def _get_aoas(df_aero: pd.DataFrame) -> tuple[list[str], pd.DataFrame]:
        """Extracts all columns out of the dataframe "df_aero" that are angles of attack. It does that 
        by looking for columns that have "alpha" in them. Then returns the name of the columns and a 
        new dataframe containing these columns. The values from "df_aero" to the new dataframe undergo
        a radian to degree conversion.

        :param df_aero: Dataframe containing the simulation results related to the aerodynamics.
        :type df_aero: pd.DataFrame
        :return: A list of the columns that contain angles of attack and a dataframe of these values. A radians
        to degree conversion is applied to the new dataframe.
        :rtype: tuple[list[str], pd.DataFrame]
        """
        aoas = [column for column in df_aero.columns if "alpha" in column and column != "d_alpha_qs_dt"]
        aoas.pop(aoas.index("alpha_steady"))
        aoas = ["alpha_steady"] + aoas  # plot "alpha_steady" first
        return aoas
    
    @staticmethod
    def _get_force_and_moment_coeffs(df_aero: pd.DataFrame):
        params = df_aero.columns
        if any(["C_n" in param for param in params]):
            collect_with =  ["C_n", "C_t"]
        elif any(["C_l" in param for param in params]):
            collect_with = ["C_l", "C_d"]
        else:
            raise ValueError("No parameters with 'C_l' or 'C_n' found in 'df_aero'.")
        collect_with.append("C_m")
        coeffs = {category: None for category in collect_with}
        for coeff_category in collect_with:
            coeffs[coeff_category] = [param for param in params if coeff_category in param]
        return coeffs
    

class AnimationPreparation(PlotPreparation, DefaultPlot):
    def __init__(self, chord: float) -> None:
        DefaultPlot.__init__(self)
        self._chord = chord
        
    
    def _prepare_force_animation(self, dfs: pd.DataFrame, until: float, equal_y: tuple[str]=None):
        fig, axs, handler = self._prepare_force_plot(equal_y)
        aoas = self._get_aoas(dfs["f_aero"])
        x_lims_from = {
            "profile": [dfs["general"]["pos_x"]-.4*self._chord, dfs["general"]["pos_x"]+1*self._chord],
            "aoa": [0, until],
            "aero": [0, until],
            "damp": [0, until],
            "stiff": [0, until],
        }
        y_lims_from = {
            "profile": [dfs["general"]["pos_y"]-.4*self._chord, dfs["general"]["pos_y"]+0.3*self._chord],
            "aoa": [np.rad2deg(dfs["f_aero"][col]) for col in aoas],
            "aero": [dfs["f_aero"]["aero_edge"], dfs["f_aero"]["aero_flap"], self.df_f_aero["aero_mom"]],
            "damp": [dfs["f_structural"]["damp_edge"], dfs["f_structural"]["damp_flap"], 
                     dfs["f_structural"]["damp_tors"]],
            "stiff": [dfs["f_structural"]["stiff_edge"], dfs["f_structural"]["stiff_flap"], 
                      dfs["f_structural"]["stiff_tors"]],
        }
        
        plot = {
            "profile": ["qc_trail", "profile", "drag", "lift", "mom"],
            "aoa": aoas,
            "aero": ["aero_edge", "aero_flap", "aero_mom"],
            "damp": ["damp_edge", "damp_flap", "damp_tors"],
            "stiff": ["stiff_edge", "stiff_flap", "stiff_tors"]
        }
        def map_cols_to_settings(column: str) -> str:
            if any([force_type in column for force_type in ["aero", "damp", "stiff"]]):
                return column[column.find("_")+1:]
            elif column in ["drag", "lift"]:
                return f"arrow_{column}"
            else:
                return column 
        fig, axs = handler.update(x_lims_from=x_lims_from, y_lims_from=y_lims_from, scale_limits=1.2)
        plt_lines, plt_arrows = self._get_lines_and_arrows(axs, plot, map_cols_to_settings)
        fig, axs = handler.update(legend=True)
        return fig, plt_lines, plt_arrows, aoas
    
    def _prepare_energy_animation(self, dfs: pd.DataFrame, equal_y: tuple[str]=None):
        fig, axs, handler = self._prepare_energy_plot(equal_y)
        x_lims_from = {
            "profile": [dfs["general"]["pos_x"]-.4, dfs["general"]["pos_x"]+1],
            "total": dfs["general"]["time"],
            "power": dfs["general"]["time"],
            "kinetic": dfs["general"]["time"],
            "potential": dfs["general"]["time"],
        }
        y_lims_from = {
            "profile": [dfs["general"]["pos_y"]-.4, dfs["general"]["pos_y"]+0.3],
            "total": [dfs["e_tot"]["e_total"], dfs["e_tot"]["e_kin"], dfs["e_tot"]["e_pot"]],
            "power": [dfs["power"]["aero_drag"], dfs["power"]["aero_lift"], dfs["power"]["aero_mom"],
                     dfs["power"]["damp_edge"], dfs["power"]["damp_flap"], dfs["power"]["damp_tors"]],
            "kinetic": [dfs["e_kin"]["edge"], dfs["e_kin"]["flap"], dfs["e_kin"]["tors"]],
            "potential": [dfs["e_pot"]["edge"], dfs["e_pot"]["flap"], dfs["e_pot"]["tors"]],
        }
        
        plot = {
            "profile": ["qc_trail", "profile", "drag", "lift", "mom"],
            "total": ["e_total", "e_kin", "e_pot"],
            "power": ["aero_drag", "aero_lift", "aero_mom", "damp_edge", "damp_flap", "damp_tors"],
            "kinetic": ["kin_edge", "kin_flap", "kin_tors"],
            "potential": ["pot_edge", "pot_flap", "pot_tors"]
        }
        def map_cols_to_settings(column: str) -> str:
            return column
        fig, axs = handler.update(x_lims_from=x_lims_from, y_lims_from=y_lims_from, scale_limits=1.2)
        plt_lines, plt_arrows = self._get_lines_and_arrows(axs, plot, map_cols_to_settings)
        fig, axs = handler.update(legend=True)
        return fig, plt_lines, plt_arrows
    
    def _get_lines_and_arrows(   
            self,
            axes: dict[str, matplotlib.axes.Axes],
            plot: dict[str, list[str]],
            map_column_to_settings: Callable) -> tuple[dict[str, matplotlib.lines.Line2D],
                                                       dict[str, matplotlib.patches.Patch]]:
        lines = {}
        force_arrows = {}
        for ax, cols in plot.items():
            for col in cols:
                try: 
                    self.plt_settings[map_column_to_settings(col)]
                except KeyError:
                    raise NotImplementedError(f"Default plot styles for '{map_column_to_settings(col)}' are missing.")
                if col in ["lift", "drag"]:
                    force_arrows[col] = axes[ax].arrow(0, 0, 0, 0, **self.plt_settings[map_column_to_settings(col)])
                else:
                    lines[col] = axes[ax].plot(0, 0, **self.plt_settings[map_column_to_settings(col)])[0]
        return lines, force_arrows
    

def get_colourbar(values: np.ndarray):
    ncolors = 256

    min_val = values.min()
    max_val = values.max()
    val_range = abs(max_val-min_val)

    scale_GrYl = abs(min_val/val_range)
    if max_val > 0:
        colors = [(0.0, "green"), (scale_GrYl, "yellow"), (1.0, "red")]
        bounds = np.r_[np.linspace(min_val, 0, int(ncolors*scale_GrYl), endpoint=False), 
                       np.linspace(0, max_val, int(ncolors*(1-scale_GrYl)))]
    else:
        colors = [to_rgba("green"), to_rgba("yellow")]
        bounds = np.linspace(min_val, 0, ncolors)

    cmap = LinearSegmentedColormap.from_list("custom_cmap", colors, N=ncolors)    
    norm = BoundaryNorm(boundaries=bounds, ncolors=ncolors)
    return cmap, norm
