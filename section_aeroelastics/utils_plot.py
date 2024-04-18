import matplotlib.figure
import numpy as np
import pandas as pd
from numpy.typing import ArrayLike
from typing import Any
import matplotlib
from copy import copy

class MosaicHandler:
    _default = object()

    _special_options_implemented = ["legend"]
    _empty_call = "empty_call"
    _skip_call = "skip_call"
    _special_calls = [_empty_call, _skip_call]
    
    def __init__(self, mosaic_figure, mosaic_axes):
        self.fig = mosaic_figure
        self.axs = mosaic_axes
        self._ax_labels = self.axs.keys()

    def update(
            self,
            x_labels: str | dict[str]=_default,
            y_labels: str | dict[str]=_default,
            x_lims: tuple | dict[tuple]=_default,
            y_lims: tuple | dict[tuple]=_default,
            x_lims_from: tuple | dict=_default,
            y_lims_from: tuple | dict=_default,
            legend: bool | dict[bool]=_default,
            aspect: str | float | dict[str] | dict[float] =_default,
            equal_y_lim: tuple[str] = _default
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
        lcs = {k: v for k,v in locals().items() if k != "self"}
        
        self._check_input_exculsivity(x_lims, y_lims, x_lims_from, y_lims_from)
        if x_lims_from != self._default:
            lcs["x_lims_from"] = self._get_limits(x_lims_from)
        if y_lims_from != self._default:
            lcs["y_lims_from"] = self._get_limits(y_lims_from)

        # self._check_mosaic_options_completeness(self._ax_labels, **lcs)  # not used/wanted
        # filter for parameters that the user wants to update
        is_set = {param: values for param, values in lcs.items() if values != self._default}
        is_set = self._treat_special_options(**is_set)
        filled = self._handle_mosaic_options(self._ax_labels, **is_set)
        map_params_to_axs_methods = {
            "x_labels": "set_xlabel",
            "y_labels": "set_ylabel",
            "x_lims": "set_xlim",
            "y_lims": "set_ylim",
            "x_lims_from": "set_xlim",
            "y_lims_from": "set_ylim",
            "legend": "legend",
            "aspect": "set_aspect"
        }
        self._handle_mosaic(**{map_params_to_axs_methods[param]: values for param, values in filled.items()})
        return self.fig, self.axs
        
    def _handle_mosaic(
            self,
            set_xlabel: dict[str]=_default,
            set_ylabel: dict[str]=_default,
            set_xlim: dict[tuple]=_default,
            set_ylim: dict[tuple]=_default,
            legend: dict[bool]=_default,
            set_aspect: dict[str] | dict[tuple[float]]=_default
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
                if update_to not in [self._empty_call, self._skip_call]:  # spsecial calls are found as strings
                    update(*update_to)  # the ->*<-update_to is the reason for turning the values to tuples in 
                    # _fill_mosaic_options()
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

        :raises ValueError: Error for coder; occurs when MosaicHandler is not updated sufficiently to handle new
        special cases.
        :return: Returns the treated settings.  
        :rtype: dict
        """
        map_params = {
            "legend": {True: self._empty_call, False: self._skip_call}
        }
        for param, values in copy(kwargs).items():
            if param not in self._special_options_implemented:
                if param in map_params.keys():
                    raise ValueError("For coder: update 'self._special_options_implemented'.")
                continue
            if not isinstance(values, dict):
                kwargs[param] = map_params[param][values]
            else:
                for axs_label, value in values.items():
                    kwargs[param][axs_label] = map_params[param][value]
        return kwargs  

    def _handle_mosaic_options(self, ax_labels: tuple[str], **kwargs) -> dict:
        """This method serves three functions:
        - if all axes should have the same value for a certain setting, define this setting for all axes
        - turns setting values that are not a tuple into a tuple (needed for ._handle_mosaic(), read there)
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
                if isinstance(value, tuple): # turn all values into tuples; needed for ._handle_mosaic()
                    continue  
                if value in self._special_calls: 
                    # skip transformation of the special calls into tuples; they must remain strings. Needed for for 
                    # ._handle_mosaic().
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
    def _get_limits(limits_from: tuple | dict[tuple]) -> tuple[float, float] | dict[tuple[float, float]]:
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
            except IndexError:
                param_values = [param_values]
            lim_max = param_values[0][0]
            lim_min = param_values[0][0]
            for values in param_values:
                lim_max = max(lim_max, max(values))
                lim_min = min(lim_min, min(values))
            limits[ax_label] = (lim_min, lim_max)
        return limits if "tmp" not in limits.keys() else (limits["tmp"][0], limits["tmp"][1])     
    
    @staticmethod
    def _check_mosaic_options_completeness(ax_labels: tuple[str], **kwargs):
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
