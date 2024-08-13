import matplotlib.pyplot as plt
from abc import ABC, abstractmethod
from copy import copy
import numpy as np
plt.rcParams.update({"font.size": 10})


class _BaseDefaultPltSettings(ABC):
    line = {
        "color": "black",
        "lw": 1,
        "ls": None,
        "marker": None,
        "ms": 3,
    }

    arrow = {
        "width": 1,
        "color": "black"
    }

    def __init__(self, kind: str) -> None:
        ABC.__init__(self)
        self.kind = kind

        # add copy parameters to class attributes so they can be accessed in the definition of self.plot_settings
        for param, copy_from in self._copy_params.items():
            for setting in self._settings.keys():
                getattr(self, setting)[param] = getattr(self, setting)[copy_from]
                
        settings_set = self._settings.values()
        self.plt_settings = {}
        for param in self._params+[*self._copy_params.keys()]:
            self.plt_settings[param] = {}
            for setting, pyplot_setting in self._settings.items():
                try:
                    self.plt_settings[param][pyplot_setting] = getattr(self, setting)[param]
                except KeyError:
                    self.plt_settings[param][pyplot_setting] = getattr(self, self.kind)[pyplot_setting]
            
            for setting, value in getattr(self, self.kind).items():
                if setting not in settings_set:
                    self.plt_settings[param][setting] = value

    def copy_from(self, other: "_BaseDefaultPltSettings", parameters: dict[str, list]):
        """Add full description later. 'copy()' does not overwrite settings that are already set.

        :param other: An instance of a class that inherits from '_BaseDefaultPltSettings'
        :type other: _BaseDefaultPltSettings
        :param parameters: Dictionary of the form {"parameter": [setting1, setting2, setting3, ...]}. Specifies
        which settings are copied for which parameters. Sets the settings for 'self' as defined in 'other'.
        :type parameters: dict[str, list]
        :raises KeyError: If 'other' does not specify a wanted parameter=setting combination.
        """
        for param, settings in parameters.items():
            for setting in settings:
                if setting in self.plt_settings[param].keys():
                    continue
                try:
                    self.plt_settings[param][setting] = other[param][other._settings[setting]]
                except KeyError:
                    raise KeyError(f"Instance of class '{type(other).__name__}' does not have a parameter "
                                   f"'{param}' or a setting '{setting}' for that parameter.")
                
    def add_params(self, **kwargs: dict[dict]):
        """Add parameters to the settings. Each kwarg must be a dictionary. The kwarg's name should be the parameter
        name. The dictionary holds the plot settings with pyplot.plot(**kwargs) kwargs as keys and the wanted value
        as value. Settings that are used in DefaultsPlots but that are not specified get the "_dfl" (default) value.
        These are saved in the class attributes.
        Example:
        >>> alpha_eff_plot_settings = {"label": r"$\alpha_{\text{eff}}$", "color"="red", lw=4}
        >>> plt_settings = DefaultsPlots()
        >>> plt_settings.add_params(alpha_eff=alpha_eff_plot_settings)
        >>> pyplot.plot(time, alpha_eff, **plt_settings.plt_settings)
        This should only be a convenience function of seldom use; these default classes are meant to have all plot 
        parameters defined already.
        """
        for param, defined_settigns in kwargs.items():
            self.plt_settings[param] = defined_settigns
            skip_setting = defined_settigns.keys()
            for setting, value in getattr(self, self.kind).items():
                if setting in skip_setting:
                    continue
                self.plt_settings[param][setting] = value
    
    def copy(self):
        return copy(self)

    @property
    @abstractmethod
    def _params():
        pass

    @property
    @abstractmethod
    def _copy_params():
        pass

    @property
    @abstractmethod
    def _settings():
        pass
    
    def __getitem__(self, key: str):
        return self.plt_settings[key]


class _CombineDefaults(_BaseDefaultPltSettings):
    def __init__(self, *args: tuple[_BaseDefaultPltSettings]|tuple[_BaseDefaultPltSettings, str]) -> None:
        self.plt_settings = {}
        args = tuple([arg if isinstance(arg, tuple) else (arg, "") for arg in args ])
        for part_of_settings, additional_name in args:
            params_set = self.plt_settings.keys()
            for param, settings in part_of_settings.plt_settings.items():
                new_param = additional_name+param
                if new_param in params_set:
                    raise ValueError(f"Settings for the paramter '{new_param}' are tried to be set multiple times.")
                self.plt_settings[new_param] = settings


class _Axes(_BaseDefaultPltSettings):
    _copy_params = {}

    _settings = {  # which axes.plot() settings are implemented; maps class attributes to plot() kwargs
        "_colours": "color",
        "_labels": "label"
    }

    _params = [  # parameters that are supported by default
        "x",
        "y",
        # "tors"
    ]

    copy_from = {}

    _colours = {  # line colour
        "x": "darkgreen",
        "y": "orangered",
        # "tors": "royalblue",
    }

    _labels = {  # line label
        "_dfl": None,
        "x": r"$x$",
        "y": r"$y$",
        # "tors": r"$tors$"
    }

    def __init__(self) -> None:
        _BaseDefaultPltSettings.__init__(self, "line")


class _DefaultArrow(_BaseDefaultPltSettings):
    _settings = {  # which axes.arrow() settings are implemented; maps class attributes to arrow()
        # kwargs
        "_width": "width",
        "_colour": "color"
    }

    _params = [
        "lift", 
        "drag"
    ]

    _copy_params = {}

    _width = {
        "lift": 0.01,
        "drag": 0.01
    }

    _colour = {
        "lift": None,
        "drag": None
    }

    def __init__(self) -> None:
        _BaseDefaultPltSettings.__init__(self, "arrow")


class _DefaultAngleOfAttack(_BaseDefaultPltSettings):
    _copy_params = {}

    _settings = {  # which axes.plot() settings are implemented; maps class attributes to plot() kwargs
        "_colours": "color",
        "_labels": "label"
    }

    _params = [  # parameters that are supported by default
        "alpha_steady",
        "alpha_qs",
        "alpha_eff",
        "alpha_sEq",    
        "alpha_eq",
    ]

    copy_from = {}

    _colours = {  # line colour
        "alpha_steady": "black",
        "alpha_qs": "darkgreen",
        "alpha_eff": "orangered",
        "alpha_sEq": "royalblue",
        "alpha_eq": "royalblue",
    }

    _labels = {  # line label
        "_dfl": None,
        "alpha_steady": r"$\alpha_{\text{steady}}$",
        "alpha_qs": r"$\alpha_{\text{qs}}$",
        "alpha_eff": r"$\alpha_{\text{eff}}$",
        "alpha_sEq": r"$\alpha_{\text{sEq}}$",
        "alpha_eq": r"$\alpha_{\text{eq}}$",
    }

    def __init__(self) -> None:
        _BaseDefaultPltSettings.__init__(self, "line")


class _DefaultForce(_BaseDefaultPltSettings):    
    _settings = {  # which axes.plot() settings are implemented; maps class attributes to plot() kwargs
        "_colours": "color",
        "_labels": "label",
    }

    _params = [  # parameters that are supported by default
        "lift",
        "drag",
        "mom",

        "edge",
        "flap",
        "tors",
    ]

    _copy_params =  {  # default mapping from additional parameters (keys) that have the same settings as 
        # a base default parameter of '_params' (value)
        "aero_drag": "drag",
        "aero_lift": "lift",
        "aero_mom": "mom",
        "aero_x": "drag",
        "aero_y": "lift",
        "damp_x": "edge",
        "damp_y": "flap",
        "damp_tors": "tors",
        "stiff_x": "edge",
        "stiff_y": "flap",
        "stiff_tors": "tors",
    } | {f"{spec}_{direction}": direction for direction in ["edge", "flap", "tors"] 
         for spec in ["damp", "stiff", "kin", "pot", "aero"]} 

    _colours = {  # line colour
        "lift": "orangered",
        "drag": "darkgreen",
        "mom": "mediumpurple",

        "edge": "forestgreen",
        "flap": "coral",
        "tors": "royalblue",
    }

    _labels = {  # line label
        "lift": "lift",
        "drag": "drag",
        "mom": "moment",

        "edge": "edge",
        "flap": "flap",
        "tors": "torsion"
    }

    def __init__(self) -> None:
        _BaseDefaultPltSettings.__init__(self, "line")


class _DefaultProfile(_BaseDefaultPltSettings):
    _copy_params = {}
    
    _settings = {  # which axes.plot() settings are implemented; maps class attributes to plot() kwargs
        "_colours": "color",
        "_labels": "label",
        "_markers": "marker",
        "_linestyles": "ls",
        "_marker_size": "ms"
    }

    _params = [  # parameters that are supported by default
        "profile",
        "qc_trail"
    ]

    _colours = {  # line colour
        "profile": "black",
        "qc_trail": "gray"
    }

    _labels = {  # line label
        "profile": "_",
        "qc_trail": "qc"
    }

    _markers = {  # line marker
        "qc_trail": "x"
    }

    _linestyles = {  # line style
        "qc_trail": "",
    }

    _marker_size = {  # marker size
        "qc_trail": 1
    }

    def __init__(self) -> None:
        _BaseDefaultPltSettings.__init__(self, "line")


class _DefaultEnergy(_BaseDefaultPltSettings):
    _copy_params = {}
        
    _settings = {  # which axes.plot() settings are implemented; maps class attributes to plot() kwargs
        "_colours": "color",
        "_labels": "label"
    }

    _params = [  # parameters that are supported by default
        "e_kin",
        "e_pot",
        "e_total"
    ]

    _colours = {  # line colour
        "e_kin": "blue",
        "e_pot": "green",
        "e_total": "black"
    }

    _labels = {  # line label
        "e_kin": r"$E_\text{kin}$",
        "e_pot": r"$E_\text{pot}$",
        "e_total": r"$E_\text{total}$"
    }

    def __init__(self) -> None:
        _BaseDefaultPltSettings.__init__(self, "line")

    
class _DefaultBL(_BaseDefaultPltSettings):
    _copy_params =  {}

    _settings = {  # which axes.plot() settings are implemented; maps class attributes to plot() kwargs
        "_colours": "color",
        "_labels": "label",
        "_linestyles": "ls"
    }

    _prep_params = [  # parameters for the preparation validation
        "f_n_aerohor",
        "f_n_section",
        "f_t_aerohor",
        "f_t_section",
        "C_l_rec_aerohor",
        "C_l_rec_section",
        "C_d_rec_aerohor",
        "C_d_rec_section",
    ]
    _sim_params = [
        "C_nc",
        "C_ni",
        "C_npot",
        "C_lpot",
        "C_tpot",
        "C_nsEq",
        "C_nf",
        "C_tf",
        "C_nv_instant",
        "C_nv",
        "C_mqs",
        "C_mnc",
        "f_n",
        "f_t",
        "C_l_rec",
        "C_d_rec",
        "C_lc", 
        "C_lnc", 
        "C_ds",   
        "C_dc", 
        "C_dsep", 
        "C_ms",
        "C_liner",
        "C_lcent",
        "C_dind",
        "C_lift",
        "C_miner",
        "C_dus",
        "C_lus",
        "C_mus",
        "C_nvisc",
        "C_mf",
        "C_mV",
        "C_mC",
    ]
    _params = _prep_params + _sim_params

    _c_models = {"aerohor": "forestgreen", "section": "orangered"}
    def _prep_colours(params, colours):  # this is a bit narly; it replaces a dict comprehension that wouldn't work here
        return {param: colours[param.split("_")[-1]] for param in params}
    _prep_colours = _prep_colours(_prep_params, _c_models)
    _c_rest = {
        "C_nc": "blue",
        "C_ni": "red",
        "C_npot": "green",
        "C_lpot": "green",
        "C_nsEq": "orange",
        "C_nf": "blue",
        "C_nv_instant": "red",
        "C_nv": "green",
        "C_tpot": "orange",
        "C_lpot": "orange",
        "C_tf": "green",
        "C_mqs": "black",
        "C_mnc": "orange",
        "f_n": "forestgreen",
        "f_t": "mediumblue",
        "C_l_rec": "forestgreen", 
        "C_d_rec": "mediumblue", 
        "C_lc": "blue", 
        "C_lnc": "red", 
        "C_ds": "green",
        "C_dc": "orange", 
        "C_dsep": "blue", 
        "C_ms": "red",
        "C_liner": "blue", 
        "C_lcent": "red", 
        "C_dind": "green",
        "C_lift": "orange", 
        "C_miner": "blue", 
        "C_dus": "black",
        "C_lus": "black",
        "C_mus": "black",
        "C_nvisc": "black",
        "C_mf": "black",
        "C_mV": "black",
        "C_mC": "black",
    }
    _colours = _prep_colours|_c_rest

    _labels = {  # line label
        "f_n_aerohor": r"aerohor $f_n$", 
        "f_n_section": r"section $f_n$", 
        "f_t_aerohor": r"aerohor $f_t$", 
        "f_t_section": r"section $f_t$", 
        "C_l_rec_aerohor": r"aerohor reconstructed $C_l$", 
        "C_l_rec_section": r"section reconstructed $C_l$", 
        "C_d_rec_aerohor": r"aerohor reconstructed $C_d$", 
        "C_d_rec_section": r"section reconstructed $C_d$",
        "C_l_rec": r"reconstructed $C_l$", 
        "C_d_rec": r"reconstructed $C_d$", 
        "C_nc": r"$C_{n\text{,c}}$",
        "C_ni": r"$C_{n\text{,i}}$",
        "C_npot": r"$C_{n\text{,p}}$",
        "C_lpot": r"$C_{l\text{,p}}$",
        "C_tpot": r"$C_{t\text{,c}}$",
        "C_nsEq": r"$C_{n\text{,seq}}$",
        "C_nf": r"$C_{n\text{,f}}$",
        "C_tf": r"$C_{t\text{,f}}$",
        "C_nv_instant": r"$C_{n\text{,vi}}$",
        "C_nv": r"$C_{n\text{,v}}$",
        "C_mqs": r"$C_{m\text{,qs}}$",
        "C_mnc": r"$C_{m\text{,nc}}$",
        "f_n": r"$f_n$",
        "f_t": r"$f_t$",
        "C_lc": "$C_{lc}$",
        "C_lnc": "$C_{lnc}$",
        "C_ds": "$C_{ds}$",
        "C_dc": "$C_{dc}$",
        "C_dsep": "$C_{dsep}$",
        "C_ms": "$C_{ms}$",
        "C_liner": r"$C_{l\text{,iner}}$",
        "C_lcent": r"$C_{l\text{,cent}}$",
        "C_dind": r"$C_{d\text{,ind}}$",
        "C_lift": r"$C_{m\text{,us}}$",
        "C_miner": r"$C_{m\text{,iner}}$",
        "C_dus": r"$C_{d\text{,us}}$",
        "C_lus": r"$C_{l\text{,us}}$",
        "C_mus": r"$C_{m\text{,us}}$",
        "C_nvisc": r"$C_{n\text{,visc}}$",
        "C_mf": r"$C_{m\text{,f}}$",
        "C_mV": r"$C_{m\text{,V}}$",
        "C_mC": r"$C_{m\text{,C}}$",
    }

    _linestyles = {  # line style
        "f_n_section": "--",
        "f_t_section": "--",
        "C_l_rec_section": "--",
        "C_d_rec_section": "--",
        "C_nc": "--",
        "C_ni": "--",
        "C_npot": "--",
        "C_lpot": "--",
        "C_dus": "--",
        "C_lus": "--",
        "C_mus": "--",
    }

    def __init__(self) -> None:
        _BaseDefaultPltSettings.__init__(self, "line")


class _DefaultMeasurement(_BaseDefaultPltSettings):
    _copy_params = {}

    _settings = {  # which axes.plot() settings are implemented; maps class attributes to plot() kwargs
        "_labels": "label",
        "_markers": "marker",
        "_linestyles": "ls",
        "_marker_size": "ms"
    }

    _params = [ 
        "C_l",
        "C_d",
        "C_l_specify",
        "C_d_specify",
        "C_m",
        "C_l_HAWC2",
        "C_d_HAWC2",
        "C_m_HAWC2",
        "C_l_openFAST",
        "C_d_openFAST",
        "C_m_openFAST",
        "C_l_measurement",
        "C_d_measurement",
        "C_m_measurement",
        "C_l_IAG",
        "C_d_IAG",
        "C_m_IAG",
    ]

    _labels = {
        "C_l": "steady",
        "C_d": "steady",
        "C_l_specify": r"$C_l$ steady",
        "C_d_specify": r"$C_d$ steady",
        "C_m": "steady",
        "C_l_HAWC2": "HAWC2",
        "C_d_HAWC2": "HAWC2",
        "C_m_HAWC2": "HAWC2",
        "C_l_openFAST": "openFAST",
        "C_d_openFAST": "openFAST",
        "C_m_openFAST": "openFAST",
        "C_l_IAG": "IAG",
        "C_d_IAG": "IAG",
        "C_m_IAG": "IAG",
        "C_l_measurement": "measurement",
        "C_d_measurement": "measurement",
        "C_m_measurement": "measurement",
    }

    _markers = {
        "C_l": "o",
        "C_d": "o",
        "C_l_specify": "x",
        "C_d_specify": "+",
        "C_m": "o",
        "C_l_HAWC2": "+",
        "C_d_HAWC2": "+",
        "C_m_HAWC2": "+",
        "C_l_openFAST": "x",
        "C_d_openFAST": "x",
        "C_m_openFAST": "x",
        "C_l_IAG": "x",
        "C_d_IAG": "x",
        "C_m_IAG": "x",
        "C_l_measurement": "+",
        "C_d_measurement": "+",
        "C_m_measurement": "+",
    }

    _marker_size = {
        "C_l_HAWC2": 0.8,
        "C_d_HAWC2": 0.8,
        "C_m_HAWC2": 0.8,
    }
    
    def _linestyles(params):
        return {param: "" for param in params}
    _linestyles = _linestyles(_params)
    _linestyles["C_l"] = "-"
    _linestyles["C_d"] = "-"
    _linestyles["C_m"] = "-"

    def __init__(self) -> None:
        _BaseDefaultPltSettings.__init__(self, "line")


class _DefaultGeneral(_BaseDefaultPltSettings):
    _copy_params = {}
        
    _settings = {  # which axes.plot() settings are implemented; maps class attributes to plot() kwargs
        "_colours": "color",
        "_labels": "label"
    }

    _params = [  # parameters that are supported by default
        "HAWC2",
        "openFAST",
        "section",
        "section_staeblein",
    ]

    _colours = {  # line colour
        "HAWC2": "forestgreen",
        "openFAST": "coral",
        "section": "royalblue",
        "section_staeblein": "royalblue"
    }

    _labels = {  # line label
        "HAWC2": "HAWC2",
        "openFAST": "openFAST",
        "section": "section",
        "section_staeblein": "section, Stäblein"
    }
    
    def __init__(self) -> None:
        _BaseDefaultPltSettings.__init__(self, "line")


class DefaultPlot:
    """Class to specify plot settings for parameters of this project. Example:
    Assume time, lift given.
    >>> dfl = DefaultsPlots()
    >>> fig, ax = plt.subplots()
    >>> dfl = DefaultsPlots()
    >>> matplotlib.pyplot.plot(time, lift, **dfl.settings["lift"])
    This will cause the plot to have all axes settings defined in this class to be set for the values connected 
    to "lift.
    """
    plt_settings = {}
    def __init__(self) -> None:
        axes = _Axes()
        arrow = _DefaultArrow()
        aoa = _DefaultAngleOfAttack()
        force = _DefaultForce()
        profile = _DefaultProfile()
        energy = _DefaultEnergy()
        BL = _DefaultBL()
        measurement = _DefaultMeasurement()
        general = _DefaultGeneral()
        _CombineDefaults.__init__(self, axes, (arrow, "arrow_"), aoa, force, profile, energy, BL, measurement, general)
        

class DefaultStructure:
    """C_lass that maps parameters of a certain kind to the filename they should be saved into. The keys must not
    be changed other than new keys added. Change the values if new names are required.
    """
    _dfl_filenames = {
        "f_aero": "f_aero.dat",
        "f_structural": "f_structural.dat",
        "general": "general.dat",
        "section_data": "section_data.json" ,
        "power": "power.dat",
        "e_kin": "e_kin.dat",
        "e_pot": "e_pot.dat",
        "aoa_threshold": "aoa_thresholds.dat",
        "period_work": "period_work.dat",
        "peaks": "peaks.json"
    }


class DefaultsSimulation(DefaultStructure):
    """C_lass that is used for class SimulationResults.
    """
    _dfl_params = [
        # parameters that are automatically created as property of SimulationResults
        "alpha_steady",
        ("pos", 3),
        ("vel", 3),
        ("accel", 3),
        ("aero", 3),
        ("damp", 3),
        ("stiff", 3),
    ]

    _dfl_files = {
        # {file_name: [params to be saved]} as saved by SimulationResults._save
        DefaultStructure._dfl_filenames["general"]: ["time", "inflow", "pos", "vel", "accel"],
        DefaultStructure._dfl_filenames["f_aero"]: ["alpha_steady", "aero"],
        DefaultStructure._dfl_filenames["f_structural"]: ["damp", "stiff"]
    }

    _dfl_split = {
        # if _dfl_params are multidimensional arrays, what is the name of each column of the nump array
        # these splits rely on ThreeDOFsAirfoil running its simulation in the x-y-z coordinate system
        "inflow": ["x", "y"],
        "pos": ["x", "y", "tors"],
        "vel": ["x", "y", "tors"],
        "accel": ["x", "y", "tors"],
        "aero": ["x", "y", "mom"],
        "damp": ["x", "y", "tors"],
        "stiff": ["x", "y", "tors"],
        # the following are for post caluclations
        "aero_projected_dl": ["drag", "lift"],
        "aero_projected_ef": ["edge", "flap"],
        "damp_projected": ["edge", "flap"],
        "stiff_projected": ["edge", "flap"],
        "pos_projected": ["edge", "flap"],
        "vel_projected_xy": ["edge_xy", "flap_xy"],
        "vel_projected_ef": ["edge_ef", "flap_ef"],
        "accel_projected": ["edge", "flap"],
    }

    def _get_split(self, param: str, without_param: bool=False):
        param = "" if without_param else param
        return [param+"_"+split for split in self._dfl_split[param]]


class Staeblein:
    def __init__(self) -> None:
        # aero
        self.C_l_alpha = 7.15
        self.C_d = 0.01
        self.C_m = -0.1

        # structure from Stäblein paper
        self.c = 3.292
        self.e_ac = 0.113
        self.e_cg = 0.304
        self.r = 0.785 

        # self.e_ac = 0.3283  # from DTU 10 MW RWT description

        self.m = 203

        self.nat_freqs = np.asarray([0.93, 0.61, 6.66])*2*np.pi
        self.damp_ratios = np.asarray([0.0049, 0.0047, 0.0093])
        # self.nat_freqs = np.asarray([0.93, 0.61, 5.69])*2*np.pi
        # self.damp_ratios = np.asarray([0.0049, 0.0047, 0.0331])

        self.inertia = np.asarray([self.m, self.m, self.m*(self.e_cg**2+self.r**2)])
        lin_stiff = np.asarray(self.nat_freqs[:2])**2*self.inertia[:2]
        tors_stiff = self.m*self.r**2*self.nat_freqs[2]**2
        self.stiffness = np.r_[lin_stiff, tors_stiff] 
        self.damping = 2*self.damp_ratios*self.inertia*self.nat_freqs

