import matplotlib.pyplot as plt

plt.rcParams.update({"font.size": 6})

class DefaultsPlots:
    """Class to specify plot settings for parameters of this project. Example:
    Assume time, lift given.
    >>> dfl = DefaultsPlots()
    >>> fig, ax = plt.subplots()
    >>> dfl = DefaultsPlots()
    >>> matplotlib.pyplot.plot(time, lift, **dfl.settings["lift"])
    This will cause the plot to have all axes settings defined in this class to be set for the values connected 
    to "lift.
    """
    _settings_implemented = {  # which axes settings are implemented; maps class attributes to plot() kwargs
        "_colours": "color",
        "_labels": "label",
        "_markers": "marker",
        "_linestyles": "linestyle",
        "_ms": "ms",
        "_linewidths": "lw"
    }

    _params = [  # parameters that are supported by default
        "alpha_steady",
        "alpha_qs",
        "lift",
        "drag",
        "mom",
        "edge",
        "flap",
        "tors",
        "profile",
        "qc_trail",
        "e_kin",
        "e_pot",
        "e_total"
    ]

    _params_copy =  {  # default mapping from additional parameters (keys) that have the same settings as 
        # a base default parameter of '_params' (value)
        "aero_drag": "drag",
        "aero_lift": "lift",
        "aero_mom": "mom"
    }

    _colours = {  # line colour
        "_dfl": "black",
        "alpha_steady": "black",
        "alpha_qs": "darkgreen",
        "lift": "orangered",
        "drag": "darkgreen",
        "mom": "mediumpurple",
        "edge": "forestgreen",
        "flap": "coral",
        "tors": "royalblue",
        "profile": "black",
        "qc_trail": "gray",
        "e_kin": "blue",
        "e_pot": "green",
        "e_total": "black"
    }

    _labels = {  # line label
        "_dfl": None,
        "alpha_steady": r"$\alpha_{\text{steady}}$",
        "alpha_qs": r"$\alpha_{\text{qs}}$",
        "lift": "lift",
        "drag": "drag",
        "mom": "mom",
        "edge": "edge",
        "flap": "flap",
        "tors": "torsion",
        "profile": "_",
        "qc_trail": "qc",
        "e_kin": r"$E_\text{kin}$",
        "e_pot": r"$E_\text{pot}$",
        "e_total": r"$E_\text{total}$"
    }

    _markers = {  # line marker
        "_dfl": None,
        "alpha_steady": None,
        "alpha_qs": None,
        "lift": None,
        "drag": None,
        "mom": None,
        "edge": None,
        "flap": None,
        "tors": None,
        "profile": None,
        "qc_trail": "x",
        "e_kin": None,
        "e_pot": None,
        "e_total": None
    }

    _linestyles = {  # line style
        "_dfl": None,
        "alpha_steady": None,
        "alpha_qs": None,
        "lift": None,
        "drag": None,
        "mom": None,
        "edge": None,
        "flap": None,
        "tors": None,
        "profile": None,
        "qc_trail": "",
        "e_kin": None,
        "e_pot": None,
        "e_total": None
    }

    _linewidths = {  # line width
        param: 1 for param in ["_dfl"]+_params
    }

    _ms = {  # marker size
        "_dfl": None,
        "alpha_steady": None,
        "alpha_qs": None,
        "lift": None,
        "drag": None,
        "mom": None,
        "edge": None,
        "flap": None,
        "tors": None,
        "profile": None,
        "qc_trail": 1,
        "e_kin": None,
        "e_pot": None,
        "e_total": None
    }

    def __init__(self) -> None:
        # add copy parameters to class attributes so they can be accessed in the definition of self.plt_settings
        for param, copy_from in self._params_copy.items():
            for setting in self._settings_implemented.keys():
                getattr(self, setting)[param] = getattr(self, setting)[copy_from]
        
        # initialise the settings; now plt.settings[param] holds all settings for that param
        self.plt_settings = {}
        for param in self._params+[*self._params_copy.keys()]:
            self.plt_settings[param] = {}
            for setting, pyplot_setting in self._settings_implemented.items():
                self.plt_settings[param][pyplot_setting] = getattr(self, setting)[param]
                
    def add_params(self, **kwargs: dict[dict]):
        """Add parameters to the settings. Each kwarg must be a dictionary. The kwarg's name should be the parameter
        name. The dictionary holds the plot settings with pyplot.plot(**kwargs) kwargs as keys and the wanted value
        as value. Settings that are used in DefaultsPlots but that are not specified get the "_dfl" (default) value.
        These are saved in the class attributes.
        Example:
        >>> alpha_eff_plot_settings = {"label": r"$\alpha_{\text{eff}}$", "color"="red", lw=4}
        >>> plt_settings = DefaultsPlots()
        >>> plt_settings.add_params(alpha_eff = alpha_eff_plot_settings)
        >>> pyplot.plot(time, alpha_eff, **plt_settings.plt_settings)
        """
        map_plt_to_attr = {plt_name: attr for attr, plt_name in self._settings_implemented.items()}
        for param, user_def_settings in kwargs.items():
            self.plt_settings[param] = user_def_settings
            skip_setting = user_def_settings.keys()
            for setting in self._settings_implemented.values():
                if setting in skip_setting:
                    continue
                self.plt_settings[param][setting] = getattr(self, map_plt_to_attr[setting])["_dfl"]

    def _update_colours(self, _colours: dict):
        self._update(_colours, "color")

    def _update(self, to_update: dict, setting: str):
        for param, value_setting in to_update.items():
            self.plt_settings[param][setting] = value_setting


class DefaultStructure:
    """Class that maps parameters of a certain kind to the filename they should be saved into. The keys must not
    be changed other than new keys added. Change the values if new names are required.
    """
    _dfl_filenames = {
        "f_aero": "f_aero.dat",
        "f_structural": "f_structural.dat",
        "general": "general.dat",
        "section_data": "section_data.json" ,
        "work": "work.dat",
        "e_kin": "e_kin.dat",
        "e_pot": "e_pot.dat",
    }


class DefaultsSimulation(DefaultStructure):
    """Class that is used for class SimulationResults.
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
        "aero_projected": ["drag", "lift"],
        "damp_projected": ["edge", "flap"],
        "stiff_projected": ["edge", "flap"],
        "pos_projected": ["edge", "flap"],
        "vel_projected": ["edge", "flap"]
    }

    def _get_split(self, param: str, without_param: bool=False):
        param = "" if without_param else param
        return [param+"_"+split for split in self._dfl_split[param]]
