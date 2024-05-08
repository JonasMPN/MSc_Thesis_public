import numpy as np
from scipy.linalg import solve
import pandas as pd
from os.path import join, isdir
from typing import Callable
import json
from helper_functions import Helper
from defaults import DefaultsSimulation
from scipy import interpolate
helper = Helper()


class SimulationResults(DefaultsSimulation):
    """Instances of this class are used as data storage variables during the airfoil simulation. Parameters are stored
    as attributes of the instance. Attributes can be dynamically added using "add_param()". Each instance comes with a
     set of default parameters; these are defined in the parent class DefaultsSimulation. All parameter attributes are initialised as numpy arrays with as many values as there are timesteps in the simulation. 

    :param DefaultsSimulation: Class defining default settings for the simulation parameters.
    """
    def __init__(self, time: np.ndarray) -> None:
        """Initialise an instance. The instance has a set of default parameters as defined in the parent class 
        DefaultsSimulation.

        :param time: Time array that is used for the simulation.
        :type time: np.ndarray
        """
        DefaultsSimulation.__init__(self)
        self._n_time_steps = time.size
        self.time = time
        self.add_param(*self._dfl_params)  # _dfl_params comes from DefaultsSimulation
    
    def set_param(self, **kwargs):
        """Adds an instance attribute with the name of the kwarg and the value of the kwarg.
        """
        for name, value in kwargs.items():
            vars(self)[name] = value

    def add_param(self, *args: str | tuple[str, int], init_as=np.zeros):
        """Add instance attributes. Each attribute is initialised to hold a value for each time step. 

        :param init_as: _description_, defaults to np.zeros
        :type init_as: _type_, optional
        """
        args = [(arg, 1) if isinstance(arg, str) else arg for arg in args]
        for name, n_params in args:
            vars(self)[name] = init_as((self._n_time_steps, n_params)).squeeze()
    
    def _save(
        self, 
        root: str, 
        files: dict[str, list[str]]=None, 
        split: dict[str, list[str]]=None, 
        use_default: bool=True):
        """Saves instance attributes to CSV .dat files.

        :param root: Root directory in relation to the current working directory into which the files are saved.
        :type root: str
        :param files: Sets the name of the file (key) and the instance attributes saved into it (values), defaults to 
        None. If "use_default=True", the default parameters from DefaultsSimulation._dfl_files are always saved.
        :type files: dict[str: list[str]], optional
        :param split: Instance attributes that are multidimensional arrays should be split into each component. 
        "split" defines the appendices the names of the components get. Example: split={"vel": ["x", "y", "rot"]}
        will cause the attribute "vel" to be saved as three columns with names "vel_x", "vel_y", and "vel_z". If
        "use_default=True", the default splot from DefaultsSimulation._dfl_splot is always used. defaults to None
        :type split: dict[str: str], optional
        :param use_default: Whether or not to use the defaults for "files" and "split". If "True", the default 
        settings are added to the user-specified "files" and "split". To only use the user-specified "files" and 
        "split", "use_default=False" has to be set. defaults to True
        :type use_default: bool, optional
        """
        files = files if files is not None else {}
        split = split if split is not None else {}
        user_defined_split = split.keys()
        user_defined_files = files.keys()

        if use_default:
            for param, split_into in self._dfl_split.items():
                if param not in user_defined_split:
                    split[param] = split_into
                    
            for filename, params in self._dfl_files.items():
                if filename in user_defined_files:
                    if files[filename] is not None:
                        params = list(set(params+files[filename]))
                    files[filename] = params
                else:
                    files[filename] = self._dfl_files[filename]
                    
        for file, params in files.items():
            df = pd.DataFrame()
            for param in params:
                if param in split.keys():
                    for i, split_name in enumerate(split[param]):
                        try:
                            df[param+"_"+split_name] = vars(self)[param][:, i]
                        except ValueError:
                            raise ValueError(f"The above ValueError was caused for parameter {param}.")
                else:
                    df[param] = vars(self)[param]
            df.to_csv(join(root, file), index=False)


class ThreeDOFsAirfoil(SimulationResults):
    """Class that can simulate the aeroelastic behaviour of an airfoil. There exist different options for the schemes
    used to calculate the aerodynamic forces (see class AeroForce), the structural forces (see StructForce), and how 
    the time integration for the next time step is performed (see class TimeIntegration).

    Some parameters during the simulation are automatically saved. These are defined in the class DefaultsSimulation, 
    which is a parent class to SimulationResults, from which ThreeDOFsAirfoil inherits functionalities. 
    """
    _sim_assert = {
        "_aero_force": "set_aero_calc", 
        "_struct_force": "set_struct_calc",
        "_time_integration": "set_time_integration"
        }

    def __init__(
        self, 
        dir_polar: str,
        time: np.ndarray, 
        mass: float,
        mom_inertia: float,
        chord: float, 
        stiffness_edge: float, 
        stiffness_flap: float,
        stiffness_tors: float,
        damping_edge: float, 
        damping_flap: float,
        damping_tors: float,
        file_polar: str="polars.dat"
        ) -> None:
        """Initialises instance object.

        :param file_polar_data: Path to the file containing polar data for the airfoil. The file must contain the
        columns ["alpha", "C_d", "C_l", "C_m"].
        :type file_polar_data: str
        :param time: Numpy array of the time series of the time used for the simulation.
        :type time: np.ndarray
        :param mass: Mass of the airfoil in kg.
        :type mass: float
        :param mom_inertia: Moment of inertia of the airfoil in kg.m^2
        :type mom_inertia: float
        :param chord: Chord in meters
        :type chord: float
        :param stiffness_edge: Edgewise spring stiffness in N/m
        :type stiffness_edge: float
        :param stiffness_flap: Flapwise spring stiffness in N/m
        :type stiffness_flap: float
        :param stiffness_tors: Torsional spring stiffness in N/rad
        :type stiffness_tors: float
        :param damping_edge: Speed-proportional damping parameter in the edgewise direction in N/(m/s).
        :type damping_edge: float
        :param damping_flap: Speed-proportional damping parameter in the flapwise direction in N/(m/s).
        :type damping_flap: float
        :param damping_tors: Speed-proportional damping parameter in the rotational direction in N/(rad/s).
        :type damping_tors: float
        """
        SimulationResults.__init__(self, time)
        self.dir_polar = dir_polar
        self.file_polar_data = file_polar

        self.mass = mass
        self.mom_inertia = mom_inertia
        self.chord = chord

        self._struct_params = {
            "stiffness_edge": stiffness_edge,
            "stiffness_flap": stiffness_flap,
            "stiffness_tors": stiffness_tors,
            "damping_edge": damping_edge,
            "damping_flap": damping_flap,
            "damping_tors": damping_tors
            }

        self._aero_force = None
        self._struct_force = None
        self._time_integration = None

        self._init_aero_force = None
        self._init_struct_force = None
        self._init_time_integration = None

        self._aero_scheme_settings = None
        self._struct_scheme_settings = None
        self._time_integration_scheme_settings = None

        self._added_sim_params = {filename: None for filename in self._dfl_files.keys()}
  
    def set_aero_calc(self, scheme: str="quasi_steady", **kwargs):
        """Sets how the aerodynamic forces are calculated in the simulation. If the scheme is dependet on constants,
        these must be given in the kwargs. Which kwargs are necessary are defined in the match-case statements in 
        this function's implementation; they are denoted "must_haves". These are mandatory. The "can_haves" are 
        constants for which default values are set. They (the "can_haves") can, but do not have to, thus be set.

        :param scheme: Existing schemes are "steady", "quasi_steady", and "BL". See details in class AeroForce., defaults to "quasi_steady"
        :type scheme: str, optional
        :raises NotImplementedError: If a scheme is wanted that is not implemented.
        """
        params_given = kwargs.keys()
        self._aero_scheme_settings = {}
        match scheme:
            case "steady":
                pass
            case "quasi_steady":
                can_haves = ["pitching_around", "alpha_at"]
                self._aero_scheme_settings = {param: value for param, value in kwargs.items() if param in can_haves}
                add_aero_sim_params = ["alpha_qs"]
            case "BL":
                must_haves = ["A1", "A2", "b1", "b2", "pitching_around", "alpha_at"]
                can_haves = ["a", "K_alpha", "T_p", "T_bl", "Cn_vortex_detach", "tau_vortex_pure_decay", "T_v"]
                self._check_scheme_settings("BL", "set_aero_calc", must_haves, params_given)
                self._aero_scheme_settings = {param: value for param, value in kwargs.items() if param in 
                                              must_haves+can_haves}
                add_aero_sim_params = ["s", "alpha_qs", "X_lag", "Y_lag", "alpha_eff", "C_nc", "d_alpha_qs_dt", "D_i", 
                                       "C_ni", "C_npot", "C_tpot", "D_p", "C_nsEq", "alpha_sEq", "f_t_Dp", "f_n_Dp", 
                                       "D_bl_t", "D_bl_n", "f_t", "f_n", "C_nf", "C_tf", "tau_vortex", "C_nv_instant",
                                       "C_nv"]
        # add parameters the AeroForce method needs to calculate the aero forces or that are parameters the method
        # calculates that are interesting to save.
        self._added_sim_params[self._dfl_filenames["f_aero"]] = add_aero_sim_params
        for param in add_aero_sim_params:
            self.add_param(param)  # adds instance attribute to self
        AF = AeroForce(dir_polar=self.dir_polar, file_polar=self.file_polar_data)
        self._aero_force = AF.get_function(scheme)
        self._init_aero_force = AF.get_init_function(scheme)

    def set_struct_calc(self, scheme: str="linear", **kwargs):
        """Sets how the structural forces are calculated in the simulation. If the scheme is dependet on constants,
        these must be given in the kwargs. Which kwargs are necessary are defined in the match-case statements in 
        this function's implementation; they are denoted "must_haves". These are mandatory. The "can_haves" are 
        constants for which default values are set. They (the "can_haves") can, but do not have to, thus be set.

        :param scheme: Existing schemes are "linear". See details in class StructForce. defaults to "linear"
        :type scheme: str, optional
        :raises NotImplementedError: If a scheme is wanted that is not implemented.
        """
        match scheme:
            case "linear":
                self._struct_scheme_settings = {}
            case "nonlinear":
                # check if all necessary BL parameters are given in kwargs
                self._struct_scheme_settings = None
                raise NotImplementedError
        self._struct_force = StructForce(**self._struct_params).get_function(scheme)

    def set_time_integration(self, scheme: str="eE", **kwargs):
        """Sets how the time integration is done in the simulation. If the scheme is dependet on constants,
        these must be given in the kwargs. Which kwargs are necessary are defined in the match-case statements in 
        this function's implementation; they are denoted "must_haves". These are mandatory. The "can_haves" are 
        constants for which default values are set. They (the "can_haves") can, but do not have to, thus be set.

        :param scheme: Existing schemes are "eE", "RK4", and "HHT_alpha". See details in class TimeIntegration. 
        defaults to "EeE"
        :type scheme: str, optional
        :raises NotImplementedError: If a scheme is wanted that is not implemented.
        """
        params_given = kwargs.keys()
        match scheme:
            case "eE":
                self._time_integration_scheme_settings = {"dt": self.time[1]}
            case "RK4":
                raise NotImplementedError
            case "HHT-alpha":
                must_haves = ["alpha", "dt"]
                self._check_scheme_settings("HHT-alpha", "set_time_integration", must_haves, params_given)
                self._time_integration_scheme_settings = {param: value for param, value in kwargs.items() if param in 
                                                          must_haves}
        TI = TimeIntegration()
        self._time_integration = TI.get_time_step_function(scheme)
        self._init_time_integration = TI.get_init_function(scheme)

    def simuluate(
        self, 
        inflow: np.ndarray, 
        density: float, 
        init_position: np.ndarray,
        init_velocity: np.ndarray):
        """Performs the simulation. Before the simulation can be run, the methods "set_aero_calc()", 
        "set_struct_calc()", and "set_time_integration()" have to be used.

        :param inflow: The inflow at each time step with components in the x and y direction [x, y].
        :type inflow: np.ndarray
        :param density: Density of the fluid
        :type density: float
        :param init_position: Initial position of the quarter-chord for [x, y, rotation around z in rad]
        :type init_position: np.ndarray
        :param init_velocity: Initial velocity of the quarter-chord for [x, y, rotation around z in rad/s]
        :type init_velocity: np.ndarray
        """
        inflow_angle = np.arctan(inflow[:, 1]/inflow[:, 0])
        dt = self.time[1:]-self.time[:-1]
        self.set_param(inflow=inflow, inflow_angle=inflow_angle, density=density, dt=dt)
        # check whether all necessary settings have been set.
        self._check_simulation_readiness(_aero_force=self._aero_force, 
                                         _struct_force=self._struct_force,
                                         _time_integration=self._time_integration)
        self.pos[0, :] = init_position
        self.vel[0, :] = init_velocity

        init_funcs = [self._init_aero_force, self._init_struct_force, self._init_time_integration]
        scheme_settings = [self._aero_scheme_settings, self._struct_scheme_settings, 
                           self._time_integration_scheme_settings]
        for init_func, scheme_setting in zip(init_funcs, scheme_settings):
            if init_func is not None:
                init_func(self, **scheme_setting)
        for i in range(self.time.size-1):
            # get aerodynamic forces
            self.aero[i, :] = self._aero_force(self, i, **self._aero_scheme_settings)
            # get structural forces
            self.damp[i, :], self.stiff[i, :] = self._struct_force(self, i, **self._struct_scheme_settings)
            # performe time integration
            self.pos[i+1, :], self.vel[i+1, :], self.accel[i+1, :] = self._time_integration(self, i,
                                                                            **self._time_integration_scheme_settings)
        # i = self.time.size-1
        # # for the last time step, only the forces are calculated because there is no next time step for the positions
        # self.aero[i, :] = self._aero_force(self, i, **self._aero_scheme_settings)
        # self.damp[i, :], self.stiff[i, :] = self._struct_force(self, i, **self._struct_scheme_settings)

    def _check_simulation_readiness(self, **kwargs):
        """Utility function checking whether all settings given in kwargs have been set.
        """
        for property, value in kwargs.items():
            assert value, f"Method {self._sim_assert[property]}() must be used before simulation."

    def save(self, root: str, files: dict=None, split: dict=None, use_default: bool=True):
        """Wrapper for SimulationResults._save(). See more details there. Additionally to _save(), also saves the
        structural and section data.

        :param root: Root directory in relation to the current working directory into which the files are saved.
        :type root: str
        :param files: Sets the name of the file (key) and the instance attributes saved into it (values), defaults to 
        None. If "use_default=True", the default parameters from DefaultsSimulation._dfl_files are always saved.
        :type files: dict[str: list[str]], optional
        :param split: Instance attributes that are multidimensional arrays should be split into each component. 
        "split" defines the appendices the names of the components get. Example: split={"vel": ["x", "y", "rot"]}
        will cause the attribute "vel" to be saved as three columns with names "vel_x", "vel_y", and "vel_z". If
        "use_default=True", the default splot from DefaultsSimulation._dfl_splot is always used. defaults to None
        :type split: dict[str: str], optional
        :param use_default: Whether or not to use the defaults for "files" and "split". If "True", the default 
        settings are added to the user-specified "files" and "split". To only use the user-specified "files" and 
        "split", "use_default=False" has to be set. defaults to True
        :type use_default: bool, optional
        """
        helper.create_dir(root)
        files = files if files is not None else {filename: [] for filename in self._dfl_files.keys()}
        if use_default:
            for filename, params in self._added_sim_params.items():
                if params == None:
                    continue
                files[filename] = set(files[filename]+params)
        self._save(root, files|self._added_sim_params, split, use_default)
        section_data = {
            "mass": self.mass,
            "mom_inertia": self.mom_inertia,
            "chord": self.chord 
        } |self._struct_params
        with open(join(root, "section_data.json"), "w") as f:
            json.dump(section_data, f, indent=4)

    @staticmethod
    def _check_scheme_settings(scheme: str, fnc_call: str, must_haves: list, was_given: list):
        """Utility function checking whether all necessary scheme settings have been set.

        :param scheme: _description_
        :type scheme: str
        :param fnc_call: _description_
        :type fnc_call: str
        :param must_haves: _description_
        :type must_haves: list
        :param was_given: _description_
        :type was_given: list
        :raises ValueError: _description_
        """
        for must_have in must_haves:
            if must_have not in was_given:
                raise ValueError(f"Using {scheme} requires that {must_haves} are given in function call {fnc_call}, "
                                 f"but {must_have} is missing.")
    

class Rotations:
    """Class providing methods for anything related to performing rotations.
    """
    def project(self, array: np.ndarray, angles: int|float|np.ndarray) -> np.ndarray:
        """Projecst the points in "array" into a coordinate system rotated by the angles given in "angles". The 
        rotation is around the third component of "array", meaning it doesn't change that value. If "angles" is a 
        single value, all points are rotated by that angle. If "angles" has as many values as "array" has points, each 
        point is rotated by the angle in "angles" of the same index.

        :param array: (n, 3) numpy.ndarray
        :type array: np.ndarray
        :param angles: A single value or an array of values by which the points in "array" are rotated
        :type angles: np.int|float|np.ndarray
        :return: projected points
        :rtype: np.ndarray
        """
        n_rot = angles.size
        array = array.reshape((n_rot, 3, 1))
        rot_matrices = self.passive_3D_planar(angles)
        return (rot_matrices@array).reshape(n_rot, 3)
    
    def project_separate(self, array: np.ndarray, angles: np.ndarray) -> np.ndarray:
        """Same behaviour as "project()", except that the first two components of "array" are separately
        projected into the rotated coordinate system (that is given by "angles"). Example:
        array[i, :] = [drag, lift, moment] in an aerodynamic coordinate system that is rotated by "alpha"
        from the x-y-z coordinate system. then the output of "project_separate(array, alpha) is 
        [drag_x, drag_y, lift_x, lift_y, moment], i.e., the drag and lift projected onto the x and y axes.

        :param array: (n, 3) array
        :type array: np.ndarray
        :param angles: A single value or an array of values by which the points in "array" are rotated
        :type angles: np.int|float|np.ndarray
        :return: Projected points
        :rtype: np.ndarray
        """
        n_rot = angles.size
        array = array.reshape((n_rot, 3, 1))
        rot_matrices = self.passive_3D_planar_separate(angles)
        combined = (rot_matrices@array).reshape(n_rot, 5)
        return combined
    
    def project_2D(self, array: np.ndarray, angles: int|float|np.ndarray) -> np.ndarray:
        """Projecst the points in "array" into a coordinate system rotated by the angles given in "angles". The 
        rotation is around the third component of "array", meaning it doesn't change that value. If "angles" is a 
        single value, all points are rotated by that angle. If "angles" has as many values as "array" has points, each 
        point is rotated by the angle in "angles" of the same index.

        :param array: (n, 3) numpy.ndarray
        :type array: np.ndarray
        :param angles: A single value or an array of values by which the points in "array" are rotated
        :type angles: np.int|float|np.ndarray
        :return: projected points
        :rtype: np.ndarray
        """
        n_rot = angles.size
        array = array.reshape((n_rot, 2, 1))
        rot_matrices = self.passive_2D(angles)
        return (rot_matrices@array).reshape(n_rot, 2)
    
    def rotate_2D(self, array: np.ndarray, angles: int|float|np.ndarray) -> np.ndarray:
        """Rotates the points in "array" in the current coordinate system rotated by the angles given in "angles". 
        Assuming the coordinates are [x, y], the rotation is around a hypothetical z-axis following the right-hand
        convention. If "angles" is a single value, all points are rotated by that angle. If "angles" has as many values 
        as "array" has points, each point is rotated by the angle in "angles" of the same index.

        :param array: (n, 2) numpy.ndarray
        :type array: np.ndarray
        :param angles: A single value or an array of values by which the points in "array" are rotated
        :type angles: np.int|float|np.ndarray
        :return: rotated points
        :rtype: np.ndarray
        """
        n_rot = angles.size
        array = array.reshape((n_rot, 2, 1))
        rot_matrices = self.acitve_2D(angles)
        return (rot_matrices@array).reshape(n_rot, 2)
        
    @staticmethod
    def _process_rotation(func):
        """Wrapper for the rotations with two functionalities:
        - checks the validity of the angle argument
        - adjusts the rotation matrix if multiple angles are supplied
        """
        def wrapper(angle):
            is_numpy = False
            if isinstance(angle, np.ndarray):
                is_numpy = True
                angle = angle.squeeze()
                if len(angle.shape) > 1:
                    raise ValueError(f"If using '{func.__name__}' with a numpy.ndarray input, the array must "
                                    "be a one dimensional.")
            rot_mat = func(angle)
            if isinstance(angle, (float, int)):
                return rot_mat
            elif is_numpy:
                return rot_mat.transpose(2, 0, 1)
            else:
                raise ValueError(f"The angle(s) for '{func.__name__}()' have to be given as int, float, or "
                                 f"np.ndarray but they were {type(angle)}.")
        return wrapper
    
    @staticmethod
    @_process_rotation
    def passive_2D(angle: float) -> np.ndarray:
        """Passive 2D rotation.

        :param angle: Rotation angle in rad
        :type angle: float
        :return: rotation matrix as (2, 2) np.ndarray 
        :rtype: np.ndarray
        """
        return np.asarray([[np.cos(angle), np.sin(angle)],
                           [-np.sin(angle), np.cos(angle)]])

    @staticmethod
    @_process_rotation
    def acitve_2D(angle: float) -> np.ndarray:
        """Active 2D rotation.

        :param angle: Rotation angle in rad
        :type angle: float
        :return: rotation matrix as (2, 2) np.ndarray 
        :rtype: np.ndarray
        """
        return np.asarray([[np.cos(angle), -np.sin(angle)],
                           [np.sin(angle), np.cos(angle)]])
                
    @staticmethod
    @_process_rotation
    def passive_3D_planar(angle: float) -> np.ndarray:
        """Passive 3D rotation around the axis of the third array index axis.

        :param angle: Rotation angle in rad
        :type angle: float
        :return: rotation matrix as (3, 3) np.ndarray 
        :rtype: np.ndarray
        """
        return np.asarray([[np.cos(angle), np.sin(angle), np.zeros_like(angle)],
                           [-np.sin(angle), np.cos(angle), np.zeros_like(angle)],
                           [np.zeros_like(angle), np.zeros_like(angle), np.ones_like(angle)]])

    @staticmethod
    @_process_rotation
    def passive_3D_planar_separate(angle: float) -> np.ndarray:
        """Passive 3D rotation (projection) around the axis of the third array index axis. The first coordinate of the
        array will be projected onto the rotated first and second axis. The same holds for the second coordinate of the
        array that will be projected. See project_separate() for an example.

        :param angle: Rotation angle in rad
        :type angle: float
        :return: rotation matrix as (5, 3) np.ndarray 
        :rtype: np.ndarray
        """
        return np.asarray([[np.cos(angle), np.zeros_like(angle), np.zeros_like(angle)],
                           [-np.sin(angle), np.zeros_like(angle), np.zeros_like(angle)],
                           [np.zeros_like(angle), np.sin(angle), np.zeros_like(angle)],
                           [np.zeros_like(angle), np.cos(angle), np.zeros_like(angle)],
                           [np.zeros_like(angle), np.zeros_like(angle), np.ones_like(angle)]])
    

class AeroForce(Rotations):
    """Class implementing differnt ways to calculate the lift, drag, and moment depending on the
    state of the system.
    """
    _implemented_schemes = ["steady", "quasi_steady", "BL"]
    
    def __init__(self, dir_polar: str, file_polar: str="polars.dat") -> None:
        """Initialises an instance.

        :param file_polar_data: Path to the file from the current working directory containing the polar data.
        The file must contain the columns ["alpha", "C_d", "C_l", "C_m"].
        :type file_polar_data: str
        """
        super().__init__()
        self.dir_polar = dir_polar
        self.df_polars = pd.read_csv(join(dir_polar, file_polar), delim_whitespace=True)
        self.C_d = interpolate.interp1d(self.df_polars["alpha"], self.df_polars["C_d"])
        self.C_l = interpolate.interp1d(self.df_polars["alpha"], self.df_polars["C_l"])
        self.C_m = interpolate.interp1d(self.df_polars["alpha"], self.df_polars["C_m"])

        self._alpha_0 = None
        self._Cl_slope = 2*np.pi
        self._f_n = None
        self._f_t = None

    def get_function(self, scheme: str) -> Callable:
        """Returns a function object that receives a ThreeDOFsAirfoil instance and returns a numpy array with
        [aero_force_x, aero_force_y, aero_force_moment]. "scheme" sets which kind of calculation for the aerodynamic
        forces is used. Implemented schemes are defined in _implemented_schemes.

        :param scheme: Scheme for calculating the aerodynamic forces. Can be "steady", "quasi_steady", and "BL". "BL"
         uses a Beddoes-Leishman model as given in #todo add paper.
        :type scheme: str
        :raises NotImplementedError: If a scheme is wanted that is not implemented.
        :return: Function calculating the aerodynamic forces.
        :rtype: Callable
        """
        if scheme not in self._implemented_schemes:
            raise NotImplementedError(f"Scheme '{scheme}' is not implemented. Implemented schemes are "
                                      f"{self._implemented_schemes}.")
        self._prepare(scheme)
        scheme_map = {
            "steady": self._steady,
            "quasi_steady": self._quasi_steady,
            "BL": self._BL
        }
        return scheme_map[scheme]
    
    def get_init_function(self, scheme: str) -> Callable:
        if scheme not in self._implemented_schemes:
            raise NotImplementedError(f"Scheme '{scheme}' is not implemented. Implemented schemes are "
                                      f"{self._implemented_schemes}.")
        scheme_map = {
            "steady": self._init_steady,
            "quasi_steady": self._init_quasi_steady,
            "BL": self._init_BL
        }
        return scheme_map[scheme]
    
    def _prepare(self, scheme: str):
        match scheme:
            case "BL":
                dir_BL_data = join(self.dir_polar, "Beddoes_Leishman_preparation")
                f_alpha_0 = join(dir_BL_data, "alpha_0.json")
                f_sep = join(dir_BL_data, "sep_points.dat")
                if not isdir(dir_BL_data):
                    helper.create_dir(dir_BL_data)
                    self._alpha_0 = self._write_and_get_alpha0(alpha=self.df_polars["alpha"].to_numpy(),
                                                               C_l=self.df_polars["C_l"].to_numpy(),
                                                               save_as=f_alpha_0)
                    self._alpha_0 = np.deg2rad(self._alpha_0)
                    df_sep = self._write_and_get_separation_points(alpha_0=self._alpha_0, save_as=f_sep)
                else:
                    with open(f_alpha_0, "r") as f:
                        self._alpha_0 = np.deg2rad(json.load(f)["alpha_0"])
                    df_sep = pd.read_csv(f_sep)
                self._f_n = interpolate.interp1d(df_sep["alpha"], df_sep["f_n"])
                self._f_t = interpolate.interp1d(df_sep["alpha"], df_sep["f_t"])

    def _write_and_get_separation_points(
            self,
            alpha_0: float,
            save_as: str,
            res: int=1000
            ) -> pd.DataFrame:
        alpha_given = self.df_polars["alpha"]
        alpha_interp = np.linspace(alpha_given.min(), alpha_given.max(), res)
        coefficients = np.c_[self.C_d(alpha_interp), self.C_l(alpha_interp)]
        
        alpha_given = np.deg2rad(alpha_given)
        alpha_interp = np.deg2rad(alpha_interp)
        alpha_0 = np.deg2rad(alpha_0)

        coeff_rot = self.project_2D(coefficients, -alpha_interp)
        C_t, C_n = -coeff_rot[:, 0], coeff_rot[:, 1]
        
        f_t = C_t/(self._Cl_slope*np.sin(alpha_interp-alpha_0)**2)
        f_t = f_t**2*np.sign(f_t)
        f_t[f_t>1] = 1
        f_t[f_t< -1] = -1

        f_n = 2*np.sqrt(C_n/(self._Cl_slope*np.sin(alpha_interp-alpha_0)))-1
        f_n = f_n**2*np.sign(f_n)
        f_n[f_n>1] = 1
        f_n[f_n< -1] = -1

        df = pd.DataFrame({"alpha": np.rad2deg(alpha_interp), "f_t": f_t, "f_n": f_n})
        df.to_csv(save_as, index=None)
        return df
    
    @staticmethod
    def _steady():
        raise NotImplementedError

    @staticmethod
    def _init_steady():
        raise NotImplementedError
    
    def _quasi_steady(
            self,
            sim_res: ThreeDOFsAirfoil, 
            i: int,
            pitching_around: float=0.25,
            alpha_at: float=0.7,
            **kwargs) -> np.ndarray: # the kwargs are needed because of the call in simulate()
        """The following is related to the x-y coordinate system of ThreeDOFsAirfoil. A positive pitching_speed 
        turns the nose down (if pitch=0). Positive x is downstream for inflow_angle=0. Positive y is to the suction
        side of the airfoilf (if AoA=0).

        :param sim_res: Instance of ThreeDOFsAirfoil. This instance has the state of the system as its attributes.
        :type sim_res: ThreeDOFsAirfoilBase
        :param i: Index of the current time step 
        :type i: int
        :param pitching_around: position of the pitch axis as distance from the leading edge, normalised on the chord.
         defaults to 0.25
        :type pitching_around: float, optional
        :param alpha_at: position of the calculation of the angle of attack as distance from the leading edge, 
        normalised on the chord. defaults to 0.7
        :type alpha_at: float, optional
        :return: Numpy array containing [aero_force_x, aero_force_y, aero_force_moment].
        :rtype: np.ndarray
        """
        sim_res.alpha_steady[i] = -sim_res.pos[i, 2]+sim_res.inflow_angle[i]
        qs_flow_angle = self._quasi_steady_flow_angle(sim_res.vel[i, :], sim_res.pos[i, :],
                                                      sim_res.inflow[i, :], sim_res.chord, pitching_around, 
                                                      alpha_at)
        
        rot = self.passive_3D_planar(-qs_flow_angle)
        sim_res.alpha_qs[i] = -sim_res.pos[i, 2]+qs_flow_angle

        rel_speed = np.sqrt((sim_res.inflow[i, 0]-sim_res.vel[i, 0])**2+
                            (sim_res.inflow[i, 1]-sim_res.vel[i, 1])**2)

        dynamic_pressure = sim_res.density/2*rel_speed
        alpha_qs_deg = np.rad2deg(sim_res.alpha_qs[i])
        coefficients = np.asarray([self.C_d(alpha_qs_deg), self.C_l(alpha_qs_deg), -self.C_m(alpha_qs_deg)])
        return dynamic_pressure*sim_res.chord*rot@coefficients
    
    @staticmethod
    def _init_quasi_steady(sim_res:ThreeDOFsAirfoil, **kwargs):
        pass

    def _BL(self, 
            sim_res: ThreeDOFsAirfoil,
            i: int,
            A1: float, A2: float, b1: float, b2: float,
            a: float=343, K_alpha: float=0.75,
            T_p: float=1.5, T_bl: float=5,
            Cn_vortex_detach: float=1.0093, tau_vortex_pure_decay: float=11, T_v: float=5,
            pitching_around: float=0.25, alpha_at: float=0.75):
        sim_res.alpha_steady[i] = -sim_res.pos[i, 2]+sim_res.inflow_angle[i]

        # --------- MODULE unsteady attached flow
        flow_vel = np.sqrt(sim_res.inflow[i, :]@sim_res.inflow[i, :].T)
        ds = 2*sim_res.dt[i]*flow_vel/sim_res.chord
        sim_res.s[i] = sim_res.s[i-1]+ds
        qs_flow_angle = self._quasi_steady_flow_angle(sim_res.vel[i, :], sim_res.pos[i, :], 
                                                      sim_res.inflow[i, :], sim_res.chord, pitching_around, 
                                                      alpha_at)
        sim_res.alpha_qs[i] = -sim_res.pos[i, 2]+qs_flow_angle
        d_alpha_qs = sim_res.alpha_qs[i]-sim_res.alpha_qs[i-1] if i != 0 else 0
        sim_res.X_lag[i] = sim_res.X_lag[i-1]*np.exp(-b1*ds)+d_alpha_qs*A1*np.exp(-0.5*b1*ds)
        sim_res.Y_lag[i] = sim_res.Y_lag[i-1]*np.exp(-b2*ds)+d_alpha_qs*A2*np.exp(-0.5*b2*ds)
        sim_res.alpha_eff[i] = sim_res.alpha_qs[i]-sim_res.X_lag[i]-sim_res.Y_lag[i]
        sim_res.C_nc[i] = self._Cl_slope*np.sin(sim_res.alpha_eff[i]-self._alpha_0)  #todo check that this sin() is
        #todo also correct for high aoa in potential flow

        # impulsive (non-circulatory) normal force coefficient
        sim_res.d_alpha_qs_dt[i] = d_alpha_qs/sim_res.dt[i-1]
        tmp = -a*sim_res.dt[i]/(K_alpha*sim_res.chord)
        tmp_2 = sim_res.d_alpha_qs_dt[i]-sim_res.d_alpha_qs_dt[i-1]
        sim_res.D_i[i] = sim_res.D_i[i-1]*np.exp(tmp)+tmp_2*np.exp(0.5*tmp)
        sim_res.C_ni[i] = 4*K_alpha*sim_res.chord/flow_vel*(-sim_res.vel[i, 2]-sim_res.D_i[i])

        # add circulatory and impulsive
        sim_res.C_npot[i] = sim_res.C_nc[i]+sim_res.C_ni[i]
        sim_res.C_tpot[i] = sim_res.C_npot[i]*np.tan(sim_res.alpha_eff[i])

        # --------- MODULE nonlinear trailing edge separation
        #todo why does the impulsive part of the potential flow solution go into the lag term?
        sim_res.D_p[i] = sim_res.D_p[i-1]*np.exp(-ds/T_p)+(sim_res.C_npot[i]-sim_res.C_npot[i-1])*np.exp(-0.5*ds/T_p)
        sim_res.C_nsEq[i] = sim_res.C_npot[i]-sim_res.D_p[i]
        sim_res.alpha_sEq[i] = sim_res.C_nsEq[i]/self._Cl_slope+self._alpha_0  #todo check alpha_0 rad or deg
        
        sim_res.f_t_Dp[i] = self._f_t(np.rad2deg(sim_res.alpha_sEq[i]))
        sim_res.f_n_Dp[i] = self._f_n(np.rad2deg(sim_res.alpha_sEq[i]))

        sim_res.D_bl_t[i] = sim_res.D_bl_t[i-1]*np.exp(-ds/T_bl)+\
            (sim_res.f_t_Dp[i]-sim_res.f_t_Dp[i-1])*np.exp(-0.5*ds/T_bl)
        sim_res.D_bl_n[i] = sim_res.D_bl_n[i-1]*np.exp(-ds/T_bl)+\
            (sim_res.f_n_Dp[i]-sim_res.f_n_Dp[i-1])*np.exp(-0.5*ds/T_bl)
        
        sim_res.f_t[i] = sim_res.f_t_Dp[i]-sim_res.D_bl_t[i]
        sim_res.f_n[i] = sim_res.f_n_Dp[i]-sim_res.D_bl_n[i]

        sim_res.C_nf[i] = sim_res.C_nc[i]*((1+np.sign(sim_res.f_n[i])*np.sqrt(np.abs(sim_res.f_n[i])))/2)**2+sim_res.C_ni[i]
        sim_res.C_tf[i] = self._Cl_slope*np.sin(sim_res.alpha_eff[i]-self._alpha_0)**2*\
            np.sign(sim_res.f_t[i])*np.sqrt(np.abs(sim_res.f_t[i]))
        
        # --------- MODULE leading-edge vortex position
        if sim_res.C_nsEq[i] >= Cn_vortex_detach or d_alpha_qs < 0:
            sim_res.tau_vortex[i] = sim_res.tau_vortex[i-1]+0.45*ds
        else:
            sim_res.tau_vortex[i] = 0

        # --------- MODULE leading-edge vortex lift
        sim_res.C_nv_instant[i] = sim_res.C_nc[i]*(1-((1+np.sign(sim_res.f_n[i])*np.sqrt(sim_res.f_n[i]))/2)**2)
        sim_res.C_nv[i] = sim_res.C_nv[i-1]*np.exp(-ds/T_v)
        if sim_res.tau_vortex[i] < tau_vortex_pure_decay:
            sim_res.C_nv[i] += (sim_res.C_nv_instant[i]-sim_res.C_nv_instant[i-1])*np.exp(-0.5*ds/T_v)
        

        # --------- Combining everything
        # C_t and C_d point in opposite directions!
        # todo update C_m as C_m(alpha_eff)+pitch_rate_term from AEFLap paper
        coefficients = np.asarray([sim_res.C_tf[i], sim_res.C_nf[i]+sim_res.C_nv[i], -self.C_m(sim_res.alpha_qs[i])])
        # rot = self.passive_3D_planar(sim_res.pos[i, 2])  # for [-f_x, f_y, mom]
        rot = self.passive_3D_planar(sim_res.pos[i, 2]+sim_res.inflow_angle[i])  # for [-C_d, C_l, C_m]
        rel_speed = np.sqrt((sim_res.inflow[i, 0]-sim_res.vel[i, 0])**2+
                            (sim_res.inflow[i, 1]-sim_res.vel[i, 1])**2)
        dynamic_pressure = sim_res.density/2*rel_speed
        forces = dynamic_pressure*sim_res.chord*rot@coefficients  # for [-f_x, f_y, mom]
        forces[0] *= -1
        # return forces
        return rot@coefficients  # for [-C_d, C_l, C_m]
    
    def _init_BL(
            self,
            sim_res: ThreeDOFsAirfoil,
            **kwargs
            ):
        sim_res.alpha_steady[0] = -sim_res.pos[0, 2]+np.arctan(sim_res.inflow[0, 1]/sim_res.inflow[0, 0])
        qs_flow_angle = self._quasi_steady_flow_angle(sim_res.vel[0, :], sim_res.pos[0, :], sim_res.inflow[0, :], 
                                                      sim_res.chord, kwargs["pitching_around"], kwargs["alpha_at"])
        sim_res.alpha_qs[0] = -sim_res.pos[0, 2]+qs_flow_angle
    
    @staticmethod
    def _quasi_steady_flow_angle(
        velocity: np.ndarray,
        position: np.ndarray,
        flow: np.ndarray,
        chord: float,
        pitching_around: float,
        alpha_at: float
        ):
        pitching_speed = velocity[2]*chord*(alpha_at-pitching_around)
        v_pitching_x = np.sin(-position[2])*pitching_speed  # velocity of the point
        v_pitching_y = np.cos(position[2])*pitching_speed  # velocity of the point

        v_x = flow[0]-velocity[0]-v_pitching_x
        v_y = flow[1]-velocity[1]-v_pitching_y
        return np.arctan(v_y/v_x)
    
    @staticmethod
    def _write_and_get_alpha0(alpha: np.ndarray, C_l: np.ndarray, save_as: str, alpha0_in: tuple=(-10, 5)):
        ids_subset = np.logical_and(alpha>=alpha0_in[0], alpha<=alpha0_in[1])
        alpha_subset = alpha[ids_subset]
        Cl_subset = C_l[ids_subset]
        find_sign_change = Cl_subset[1:]*Cl_subset[:-1]
        if np.all(find_sign_change>0):
            raise ValueError("The given C_n does not have a sign change in the current AoA interval of "
                            f"{alpha0_in} degree. Linear interpolation can thus not find alpha_0.")
        idx_greater_0 = np.argmax(Cl_subset > 0)
        d_Cn = Cl_subset[idx_greater_0]-Cl_subset[idx_greater_0-1]
        d_alpha = alpha_subset[idx_greater_0]-alpha_subset[idx_greater_0-1]
        alpha_0 = alpha_subset[idx_greater_0]-Cl_subset[idx_greater_0]*d_alpha/d_Cn
        with open(save_as, "w") as f:
            json.dump({"alpha_0": alpha_0}, f)
        return alpha_0

         
class StructForce(Rotations):
    """Class implementing differnt ways to calculate the structural stiffness and damping forces depending on the
    state of the system.
    """
    _implemented_schemes = ["linear"]

    def __init__(
            self,
            stiffness_edge: float, 
            stiffness_flap: float,
            stiffness_tors: float,
            damping_edge: float, 
            damping_flap: float,
            damping_tors: float) -> None:
        super().__init__()
        self.stiffness = np.diag([stiffness_edge, stiffness_flap, stiffness_tors])
        self.damp = np.diag([damping_edge, damping_flap, damping_tors])

    def get_function(self, scheme: str) -> Callable:
        """ Returns a function object that receives a ThreeDOFsAirfoil instance and returns a tuple with two numpy
        arrays of the forces ([damping_x, damping_y, damping_torsion], [stiffness_x, stiffness_y, 
        stiffness_torsion]). "scheme" sets which kind of calculation for the structural forces is used. Implemented 
        schemes are defined in _implemented_schemes.

        :param scheme: Scheme for the calculation of the structural forces. Can be ["linear"].
        :type scheme: str
        :raises NotImplementedError: If a scheme is wanted that is not implemented.
        :return: Function calculating the structural forces.
        :rtype: Callable
        """
        if scheme not in self._implemented_schemes:
            raise NotImplementedError(f"Scheme '{scheme}' is not implemented. Implemented schemes are "
                                      f"{self._implemented_schemes}.")
        scheme_map = {
            "linear": self._linear
        }
        return scheme_map[scheme]

    def _linear(
        self, 
        sim_res: ThreeDOFsAirfoil, 
        i: int, 
        **kwargs) -> tuple[np.ndarray, np.ndarray]:  # the kwargs are needed because of the call in simulate()
        """Calculates the structural stiffness and damping forces based on linear theory.

        :param sim_res: Instance of ThreeDOFsAirfoil. This instance has the state of the system as its attributes.
        :type sim_res: ThreeDOFsAirfoilBase
        :param i: index of current time step
        :type i: int
        :return: Damping and stiffness forces
        :rtype: tuple[np.ndarray, np.ndarray]
        """
        rot = self.passive_3D_planar(sim_res.pos[i, 2])
        stiff = -rot.T@self.stiffness@rot@sim_res.pos[i, :]
        damping = -rot.T@self.damp@rot@sim_res.vel[i, :]
        return damping, stiff


class TimeIntegration:
    """Class implementing differnt ways to calculate the position, velocity, and acceleration of the system at the
     next time step.
    """
    
    _implemented_schemes = ["eE", "HHT-alpha", "HHT-alpha-adapted", "RK4"]    
    # eE: explicit Euler
    # HHT_alpha: algorithm as given in #todo add paper
    # RK4: Runge-Kutta 4th order    

    def __init__(self) -> None:
        self._M_current = None
        self._M_last = None
        self._C_last = None
        self._K_last = None
        self._external_current = None
        self._external_last = None
        self._beta = None
        self._gamma = None
        self._dt = None

    def get_time_step_function(self, scheme: str) -> Callable:
        """ Returns a function object that receives a ThreeDOFsAirfoil instance and returns a numpy array with
        [position, velocity, acceleration] of the system at the next time step. "scheme" sets which kind of calculation for the next state used. Implemented schemes are defined in _implemented_schemes.

        :param scheme: Scheme for the calculation of the next [position, velocity, acceleration]. Can be "eE" (explicit
        Euler), "HHT_alpha" (HHT-alpha method from), or "RK4" (Runge-Kutta 4th order)
        :type scheme: str
        :raises NotImplementedError: If a scheme is wanted that is not implemented.
        :return: Function calculating the next [position, velocity, acceleration]
        :rtype: Callable
        """
        if scheme not in self._implemented_schemes:
            raise NotImplementedError(f"Scheme '{scheme}' is not implemented. Implemented schemes are "
                                      f"{self._implemented_schemes}.")
        scheme_map = {
            "eE": self._eE,
            "RK4": self._RK4,
            "HHT-alpha": self._HHT_alpha,
            "HHT-alpha-adapted": self._HHT_alpha_adapated
        }
        return scheme_map[scheme]
    
    def get_init_function(self, scheme: str) -> Callable:
        if scheme not in self._implemented_schemes:
            raise NotImplementedError(f"Scheme '{scheme}' is not implemented. Implemented schemes are "
                                      f"{self._implemented_schemes}.")
        scheme_map = {
            "eE": None,
            "RK4": None,
            "HHT-alpha": self._init_HHT_alpha,
            "HHT-alpha_adpated": self._init_HHT_alpha
        }
        return scheme_map[scheme]
    
    def _eE(
        self,
        sim_res: ThreeDOFsAirfoil, 
        i: int,
        dt: float,
        **kwargs  # the kwargs are needed because of the call in simulate()
    ):
        """Calculates the system's next position, velocity, and acceleration

        :param sim_res: Instance of ThreeDOFsAirfoil. This instance has the state of the system as its attributes.
        :type sim_res: ThreeDOFsAirfoilBase
        :param i: index of current time step
        :type i: int
        :param dt: time step duration
        :type: float
        :return: next position, velocity, and acceleration
        :rtype: np.ndarray
        """
        accel = self.get_accel(sim_res, i)
        next_vel = sim_res.vel[i, :]+sim_res.accel[i, :]*dt
        next_pos = sim_res.pos[i, :]+(sim_res.vel[i, :]+next_vel)/2*dt
        return next_pos, next_vel, accel
    
    @staticmethod
    def _RK4():
        pass

    def _HHT_alpha(
            self,
            sim_res: ThreeDOFsAirfoil, 
            i: int,
            dt: float,
            **kwargs):  # the kwargs are needed because of the call in simulate():
        rhs = -self._M_last@sim_res.accel[i, :]-self._C_last@sim_res.vel[i, :]-self._K_last@sim_res.pos[i, :]
        rhs += self._external_current*sim_res.aero[i+1, :]+self._external_last*sim_res.aero[i, :]
        #todo the following lines are in the EFRot coordinate system!!! Rotate into XYRot
        accel = solve(self._M_current, rhs, assume_a="sym") 
        vel = sim_res.vel[i, :]+dt*((1-self._gamma)*sim_res.accel[i, :]+self._gamma*accel)
        pos = sim_res.pos[i, :]+dt*sim_res.vel[i, :]+dt**2*((0.5-self._beta)*sim_res.accel[i, :]+self._beta*accel)
        return pos, vel, accel

    def _HHT_alpha_adapated(
            self,
            sim_res: ThreeDOFsAirfoil, 
            i: int,
            dt: float,
            **kwargs):  # the kwargs are needed because of the call in simulate():
        rhs = -self._M_last@sim_res.accel[i, :]-self._C_last@sim_res.vel[i, :]-self._K_last@sim_res.pos[i, :]
        rhs += self._external_current*sim_res.aero[i, :]+self._external_last*sim_res.aero[i-1, :]
        #todo the following lines are in the EFRot coordinate system!!! Rotate into XYRot
        accel = solve(self._M_current, rhs, assume_a="sym") 
        vel = sim_res.vel[i, :]+dt*((1-self._gamma)*sim_res.accel[i, :]+self._gamma*accel)
        pos = sim_res.pos[i, :]+dt*sim_res.vel[i, :]+dt**2*((0.5-self._beta)*sim_res.accel[i, :]+self._beta*accel)
        return pos, vel, accel
    
    def _init_HHT_alpha(
            self,
            sim_res: ThreeDOFsAirfoil,
            alpha: float,
            dt: float,
            **kwargs
            ):  #todo implies constant dt!!!
        self._dt = dt
        self._beta = (1+alpha)**2/4
        self._gamma = 0.5+alpha
        
        M = np.diag([sim_res.mass, sim_res.mass, sim_res.mom_inertia])
        C = np.diag([sim_res._struct_params["damping_edge"], sim_res._struct_params["damping_flap"],
                     sim_res._struct_params["damping_tors"]])
        K = np.diag([sim_res._struct_params["stiffness_edge"], sim_res._struct_params["stiffness_flap"],
                     sim_res._struct_params["stiffness_tors"]])
        
        self._M_current = M+dt*(1-alpha)*self._gamma*C+dt**2*(1-alpha)*self._beta*K
        self._M_last = dt*(1-alpha)*(1-self._gamma)*C+dt**2*(1-alpha)*(0.5-self._beta)*K
        self._C_last = C+dt*(1-alpha)*K
        self._K_last = K
        self._external_current = 1-alpha
        self._external_last = alpha
    
    @staticmethod
    def get_accel(sim_res: ThreeDOFsAirfoil, i: int) -> np.ndarray:
        """Calculates the acceleration for the current time step based on the forces acting on the system.

        :param sim_res: Instance of ThreeDOFsAirfoil. This instance has the state of the system as its attributes.
        :type sim_res: ThreeDOFsAirfoilBase
        :param i: index of current time step
        :type i: int
        :return: acceleration in [x, y, rot z] direction
        :rtype: np.ndarray
        """
        inertia = np.asarray([sim_res.mass, sim_res.mass, sim_res.mom_inertia])
        return (sim_res.aero[i, :]+sim_res.damp[i, :]+sim_res.stiff[i, :])/inertia


class Oscillations:
    def __init__(
            self,
            inertia: np.ndarray,
            natural_frequency: np.ndarray,
            damping_coefficient: np.ndarray,
            ) -> None:
        self._M = inertia 
        self._K = natural_frequency**2*self._M
        self._C = damping_coefficient*2*np.sqrt(self._M*self._K)

        self._zeta = damping_coefficient
        self._omega_n = natural_frequency
        self._omega_d = self._omega_n*np.sqrt(1-self._zeta**2)

    def undamped(self, t: np.ndarray, x_0: np.ndarray=1):
        return x_0*np.cos(self._omega_n*t)
    
    def damped(self, t: np.ndarray, x_0: np.ndarray=1):
        # Theory of Vibration with Applications 2.6-16
        delta = self._zeta*self._omega_n
        return x_0*np.exp(-delta*t)*(delta/self._omega_d*np.sin(self._omega_d*t)+np.cos(self._omega_d*t))
    
    def forced(self, t: np.ndarray, amplitude: np.ndarray, frequency: np.ndarray, x_0: np.ndarray, v_0: np.ndarray):
        # Theory of Vibration with Applications 3.1-2 - 3.1-4
        if amplitude.size != frequency.size:
            raise ValueError("The same number of amplitudes and frequencies for the external excitations have to be "
                            "given.")
        delta = self._zeta*self._omega_n
        ampl_steady = amplitude/np.sqrt((self._K-frequency**2*self._M)**2+(frequency*self._C)**2)
        phase_shift = np.arctan2(frequency*self._C, self._K-frequency**2*self._M)
        A = x_0-(ampl_steady*np.sin(-phase_shift)).sum()
        B = v_0+(delta*A-(ampl_steady*frequency*np.cos(-phase_shift)).sum())/self._omega_d
        x_p = np.exp(-delta*t)*(A*np.cos(self._omega_d*t)+B*np.sin(self._omega_d*t))
        t = t[:, np.newaxis]
        return x_p+(ampl_steady*np.sin(frequency*t-phase_shift)).sum(axis=1)
    
    def step(self, t: np.ndarray, amplitude: np.ndarray):
        # Theory of Vibration with Applications 4.2-3
        phase_shift = np.arctan(self._zeta/np.sqrt(1-self._zeta**2))
        transient_ampl = np.exp(-self._zeta*self._omega_n*t)/np.sqrt(1-self._zeta**2)
        transient_shape = np.cos(np.sqrt(1-self._zeta**2)*self._omega_n*t-phase_shift)
        return amplitude/self._K*(1-transient_ampl*transient_shape)


