import numpy as np
from scipy.linalg import solve
import pandas as pd
from os.path import join
from copy import copy
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
                        df[param+"_"+split_name] = vars(self)[param][:, i]
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
        file_polar_data: str,
        time: np.ndarray, 
        mass: float,
        mom_inertia: float,
        chord: float, 
        stiffness_edge: float, 
        stiffness_flap: float,
        stiffness_tors: float,
        damping_edge: float, 
        damping_flap: float,
        damping_tors: float
        ) -> None:
        """Initialises instance object.

        :param file_polar_data: Path to the file containing polar data for the airfoil. The file must contain the
        columns ["alpha", "Cd", "Cl", "Cm"].
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
        self.file_polar_data = file_polar_data

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
                must_haves = ["A1", "A2", "b1", "b2"]
                self._check_scheme_settings("BL", "set_aero_calc", must_haves, params_given)
                self._aero_scheme_settings = {param: value for param, value in kwargs.items() if param in must_haves}
                raise NotImplementedError
        # add parameters the AeroForce method needs to calculate the aero forces or that are parameters the method
        # calculates that are interesting to save.
        self._added_sim_params[self._dfl_filenames["f_aero"]] = add_aero_sim_params 
        for param in add_aero_sim_params:
            self.add_param(param)  # adds instance attribute to self
        self._aero_force = AeroForce(self.file_polar_data).get_function(scheme)

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
        match scheme:
            case "eE":
                self._time_integration_scheme_settings = {"dt": self.time[1]}
            case "RK4":
                raise NotImplementedError
            case "HHT_alpha":
                # check if all necessary BL parameters are given in kwargs
                raise NotImplementedError
        self._time_integration = TimeIntegration().get_time_step_function(scheme)

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
        self.set_param(inflow=inflow, inflow_angle=inflow_angle, density=density)
        # check whether all necessary settings have been set.
        self._check_simulation_readiness(_aero_force=self._aero_force, 
                                         _struct_force=self._struct_force,
                                         _time_integration=self._time_integration)

        self.pos[0, :] = init_position
        self.vel[0, :] = init_velocity
        for i in range(self.time.size-1):
            # get aerodynamic forces
            self.aero[i, :] = self._aero_force(self, i, **self._aero_scheme_settings)
            # get structural forces
            self.damp[i, :], self.stiff[i, :] = self._struct_force(self, i, **self._struct_scheme_settings)
            # performe time integration
            self.pos[i+1, :], self.vel[i+1, :], self.accel[i+1, :] = self._time_integration(self, i,
                                                                            **self._time_integration_scheme_settings)
        i = self.time.size-1
        # for the last time step, only the forces are calculated because there is no next time step for the positions
        self.aero[i, :] = self._aero_force(self, i, **self._aero_scheme_settings)
        self.damp[i, :], self.stiff[i, :] = self._struct_force(self, i, **self._struct_scheme_settings)

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
        files = files if files is not None else {}
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
    
    def __init__(self, file_polar_data: str) -> None:
        """Initialises an instance.

        :param file_polar_data: Path to the file from the current working directory containing the polar data.
        The file must contain the columns ["alpha", "Cd", "Cl", "Cm"].
        :type file_polar_data: str
        """
        super().__init__()
        df_polars = pd.read_csv(file_polar_data, delim_whitespace=True)
        self.Cd = interpolate.interp1d(df_polars["alpha"], df_polars["Cd"])
        self.Cl = interpolate.interp1d(df_polars["alpha"], df_polars["Cl"])
        self.Cm = interpolate.interp1d(df_polars["alpha"], df_polars["Cm"])

    def get_function(self, scheme: str) -> callable:
        """Returns a function object that receives a ThreeDOFsAirfoil instance and returns a numpy array with
        [aero_force_x, aero_force_y, aero_force_moment]. "scheme" sets which kind of calculation for the aerodynamic
        forces is used. Implemented schemes are defined in _implemented_schemes.

        :param scheme: Scheme for calculating the aerodynamic forces. Can be "steady", "quasi_steady", and "BL". "BL"
         uses a Beddoes-Leishman model as given in #todo add paper.
        :type scheme: str
        :raises NotImplementedError: If a scheme is wanted that is not implemented.
        :return: Function calculating the aerodynamic forces.
        :rtype: callable
        """
        if scheme not in self._implemented_schemes:
            raise NotImplementedError(f"Scheme '{scheme}' is not implemented. Implemented schemes are "
                                      f"{self._implemented_schemes}.")
        scheme_map = {
            "steady": self._steady,
            "quasi_steady": self._quasi_steady,
            "BL": self._BL
        }
        return scheme_map[scheme]
    
    @staticmethod
    def _steady():
        raise NotImplementedError

    def _quasi_steady(
            self,
            sim_results: ThreeDOFsAirfoil, 
            i: int,
            pitching_around: float=0.25,
            alpha_at: float=0.7,
            **kwargs) -> np.ndarray: # the kwargs are needed because of the call in simulate()
        """The following is related to the x-y coordinate system of ThreeDOFsAirfoil. A positive pitching_speed 
        turns the nose down (if pitch=0). Positive x is downstream for inflow_angle=0. Positive y is to the suction
        side of the airfoilf (if AoA=0).

        :param sim_results: Instance of ThreeDOFsAirfoil. This instance has the state of the system as its attributes.
        :type sim_results: ThreeDOFsAirfoilBase
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
        sim_results.alpha_steady[i] = -sim_results.pos[i, 2]+sim_results.inflow_angle[i]
        pitching_speed = sim_results.vel[i, 2]*sim_results.chord*(alpha_at-pitching_around)
        v_pitching_x = np.sin(-sim_results.pos[i, 2])*pitching_speed  # velocity of the point
        v_pitching_y = np.cos(sim_results.pos[i, 2])*pitching_speed  # velocity of the point
    
        v_x = sim_results.inflow[i, 0]-sim_results.vel[i, 0]-v_pitching_x
        v_y = sim_results.inflow[i, 1]-sim_results.vel[i, 1]-v_pitching_y
        flow_angle = np.arctan(v_y/v_x)
        
        rot = self.passive_3D_planar(-flow_angle)
        sim_results.alpha_qs[i] = -sim_results.pos[i, 2]+flow_angle

        rel_speed = np.sqrt((sim_results.inflow[i, 0]-sim_results.vel[i, 0])**2+
                            (sim_results.inflow[i, 1]-sim_results.vel[i, 1])**2)

        dynamic_pressure = sim_results.density/2*rel_speed
        alpha_qs_deg = np.rad2deg(sim_results.alpha_qs[i])
        coefficients = np.asarray([self.Cd(alpha_qs_deg), self.Cl(alpha_qs_deg), -self.Cm(alpha_qs_deg)])
        return dynamic_pressure*sim_results.chord*rot@coefficients

    @staticmethod
    def _BL():
        raise NotImplementedError


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

    def get_function(self, scheme: str) -> callable:
        """ Returns a function object that receives a ThreeDOFsAirfoil instance and returns a tuple with two numpy
        arrays of the forces ([damping_x, damping_y, damping_torsion], [stiffness_x, stiffness_y, 
        stiffness_torsion]). "scheme" sets which kind of calculation for the structural forces is used. Implemented 
        schemes are defined in _implemented_schemes.

        :param scheme: Scheme for the calculation of the structural forces. Can be ["linear"].
        :type scheme: str
        :raises NotImplementedError: If a scheme is wanted that is not implemented.
        :return: Function calculating the structural forces.
        :rtype: callable
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
        sim_results: ThreeDOFsAirfoil, 
        i: int, 
        **kwargs) -> tuple[np.ndarray, np.ndarray]:  # the kwargs are needed because of the call in simulate()
        """Calculates the structural stiffness and damping forces based on linear theory.

        :param sim_results: Instance of ThreeDOFsAirfoil. This instance has the state of the system as its attributes.
        :type sim_results: ThreeDOFsAirfoilBase
        :param i: index of current time step
        :type i: int
        :return: Damping and stiffness forces
        :rtype: tuple[np.ndarray, np.ndarray]
        """
        rot = self.passive_3D_planar(sim_results.pos[i, 2])
        stiff = -rot.T@self.stiffness@rot@sim_results.pos[i, :]
        damping = -rot.T@self.damp@rot@sim_results.vel[i, :]
        return damping, stiff

 
class TimeIntegration:
    """Class implementing differnt ways to calculate the position, velocity, and acceleration of the system at the
     next time step.
    """
    
    _implemented_schemes = ["eE", "HHT_alpha", "RK4"]    
    # eE: explicit Euler
    # HHT_alpha: algorithm as given in #todo add paper
    # RK4: Runge-Kutta 4th order    

    def get_time_step_function(self, scheme: str) -> callable:
        """ Returns a function object that receives a ThreeDOFsAirfoil instance and returns a numpy array with
        [position, velocity, acceleration] of the system at the next time step. "scheme" sets which kind of calculation for the next state used. Implemented schemes are defined in _implemented_schemes.

        :param scheme: Scheme for the calculation of the next [position, velocity, acceleration]. Can be "eE" (explicit
        Euler), "HHT_alpha" (HHT-alpha method from), or "RK4" (Runge-Kutta 4th order)
        :type scheme: str
        :raises NotImplementedError: If a scheme is wanted that is not implemented.
        :return: Function calculating the next [position, velocity, acceleration]
        :rtype: callable
        """
        if scheme not in self._implemented_schemes:
            raise NotImplementedError(f"Scheme '{scheme}' is not implemented. Implemented schemes are "
                                      f"{self._implemented_schemes}.")
        scheme_map = {
            "eE": self._eE,
            "RK4": self._RK4,
            "HHT-alhpa": self._HHT_alpha
        }
        return scheme_map[scheme]
    
    def _eE(
        self,
        sim_results: ThreeDOFsAirfoil, 
        i: int,
        dt: float,
        **kwargs  # the kwargs are needed because of the call in simulate()
    ):
        """Calculates the system's next position, velocity, and acceleration

        :param sim_results: Instance of ThreeDOFsAirfoil. This instance has the state of the system as its attributes.
        :type sim_results: ThreeDOFsAirfoilBase
        :param i: index of current time step
        :type i: int
        :param dt: time step duration
        :type: float
        :return: next position, velocity, and acceleration
        :rtype: np.ndarray
        """
        accel = self.get_accel(sim_results, i)
        next_vel = sim_results.vel[i, :]+sim_results.accel[i, :]*dt
        next_pos = sim_results.pos[i, :]+(sim_results.vel[i, :]+next_vel)/2*dt
        return next_pos, next_vel, accel
    
    @staticmethod
    def _RK4():
        pass

    @staticmethod
    def _HHT_alpha():
        # print(solve(A, b, assume_a="sym"))
        pass
    
    @staticmethod
    def get_accel(sim_results: ThreeDOFsAirfoil, i: int) -> np.ndarray:
        """Calculates the acceleration for the current time step based on the forces acting on the system.

        :param sim_results: Instance of ThreeDOFsAirfoil. This instance has the state of the system as its attributes.
        :type sim_results: ThreeDOFsAirfoilBase
        :param i: index of current time step
        :type i: int
        :return: acceleration in [x, y, rot z] direction
        :rtype: np.ndarray
        """
        inertia = np.asarray([sim_results.mass, sim_results.mass, sim_results.mom_inertia])
        return (sim_results.aero[i, :]+sim_results.damp[i, :]+sim_results.stiff[i, :])/inertia



