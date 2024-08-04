import numpy as np
from scipy.linalg import solve
from scipy.optimize import brentq, newton
import pandas as pd
from os.path import join, isfile
from typing import Callable
import json
from helper_functions import Helper
from scipy import interpolate
from calculation_utils import SimulationResults, SimulationSubRoutine, Rotations
from defaults import Staeblein
helper = Helper()


class ThreeDOFsAirfoil(SimulationResults, Rotations):
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
        time: np.ndarray,
        verbose: bool=True
        ) -> None:
        """Initialises instance object.

        :param file_polar_data: Path to the file containing polar data for the airfoil. The file must contain the
        columns ["alpha", "C_d", "C_l", "C_m"].
        :type file_polar_data: str
        :param time: Numpy array of the time series of the time used for the simulation.
        :type time: np.ndarray
        """
        SimulationResults.__init__(self, time)

        self.verbose = verbose

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

    def approximate_steady_state(self, ffile_polar: str, inflow: np.ndarray, chord: float, stiffness: np.ndarray,  
                                 x: bool=False, y: bool=False, torsion: bool=False, alpha: float=None,
                                 density: float=1.225):
        df_polars = pd.read_csv(ffile_polar, delim_whitespace=True)
        C_d = interpolate.interp1d(np.deg2rad(df_polars["alpha"]), df_polars["C_d"])
        C_l = interpolate.interp1d(np.deg2rad(df_polars["alpha"]), df_polars["C_l"])
        C_m = interpolate.interp1d(np.deg2rad(df_polars["alpha"]), df_polars["C_m"])

        inflow_speed = np.linalg.norm(inflow)
        inflow_angle = np.arctan(inflow[1]/inflow[0])
        dynamic_pressure = 0.5*density*inflow_speed**2
        base_force = dynamic_pressure*chord
        base_moment = -base_force*chord

        stiffness = np.asarray(stiffness) if not isinstance(stiffness, np.ndarray) else stiffness
        alpha = alpha if alpha is None else np.deg2rad(alpha)

        if len(stiffness.shape) == 1:  # no structural coupling
            if alpha is not None:
                tors_angle = base_moment*C_m(alpha)/stiffness[2]
                inflow_angle = alpha+tors_angle
            if torsion:
                if alpha is None:
                    def residue_moment(torsion_angle: float):
                        aero_mom = base_moment*C_m(inflow_angle-torsion_angle)
                        struct_mom = torsion_angle*stiffness[2]
                        # from -aero_mom-struct_mom which would be correct for the coordinate system
                        return aero_mom+struct_mom

                    tors_angle = newton(residue_moment, 0)
            else:
                tors_angle = 0
                inflow_angle = alpha
            aoa = inflow_angle-tors_angle
            # aero forces in x and y
            rot = self.passive_2D(-inflow_angle)
            f_aero_xy = (rot@np.c_[[base_force*C_d(aoa)], [base_force*C_l(aoa)]].T).flatten()
            f_aero_xy[0] = f_aero_xy[0] if x else 0 
            f_aero_xy[1] = f_aero_xy[1] if y else 0 
            f_aero_tors = base_moment*C_m(aoa) if torsion else 0
            f_aero = np.r_[f_aero_xy, f_aero_tors]
            pos_x = f_aero_xy[0]/stiffness[0]
            pos_y = f_aero_xy[1]/stiffness[1]
            return np.asarray([pos_x, pos_y, tors_angle]), f_aero, np.rad2deg(inflow_angle)
        else:  # with structural coupling (currently only supports heave-pitch coupling)
            k_y = stiffness[1, 1]
            k_ytors = stiffness[1, 2]
            k_tors = stiffness[2, 2]
            if alpha is not None:
                L = base_force*C_l(alpha)
                D = base_force*C_d(alpha)
                M = base_moment*C_m(alpha)
                def residue_moment(torsion_angle: float):
                    cos = np.cos(alpha+torsion_angle)
                    sin = np.sin(alpha+torsion_angle)
                    return k_ytors/k_y*(L*cos+D*sin-k_ytors*torsion_angle)+k_tors*torsion_angle-M

                tors_angle = newton(residue_moment, 0)
                inflow_angle = alpha+tors_angle
            if torsion:
                if alpha is None:
                    cos = np.cos(inflow_angle)
                    sin = np.sin(inflow_angle)
                    def residue_moment(torsion_angle: float):
                        alpha = inflow_angle-torsion_angle
                        aero_y = base_force*C_l(alpha)*cos+base_force*C_d(alpha)*sin
                        aero_mom = base_moment*C_m(alpha)
                        return k_ytors/k_y*(aero_y-k_ytors*torsion_angle)+k_tors*torsion_angle-aero_mom
                    tors_angle = newton(residue_moment, 0)
            else:
                tors_angle = 0
                
            aoa = inflow_angle-tors_angle
            # aero forces in x and y
            f_aero_xy = (rot@np.c_[[base_force*C_d(aoa)], [base_force*C_l(aoa)]].T).flatten()
            f_aero_xy[0] = f_aero_xy[0] if x else 0 
            f_aero_xy[1] = f_aero_xy[1] if y else 0 
            f_aero_tors = base_moment*C_m(aoa) if torsion else 0
            f_aero = np.r_[f_aero_xy, f_aero_tors]

            pos_x = f_aero_xy[0]/stiffness[0, 0]
            pos_y = (f_aero_xy[1]-stiffness[1, 2]*tors_angle)/stiffness[1, 1]
            return np.asarray([pos_x, pos_y, tors_angle]), f_aero, np.rad2deg(inflow_angle)
        
    def set_aero_calc(self, dir_polar: str, file_polar: str="polars_new.dat", scheme: str="quasi_steady", **kwargs):
        """Sets how the aerodynamic forces are calculated in the simulation. If the scheme is dependet on constants,
        these must be given in the kwargs. Which kwargs are necessary are defined in the match-case statements in 
        this function's implementation; they are denoted "must_haves". These are mandatory. The "can_haves" are 
        constants for which default values are set. They (the "can_haves") can, but do not have to, thus be set.

        :param scheme: Existing schemes are "steady", "quasi_steady", and "BL_first_order_IAG2". See details in class AeroForce., defaults to "quasi_steady"
        :type scheme: str, optional
        :raises NotImplementedError: If a scheme is wanted that is not implemented.
        """
        self._aero_scheme_settings = kwargs
        AF = AeroForce(dir_polar=dir_polar, file_polar=file_polar, verbose=self.verbose)
        self._added_sim_params[self._dfl_filenames["f_aero"]] = AF.get_additional_params(scheme)
        self._aero_force = AF.get_scheme(scheme, self, "set_aero_calc", **kwargs)
        self._init_aero_force = AF.get_scheme_init_method(scheme)

    def set_struct_calc(self, scheme: str="linear", **kwargs):
        """Sets how the structural forces are calculated in the simulation. If the scheme is dependet on constants,
        these must be given in the kwargs. Which kwargs are necessary are defined in the match-case statements in 
        this function's implementation; they are denoted "must_haves". These are mandatory. The "can_haves" are 
        constants for which default values are set. They (the "can_haves") can, but do not have to, thus be set.

        :param scheme: Existing schemes are "linear". See details in class StructForce. defaults to "linear"
        :type scheme: str, optional
        :raises NotImplementedError: If a scheme is wanted that is not implemented.
        """
        self._struct_scheme_settings = kwargs
        SF = StructForce(verbose=self.verbose)
        self._added_sim_params[self._dfl_filenames["f_structural"]] = SF.get_additional_params(scheme)
        self._struct_force = SF.get_scheme(scheme, self, "set_struct_calc", **kwargs)
        self._init_struct_force = SF.get_scheme_init_method(scheme)

    def set_time_integration(self, scheme: str="eE", **kwargs):
        """Sets how the time integration is done in the simulation. If the scheme is dependet on constants,
        these must be given in the kwargs. Which kwargs are necessary are defined in the match-case statements in 
        this function's implementation; they are denoted "must_haves". These are mandatory. The "can_haves" are 
        constants for which default values are set. They (the "can_haves") can, but do not have to, thus be set.

        :param scheme: Existing schemes are "eE", and "HHT_alpha". See details in class TimeIntegration. 
        defaults to "eE"
        :type scheme: str, optional
        :raises NotImplementedError: If a scheme is wanted that is not implemented.
        """
        self._time_integration_scheme_settings = kwargs
        TI = TimeIntegration(verbose=self.verbose)
        self._time_integration = TI.get_scheme(scheme, self, "set_time_integration", **kwargs)
        self._init_time_integration = TI.get_scheme_init_method(scheme)

    def simulate(
        self, 
        inflow: np.ndarray,
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
        dt = np.r_[self.time[1:]-self.time[:-1], self.time[1]]  # todo hard coded additional dt at the end. Not 
        #todo needed anymore?
        self.set_param(inflow=inflow, inflow_angle=inflow_angle, dt=dt)

        # check whether all necessary settings have been set.
        #todo sim readiness should include inits of sub modules
        self._check_simulation_readiness(_aero_force=self._aero_force, 
                                         _struct_force=self._struct_force,
                                         _time_integration=self._time_integration)
        self.pos[0, :] = init_position
        self.vel[0, :] = init_velocity
        #todo init accel based on damping, stiffness, and external forces

        init_funcs = [self._init_aero_force, self._init_struct_force, self._init_time_integration]
        scheme_settings = [self._aero_scheme_settings, self._struct_scheme_settings, 
                           self._time_integration_scheme_settings]
        for init_func, scheme_setting in zip(init_funcs, scheme_settings):
            init_func(self, **scheme_setting)

        for i in range(self.time.size-1):
            # get aerodynamic forces
            self.aero[i, :] = self._aero_force(self, i, **self._aero_scheme_settings)
            # self.aero[i, :] = 0
            # get structural forces
            self.damp[i, :], self.stiff[i, :] = self._struct_force(self, i, **self._struct_scheme_settings)
            # perform time integration
            self.pos[i+1, :], self.vel[i+1, :], self.accel[i+1, :] = self._time_integration(self, i,
                                                                            **self._time_integration_scheme_settings)
            
        i = self.time.size-1
        # for the last time step, only the forces are calculated because there is no next time step for the positions
        self.aero[i, :] = self._aero_force(self, i, **self._aero_scheme_settings)
        self.damp[i, :], self.stiff[i, :] = self._struct_force(self, i, **self._struct_scheme_settings)

    def simulate_along_path(
        self, 
        inflow: np.ndarray,
        coordinate_system: str,
        init_position: np.ndarray,
        init_velocity: np.ndarray,
        position: dict[str, np.ndarray],
        velocity: dict[str, np.ndarray],
        # acceleration: dict[str, np.ndarray]
        ):
        """Performs a simulation in which the airfoil follows the path of "position" with the respective velocity 
        "velocity" and acceleration "acceleration". Before the simulation can be run, the methods "set_aero_calc()", 
        "set_struct_calc()", and "set_time_integration()" have to be used. 

        The keys for "position", "velocity", and "acceleration" have to be "x", "y", or "rot_z". The path is only set
        for the given keys, all other directions are free to move.

        :param inflow: The inflow at each time step with components in the x and y direction [x, y].
        :type inflow: np.ndarray
        :param density: Density of the fluid
        :type density: float
        :param position: position of the quarter-chord for [x, y, rotation around z in rad]
        :type position: np.ndarray
        :param velocity: Initial velocity of the quarter-chord for [x, y, rotation around z in rad/s]
        :type velocity: np.ndarray
        :param acceleration: Initial acceleration of the quarter-chord for [x, y, rotation around z in rad/s**2]
        :type acceleration: np.ndarray
        """
        set_rotation = True if 2 in position.keys() else False
        inflow_angle = np.arctan(inflow[:, 1]/inflow[:, 0])
        dt = np.r_[self.time[1:]-self.time[:-1], self.time[1]]  # todo hard coded additional dt at the end  
        self.set_param(inflow=inflow, inflow_angle=inflow_angle, dt=dt)

        # check whether all necessary settings have been set.
        #todo sim readiness should include inits of sub modules
        self._check_simulation_readiness(_aero_force=self._aero_force, 
                                         _struct_force=self._struct_force,
                                         _time_integration=self._time_integration)

        init_funcs = [self._init_aero_force, self._init_struct_force, self._init_time_integration]
        scheme_settings = [self._aero_scheme_settings, self._struct_scheme_settings, 
                           self._time_integration_scheme_settings]
        # the init funcs might depend on pos and vel, thus they need to be set before calling the init funcs
        self.pos[0, :] = init_position
        self.vel[0, :] = init_velocity
        for init_func, scheme_setting in zip(init_funcs, scheme_settings):
            init_func(self, **scheme_setting)

        for i in range(self.time.size-1):
            if coordinate_system == "xy":
                for direction, pos in position.items():
                    self.pos[i, direction] = pos[i]
                    self.vel[i, direction] = velocity[direction][i]
                    # self.accel[i, direction] = acceleration[direction][i]
            elif coordinate_system == "ef":
                rot_xy_ef = self.passive_3D_planar(self.pos[i, 2])
                rot_ef_xy = rot_xy_ef.T if not set_rotation else self.passive_3D_planar(-position[2][i])

                pos_ef = rot_xy_ef@self.pos[i, :]
                vel_ef = rot_xy_ef@self.vel[i, :]
                # accel_ef = rot_xy_ef@self.accel[i, :]

                for direction, pos in position.items():
                    pos_ef[direction] = pos[i]
                    vel_ef[direction] = velocity[direction][i]
                    # accel_ef[direction] = acceleration[direction][i]
                
                self.pos[i, :] = rot_ef_xy@pos_ef
                self.vel[i, :] = rot_ef_xy@vel_ef
                # self.accel[i, :] = rot_ef_xy@accel_ef

            # get aerodynamic forces
            self.aero[i, :] = self._aero_force(self, i, **self._aero_scheme_settings)
            # get structural forces
            self.damp[i, :], self.stiff[i, :] = self._struct_force(self, i, **self._struct_scheme_settings)
            # perform time integration
            self.pos[i+1, :], self.vel[i+1, :], self.accel[i+1, :] = self._time_integration(self, i,
                                                                            **self._time_integration_scheme_settings)

        #todo update last time step to set values
        i = self.time.size-1
        # for the last time step, only the forces are calculated because there is no next time step for the positions
        self.aero[i, :] = self._aero_force(self, i, **self._aero_scheme_settings)
        self.damp[i, :], self.stiff[i, :] = self._struct_force(self, i, **self._struct_scheme_settings)

    def save(self, root: str, files: dict=None, split: dict=None, use_default: bool=True, 
             save_last: np.ndarray|list = None):
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
                files[filename] = list(set(files[filename]+params))
        #todo add error if use_default==False and files==None in method call
        apply = {
            "damp": lambda vals: np.multiply(-1, vals),
            "stiff": lambda vals: np.multiply(-1, vals),
        }
        save_ids = None
        if save_last is not None:
            save_ids = self.time >= self.time[-1]-save_last
        self._save(root, files, split, use_default, apply, save_ids=save_ids)

    def _check_simulation_readiness(self, **kwargs):
        """Utility function checking whether all settings given in kwargs have been set.
        """
        for property, value in kwargs.items():
            assert value, f"Method {self._sim_assert[property]}() must be used before simulation."


class AeroForce(SimulationSubRoutine, Rotations):
    """Class implementing differnt ways to calculate the lift, drag, and moment depending on the
    state of the system.
    """
    _implemented_schemes = ["quasi_steady", 
                            "BL_first_order_IAG2",
                            "BL_openFAST_Cl_disc",
                            "BL_Staeblein", 
                            "BL_openFAST_Cl_disc_f_scaled", 
                            "BL_AEROHOR"]

    _scheme_settings_must = {
        "quasi_steady": [],
        "BL_first_order_IAG2": ["A1", "A2", "b1", "b2"],
        "BL_openFAST_Cl_disc": ["A1", "A2", "b1", "b2"],
        "BL_Staeblein": ["A1", "A2", "b1", "b2"],
        "BL_openFAST_Cl_disc_f_scaled": ["A1", "A2", "b1", "b2"],
        "BL_AEROHOR": ["A1", "A2", "b1", "b2"],
    }

    _scheme_settings_can = {
        "quasi_steady": ["density", "chord", "pitching_around", "alpha_at"],
        "BL_first_order_IAG2": ["density", "chord", "a", "K_alpha", "T_p", "T_bl",
                                "tau_vortex_pure_decay", "T_v", "pitching_around", "alpha_at", "K_fC", "K_v",
                                "T_MU", "T_MD"],
        "BL_AEROHOR": ["density", "chord", "a", "K_alpha", "T_p", "T_bl", "tau_vortex_pure_decay", 
                       "T_v", "pitching_around", "alpha_at", "K_v"],
        "BL_openFAST_Cl_disc": ["density", "chord", "T_p", "T_f", "pitching_around", "alpha_at"],
        "BL_Staeblein": ["density", "chord", "T_p", "T_f", "pitching_around", "alpha_at"],
        "BL_openFAST_Cl_disc_f_scaled": ["density", "chord", "T_p", "T_f", "pitching_around", "alpha_at"],
    }

    _sim_params_required = {
        "quasi_steady": ["alpha_qs"],
        "BL_first_order_IAG2": ["ds", "tau_vortex", "D_bl_n", "f_n_Dp", "f_n", "D_p", "D_i",
                                "alpha_qs", "X_lag", "Y_lag", "alpha_eff", "alpha_sEq",
                                "C_nc", "C_nf", "C_nsEq", "C_nv_instant", "C_nv", "C_ni", "C_npot", "C_nvisc",
                                "C_tf",    
                                "C_mf", "C_mV", "C_mC"],
        "BL_AEROHOR": ["ds", "tau_vortex", "D_bl_n", "D_bl_t", "f_n_Dp", "f_n", "f_t_Dp", "f_t", "D_p", "D_i",
                       "alpha_qs", "X_lag", "Y_lag", "alpha_eff", "alpha_sEq",
                       "C_nc", "C_nf", "C_nsEq", "C_nv_instant", "C_nv", "C_ni", "C_npot", 
                       "C_tf", "C_tpot"],
        "BL_openFAST_Cl_disc": ["alpha_qs", "alpha_eff", "x1", "x2", "x3", "x4", "C_lpot", "C_lc", "C_lnc", "C_ds", 
                                "C_dc", "C_dsep", "C_ms", "C_mnc", "f_steady", "alpha_eq", "C_dus", "C_lus", "C_mus"],
        "BL_Staeblein": ["alpha_qs", "alpha_eff", "x1", "x2", "C_lc", "rel_inflow_speed", "rel_inflow_accel",
                         "C_liner", "C_lcent", "C_ds", "C_dind", "C_ms", "C_lift", "C_miner", "C_l"],
        "BL_openFAST_Cl_disc_f_scaled": ["alpha_qs", "alpha_eff", "x1", "x2", "x3", "x4", "C_lpot", "C_lc", "C_lnc", 
                                         "C_ds", "C_dc", "C_dsep", "C_ms", "C_mnc", "f_steady", "T_u", "alpha_eq", 
                                         "C_dus", "C_lus", "C_mus", "rel_inflow_speed"],
    }

    # _copy_scheme = {
    #     "BL_first_order_IAG2": ["BL_openFAST_Cl_disc"]
    # }
    
    def __init__(self, dir_polar: str, file_polar: str="polars.dat", verbose: bool=True) -> None:
        """Initialises an instance.

        :param file_polar_data: Path to the file from the current working directory containing the polar data.
        The file must contain the columns ["alpha", "C_d", "C_l", "C_m"]. Alpha must be in degree!
        :type file_polar_data: str
        """
        SimulationSubRoutine.__init__(self, verbose=verbose)

        self.dir_polar = dir_polar
        self.df_polars = pd.read_csv(join(dir_polar, file_polar), delim_whitespace=True)
        if "alpha" not in self.df_polars:
            self.df_polars = pd.read_csv(join(dir_polar, file_polar))
        
        self._raoa_given = np.deg2rad(self.df_polars["alpha"].to_numpy().flatten())
        self.C_d_polar = interpolate.interp1d(self._raoa_given, self.df_polars["C_d"])
        self.C_l_polar = interpolate.interp1d(self._raoa_given, self.df_polars["C_l"])
        self.C_m_polar = interpolate.interp1d(self._raoa_given, self.df_polars["C_m"])
        self._C_fs = None

        self._alpha_0l = None
        self._alpha_0n_inv = None
        self._alpha_0n_visc = None

        self._C_l_slope = None
        self._C_n_slope = None

        self._f_n = None
        self._f_l = None
        self._f_t = None

    def get_scheme_method(self, scheme: str) -> Callable:
        scheme_methods = {
            "quasi_steady": self._quasi_steady,
            "BL_first_order_IAG2": self._BL_first_order_IAG2,
            "BL_openFAST_Cl_disc": self._BL_openFAST_Cl_disc,
            "BL_Staeblein": self._BL_Staeblein,
            "BL_openFAST_Cl_disc_f_scaled": self._BL_openFAST_Cl_disc_f_scaled,
            "BL_AEROHOR": self._BL_AEROHOR
        }
        return scheme_methods[scheme]
    
    def get_scheme_init_method(self, scheme: str) -> Callable:
        scheme_methods = {
            "quasi_steady": self._init_quasi_steady,
            "BL_first_order_IAG2": self._init_BL_first_order_IAG2,
            "BL_openFAST_Cl_disc": self._init_BL_openFAST_Cl_disc,
            "BL_Staeblein": self._init_BL_Staeblein,
            "BL_openFAST_Cl_disc_f_scaled": self._init_BL_openFAST_f_scaled,
            "BL_AEROHOR": self._init_BL_AEROHOR
        }
        return scheme_methods[scheme]
    
    def _quasi_steady(
            self,
            sim_res: ThreeDOFsAirfoil,
            i: int,
            chord: float=1,
            density: float=1.225,
            pitching_around: float=0.25,
            alpha_at: float=0.75,
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
        qs_flow_angle, _, _ = self._quasi_steady_flow_angle(sim_res.vel[i, :], sim_res.pos[i, :], sim_res.inflow[i, :],
                                                            chord, pitching_around, alpha_at)
        rot = self.passive_3D_planar(-qs_flow_angle)
        sim_res.alpha_qs[i] = -sim_res.pos[i, 2]+qs_flow_angle

        rel_speed = np.sqrt((sim_res.inflow[i, 0]-sim_res.vel[i, 0])**2+
                            (sim_res.inflow[i, 1]-sim_res.vel[i, 1])**2)
        
        dynamic_pressure = density/2*rel_speed**2
        coefficients = np.asarray([self.C_d_polar(sim_res.alpha_qs[i]), 
                                   self.C_l_polar(sim_res.alpha_qs[i]), 
                                   -self.C_m_polar(sim_res.alpha_qs[i])*chord])
        return dynamic_pressure*chord*rot@coefficients
    
    @staticmethod
    def _init_quasi_steady(sim_res:ThreeDOFsAirfoil, **kwargs):
        pass

    def _BL_first_order_IAG2(self, 
            sim_res: ThreeDOFsAirfoil,
            i: int,
            A1: float, A2: float, b1: float, b2: float,
            chord: float=1, density: float=1.225, pitching_around: float=0.25, alpha_at: float=0.75,
            a: float=343, K_alpha: float=0.75, K_fC: float=0.1, K_v: float=0.2,
            T_p: float=1.7, T_bl: float=3, T_v: float=6, T_MU: float=1.5, T_MD: float=1.5,
            tau_vortex_pure_decay: float=6, alpha_critical=None):
        sim_res.alpha_steady[i] = -sim_res.pos[i, 2]+sim_res.inflow_angle[i]
        # --------- general calculations
        qs_flow_angle, v_x, v_y = self._quasi_steady_flow_angle(sim_res.vel[i, :], sim_res.pos[i, :],
                                                                sim_res.inflow[i, :], chord, pitching_around, alpha_at)
        sim_res.alpha_qs[i] = -sim_res.pos[i, 2]+qs_flow_angle
        rel_flow_vel = np.sqrt(v_x**2+v_y**2)
        sim_res.ds[i] = 2*sim_res.dt[i]*rel_flow_vel/chord 

        # --------- MODULE unsteady attached flow
        d_alpha_qs = sim_res.alpha_qs[i]-sim_res.alpha_qs[i-1]
        # d_alpha_qs = sim_res.alpha_steady[i]-sim_res.alpha_steady[i-1]
        sim_res.X_lag[i] = sim_res.X_lag[i-1]*np.exp(-b1*sim_res.ds[i-1])+d_alpha_qs*A1*np.exp(-0.5*b1*sim_res.ds[i-1])
        sim_res.Y_lag[i] = sim_res.Y_lag[i-1]*np.exp(-b2*sim_res.ds[i-1])+d_alpha_qs*A2*np.exp(-0.5*b2*sim_res.ds[i-1])
        sim_res.alpha_eff[i] = sim_res.alpha_qs[i]-sim_res.X_lag[i]-sim_res.Y_lag[i]
        if sim_res.f_n[i-1] != 0:
            sim_res.C_nc[i] = self._C_n_slope*(sim_res.alpha_eff[i]-self._alpha_0n_inv) 
        else:
            sim_res.C_nc[i] = 4*self._C_n_visc(sim_res.alpha_eff[i])

        # impulsive (non-circulatory) normal force coefficient
        tmp = -a*sim_res.dt[i]/(K_alpha*chord)
        tmp_2 = -(sim_res.vel[i, 2]-sim_res.vel[i-1, 2])  # minus here needed because of coordinate system
        sim_res.D_i[i] = sim_res.D_i[i-1]*np.exp(tmp)+tmp_2*np.exp(0.5*tmp)
        sim_res.C_ni[i] = 4*K_alpha*chord/rel_flow_vel*(-sim_res.vel[i, 2]-sim_res.D_i[i])  # -vel because of CS

        # add circulatory and impulsive
        sim_res.C_npot[i] = sim_res.C_nc[i]+sim_res.C_ni[i]

        # --------- MODULE nonlinear trailing edge separation
        #todo why does the impulsive part of the potential flow solution go into the lag term?
        sim_res.D_p[i] = sim_res.D_p[i-1]*np.exp(-sim_res.ds[i-1]/T_p)
        sim_res.D_p[i] += (sim_res.C_npot[i]-sim_res.C_npot[i-1])*np.exp(-0.5*sim_res.ds[i-1]/T_p)
        sim_res.C_nsEq[i] = sim_res.C_npot[i]-sim_res.D_p[i]
        sim_res.alpha_sEq[i] = sim_res.C_nsEq[i]/self._C_n_slope+self._alpha_0n_inv
        
        sim_res.f_n[i] = self._f_n(sim_res.alpha_sEq[i])
        tmp_bl = -sim_res.ds[i-1]/T_bl
        sim_res.D_bl_n[i] = sim_res.D_bl_n[i-1]*np.exp(tmp_bl)+(sim_res.f_n[i]-sim_res.f_n[i-1])*np.exp(0.5*tmp_bl)
        sim_res.f_n_Dp[i] = sim_res.f_n[i]-sim_res.D_bl_n[i]

        sim_res.C_nvisc[i] = self._C_n_slope*(sim_res.alpha_eff[i]-self._alpha_0n_visc)
        sim_res.C_nvisc[i] *= ((1+np.sqrt(sim_res.f_n_Dp[i]))/2)**2
        sim_res.C_nf[i] = sim_res.C_nvisc[i]+sim_res.C_ni[i]

        sim_res.C_tf[i] = self._C_t_visc(sim_res.alpha_sEq[i])
        
        # --------- MODULE leading-edge vortex position
        sim_res.tau_vortex[i] = sim_res.tau_vortex[i-1]
        if sim_res.C_nsEq[i] > self._C_n_crit:
            sim_res.tau_vortex[i] += 0.225*sim_res.ds[i-1]  # 0.225 because sim_res.ds[i-1]/2
        elif sim_res.C_nsEq[i] < self._C_n_crit and d_alpha_qs >= 0:
            sim_res.tau_vortex[i] *= np.exp(-sim_res.ds[i-1])
        
        # --------- MODULE leading-edge vortex lift
        sim_res.C_nv_instant[i] = sim_res.C_nc[i]*(1-((1+np.sqrt(sim_res.f_n_Dp[i]))/2)**2)
        sim_res.C_nv[i] = sim_res.C_nv[i-1]*np.exp(-sim_res.ds[i-1]/T_v)
        if 0 < sim_res.tau_vortex[i] and sim_res.tau_vortex[i] < tau_vortex_pure_decay:
            sim_res.C_nv[i] += (sim_res.C_nv_instant[i]-sim_res.C_nv_instant[i-1])*np.exp(-0.5*sim_res.ds[i-1]/T_v)
        
        # --------- MODULE moment coefficient
        # viscous
        sim_res.C_mf[i] = self.C_m_polar(sim_res.alpha_eff[i])

        # vortex
        C_Pv = K_v*(1-np.cos(np.pi*sim_res.tau_vortex[i]/tau_vortex_pure_decay))
        sim_res.C_mV[i] = -C_Pv*sim_res.C_nv[i]  # vortex

        # circulatory
        C_Pf = K_fC*self._C_n_crit
        if (sim_res.tau_vortex[i] < tau_vortex_pure_decay) and (d_alpha_qs >= 0):
            tmp = -sim_res.ds[i-1]/T_MU  # T_MU!
            tmp2 = C_Pf*(sim_res.C_nv_instant[i]-sim_res.C_nv_instant[i-1])
            sim_res.C_mC[i] = sim_res.C_mC[i]*np.exp(tmp)-tmp2*np.exp(tmp/2)
        elif d_alpha_qs < 0:
            tmp = -sim_res.ds[i-1]/T_MD  # T_MD!
            tmp2 = C_Pf*(sim_res.C_nv_instant[i]-sim_res.C_nv_instant[i-1])
            sim_res.C_mC[i] = sim_res.C_mC[i]*np.exp(tmp)-tmp2*np.exp(tmp/2)
        else:
            sim_res.C_mC[i] = sim_res.C_mC[i-1]


        # --------- Combining everything
        coefficients = np.asarray([sim_res.C_tf[i], 
                                   sim_res.C_nf[i]+sim_res.C_nv[i], 
                                   sim_res.C_mf[i]+sim_res.C_mV[i]+sim_res.C_mC[i]])
        # not -pos[i, 2] in the next line because for pos[i, 2]=0, the C_t axis is opposite to the
        # x axis.
        rot = self.passive_3D_planar(sim_res.pos[i, 2])  # since it's C_t and C_n
        coeffs = rot@coefficients
        coeffs[0] = -coeffs[0]
        C_d_polar = self.C_d_polar(sim_res.alpha_eff[i])
        if coeffs[0] < C_d_polar and sim_res.alpha_qs[i]<self._alpha_crit:
            coeffs[0] = C_d_polar

        # for return of [C_d, C_l, C_m]
        # return coeffs

        # for return of [f_x, f_y, mom]
        dynamic_pressure = density/2*rel_flow_vel**2
        forces = dynamic_pressure*np.asarray([chord, chord, -chord**2])*coefficients
        return forces  # for [f_x, f_y, mom]

    def _init_BL_first_order_IAG2(
            self,
            sim_res: SimulationResults,
            chord: float=1,
            alpha_critical: float=15.1,
            pitching_around: float=0.25,
            alpha_at: float=0.75,
            resolution: int=1000,
            characteristics_aoa_range: tuple[float, float]=(-10, 20),
            **kwargs
            ):
        # define C_n, C_t, alpha_0n_inv, alpha_0n_visc, C_n_slope
        alpha_interp = np.linspace(self._raoa_given.min(), self._raoa_given.max(), resolution)

        def C_n_visc(alpha):
            return np.cos(alpha)*self.C_l_polar(alpha)+np.sin(alpha)*self.C_d_polar(alpha)  
        self._C_n_visc = C_n_visc

        def C_t_visc(alpha):
            return np.sin(alpha)*self.C_l_polar(alpha)-np.cos(alpha)*self.C_d_polar(alpha) 
        self._C_t_visc = C_t_visc 

        dir_BL_data = helper.create_dir(join(self.dir_polar, "preparation", "BL_first_order_IAG2"))[0]
        ff_aero_characteristics = join(dir_BL_data, "aero_characteristics.json")
        if not isfile(ff_aero_characteristics):
            C_d = self.C_d_polar(alpha_interp)
            C_l = self.C_l_polar(alpha_interp)
            C_n_inv = np.cos(alpha_interp)*C_l+np.sin(alpha_interp)*(C_d-C_d.min())      
            aoa_range = np.deg2rad(np.asarray(characteristics_aoa_range))
            ids = np.logical_and(alpha_interp>=aoa_range[0], alpha_interp<=aoa_range[1])
            self._save_aero_characteristics(ff_aero_characteristics, alpha_interp[ids],
                                            {"C_n_visc": self._C_n_visc(alpha_interp[ids]), "C_n_inv": C_n_inv[ids]}, 
                                            {"C_d": C_d})
        
        with open(ff_aero_characteristics, "r") as ff_aero:
            aero_characteristics = json.load(ff_aero)

        self._alpha_0n_visc = aero_characteristics["C_n_visc_root"]
        self._alpha_0n_inv = aero_characteristics["C_n_inv_root"]
        self._C_n_slope = aero_characteristics["C_n_inv_max_slope"]
        self._C_d0 = aero_characteristics["C_d_min"]
        self._alpha_crit = np.deg2rad(alpha_critical)
        self._C_n_crit = self._C_n_slope*(self._alpha_crit-self._alpha_0n_inv)


        # define f_n
        ff_f_n = join(dir_BL_data, "f_n.dat")
        if not isfile(ff_f_n):
            def sqrt_of_f_n(alpha):
                return 2*np.sqrt(self._C_n_visc(alpha)/(self._C_n_slope*(alpha-self._alpha_0n_visc)))-1
            
            alpha_f_n, f_n = self._adjust_f(alpha_interp, sqrt_of_f_n, False, False)
            pd.DataFrame({"alpha_n": alpha_f_n, "f_n": f_n}).to_csv(ff_f_n, index=None)
        else:
            df_f = pd.read_csv(ff_f_n)
            alpha_f_n = df_f["alpha_n"].to_numpy()
            f_n = df_f["f_n"].to_numpy()
        self._f_n = interpolate.interp1d(alpha_f_n, f_n)

        # now initialise more parameters
        qs_flow_angle, _, _ = self._quasi_steady_flow_angle(sim_res.vel[0, :], sim_res.pos[0, :],
                                                                sim_res.inflow[0, :], chord, pitching_around, alpha_at)
        sim_res.alpha_qs[-1] = -sim_res.pos[0, 2]+qs_flow_angle
        sim_res.C_npot[-1] = self._C_n_slope*(sim_res.alpha_qs[-1]-self._alpha_0n_inv)
        sim_res.X_lag[-1] = 0
        sim_res.Y_lag[-1] = 0
        sim_res.f_n[-1] = self._f_n(sim_res.alpha_qs[-1])
        
        if sim_res.f_n[-1] != 0:
            sim_res.C_nc[-1] = self._C_n_slope*(sim_res.alpha_qs[-1]-self._alpha_0n_inv) 
        else:
            sim_res.C_nc[-1] = 4*self._C_n_visc(sim_res.alpha_qs[-1])
                 
    def _BL_AEROHOR(self, 
            sim_res: ThreeDOFsAirfoil,
            i: int,
            A1: float, A2: float, b1: float, b2: float,
            chord: float=1, density: float=1.225, pitching_around: float=0.25, alpha_at: float=0.75,
            a: float=343, K_alpha: float=0.75,
            T_p: float=1.7, T_bl: float=3, T_v: float=6, tau_vortex_pure_decay: float=6, alpha_critical=None):
        sim_res.alpha_steady[i] = -sim_res.pos[i, 2]+sim_res.inflow_angle[i]
        # --------- general calculations
        qs_flow_angle, v_x, v_y = self._quasi_steady_flow_angle(sim_res.vel[i, :], sim_res.pos[i, :],
                                                                sim_res.inflow[i, :], chord, pitching_around, alpha_at)
        sim_res.alpha_qs[i] = -sim_res.pos[i, 2]+qs_flow_angle
        rel_flow_vel = np.sqrt(v_x**2+v_y**2)
        sim_res.ds[i] = 2*sim_res.dt[i]*rel_flow_vel/chord 

        # --------- MODULE unsteady attached flow
        d_alpha_qs = sim_res.alpha_qs[i]-sim_res.alpha_qs[i-1]
        sim_res.X_lag[i] = sim_res.X_lag[i-1]*np.exp(-b1*sim_res.ds[i-1])+d_alpha_qs*A1*np.exp(-0.5*b1*sim_res.ds[i-1])
        sim_res.Y_lag[i] = sim_res.Y_lag[i-1]*np.exp(-b2*sim_res.ds[i-1])+d_alpha_qs*A2*np.exp(-0.5*b2*sim_res.ds[i-1])
        sim_res.alpha_eff[i] = sim_res.alpha_qs[i]-sim_res.X_lag[i]-sim_res.Y_lag[i]
        sim_res.C_nc[i] = self._C_n_slope*(sim_res.alpha_eff[i]-self._alpha_0n_inv)

        # impulsive (non-circulatory) normal force coefficient
        tmp = -a*sim_res.dt[i]/(K_alpha*chord)
        tmp_2 = -(sim_res.vel[i, 2]-sim_res.vel[i-1, 2])
        sim_res.D_i[i] = sim_res.D_i[i-1]*np.exp(tmp)+tmp_2*np.exp(0.5*tmp)
        sim_res.C_ni[i] = 4*K_alpha*chord/rel_flow_vel*(-sim_res.vel[i, 2]-sim_res.D_i[i])

        # add circulatory and impulsive
        sim_res.C_npot[i] = sim_res.C_nc[i]+sim_res.C_ni[i]
        sim_res.C_tpot[i] = sim_res.C_npot[i]*np.tan(sim_res.alpha_eff[i])

        # --------- MODULE nonlinear trailing edge separation
        #todo why does the impulsive part of the potential flow solution go into the lag term?
        sim_res.D_p[i] = sim_res.D_p[i-1]*np.exp(-sim_res.ds[i-1]/T_p)
        sim_res.D_p[i] += (sim_res.C_npot[i]-sim_res.C_npot[i-1])*np.exp(-0.5*sim_res.ds[i-1]/T_p)
        sim_res.C_nsEq[i] = sim_res.C_npot[i]-sim_res.D_p[i]
        sim_res.alpha_sEq[i] = sim_res.C_nsEq[i]/self._C_n_slope+self._alpha_0n_inv

        sim_res.f_t[i] = self._f_t(sim_res.alpha_sEq[i])
        sim_res.f_n[i] = self._f_n(sim_res.alpha_sEq[i])

        tmp_bl = -sim_res.ds[i-1]/T_bl
        sim_res.D_bl_t[i] = sim_res.D_bl_t[i-1]*np.exp(tmp_bl)+(sim_res.f_t[i]-sim_res.f_t[i-1])*np.exp(0.5*tmp_bl)
        sim_res.D_bl_n[i] = sim_res.D_bl_n[i-1]*np.exp(tmp_bl)+(sim_res.f_n[i]-sim_res.f_n[i-1])*np.exp(0.5*tmp_bl)
                
        sim_res.f_t_Dp[i] = sim_res.f_t[i]-sim_res.D_bl_t[i]
        sim_res.f_n_Dp[i] = sim_res.f_n[i]-sim_res.D_bl_n[i]

        sim_res.C_tf[i] = sim_res.C_tpot[i]*np.sign(sim_res.f_t_Dp[i])*np.sqrt(np.abs(sim_res.f_t_Dp[i]))
        C_nqs = sim_res.C_nc[i]*((1+np.sign(sim_res.f_n_Dp[i])*np.sqrt(np.abs(sim_res.f_n_Dp[i])))/2)**2
        sim_res.C_nf[i] = C_nqs+sim_res.C_ni[i]
        
        # --------- MODULE leading-edge vortex position
        if sim_res.C_nsEq[i] >= self._C_n_crit:
            sim_res.tau_vortex[i] = sim_res.tau_vortex[i-1]+0.225*sim_res.ds[i-1]
        else:
            if d_alpha_qs >= 0:
                sim_res.tau_vortex[i] = 0
            else:
                sim_res.tau_vortex[i] = sim_res.tau_vortex[i-1]
        
        # --------- MODULE leading-edge vortex lift
        sim_res.C_nv_instant[i] = sim_res.C_nc[i]*(1-((1+np.sqrt(sim_res.f_n_Dp[i]))/2)**2)
        sim_res.C_nv[i] = sim_res.C_nv[i-1]*np.exp(-sim_res.ds[i-1]/T_v)
        if 0 < sim_res.tau_vortex[i] and sim_res.tau_vortex[i] < tau_vortex_pure_decay:
            sim_res.C_nv[i] += (sim_res.C_nv_instant[i]-sim_res.C_nv_instant[i-1])*np.exp(-0.5*sim_res.ds[i-1]/T_v)

        # --------- Combining everything
        coefficients = np.asarray([sim_res.C_tf[i], sim_res.C_nf[i]+sim_res.C_nv[i], 0])
        # not -pos[i, 2] in the next line because for pos[i, 2]=0, the C_t axis is opposite to the
        # x axis.
        rot = self.passive_3D_planar(sim_res.pos[i, 2])
        coeffs = rot@coefficients
        coeffs[0] = -coeffs[0]+self._C_d0  # C_t is facing forwards, C_d backwards

        # for return of [C_d, C_l, C_m] uncomment next line
        # return coeffs

        # for return of [f_x, f_y, mom] uncommmet next lines
        dynamic_pressure = density/2*rel_flow_vel**2
        forces = dynamic_pressure*np.asarray([chord, chord, -chord**2])*coeffs 
        return forces  # for [f_x, f_y, mom]

    def _init_BL_AEROHOR(
            self,
            sim_res: SimulationResults,
            alpha_critical: float=14.1,
            chord: float=1,
            pitching_around: float=0.25,
            alpha_at: float=0.75,
            resolution: int=300,
            characteristics_aoa_range: tuple[float, float]=(-10, 20),
            **kwargs):
        # define C_n, C_t, alpha_0n_inv, alpha_0n_visc, C_n_slope
        alpha_interp = np.linspace(self._raoa_given.min(), self._raoa_given.max(), resolution)
        C_d = self.C_d_polar(alpha_interp)
        C_d0 = np.min(C_d)
        C_l = self.C_l_polar(alpha_interp)
        
        dir_BL_data = helper.create_dir(join(self.dir_polar, "preparation", "BL_AEROHOR"))[0]
        ff_aero_characteristics = join(dir_BL_data, "aero_characteristics.json")
        if not isfile(ff_aero_characteristics):
            C_n_inv = np.cos(alpha_interp)*C_l+np.sin(alpha_interp)*(C_d-C_d0)
            aoa_range = np.deg2rad(np.asarray(characteristics_aoa_range))
            ids = np.logical_and(alpha_interp>=aoa_range[0], alpha_interp<=aoa_range[1])
            self._save_aero_characteristics(ff_aero_characteristics, alpha_interp[ids], {"C_n_inv": C_n_inv[ids]})
        
        with open(ff_aero_characteristics, "r") as ff_aero:
            aero_characteristics = json.load(ff_aero)

        self._alpha_0n_inv = aero_characteristics["C_n_inv_root"]
        self._C_n_slope = aero_characteristics["C_n_inv_max_slope"]
        self._C_d0 = C_d0
        self._C_n_crit = self._C_n_slope*(np.deg2rad(alpha_critical)-self._alpha_0n_inv)
        
        # define f_n and f_t
        ff_f_n = join(dir_BL_data, "f_n.dat")
        if not isfile(ff_f_n):
            def sqrt_of_f_n(alpha):
                C_n_inv = np.cos(alpha_interp)*C_l+np.sin(alpha_interp)*(C_d-C_d0)
                return 2*np.sqrt(C_n_inv/(self._C_n_slope*(alpha-self._alpha_0n_inv)))-1
            
            alpha_f_n, f_n = self._adjust_f(alpha_interp, sqrt_of_f_n, adjust_attached=False, adjust_separated=False)
            pd.DataFrame({"alpha_n": alpha_f_n, "f_n": f_n}).to_csv(ff_f_n, index=None)
        else:
            df_f = pd.read_csv(ff_f_n)
            alpha_f_n = df_f["alpha_n"].to_numpy()
            f_n = df_f["f_n"].to_numpy()
        self._f_n = interpolate.interp1d(alpha_f_n, f_n)

        ff_f_t = join(dir_BL_data, "f_t.dat")
        if not isfile(ff_f_t):
            def sqrt_of_f_t(alpha):
                C_t_inv = np.sin(alpha_interp)*C_l-np.cos(alpha_interp)*(C_d-C_d0)
                return C_t_inv/(self._C_n_slope*(alpha-self._alpha_0n_inv)*np.tan(alpha))
            
            alpha_f_t, f_t = self._adjust_f(alpha_interp, sqrt_of_f_t, adjust_attached=False, adjust_separated=False)
            f_t[f_t>1] = 1
            f_t[f_t< -1] = -1
            pd.DataFrame({"alpha_t": alpha_f_t, "f_t": f_t}).to_csv(join(dir_BL_data, "f_t.dat"), index=None)
        else:
            df_f = pd.read_csv(ff_f_t)
            alpha_f_t = df_f["alpha_t"].to_numpy()
            f_t = df_f["f_t"].to_numpy()
        self._f_t = interpolate.interp1d(alpha_f_t, f_t)

        # init BL params
        sim_res.X_lag[-1] = 0
        sim_res.Y_lag[-1] = 0
        qs_flow_angle, _, _ = self._quasi_steady_flow_angle(sim_res.vel[0, :], sim_res.pos[0, :],
                                                                sim_res.inflow[0, :], chord, pitching_around, alpha_at)
        sim_res.alpha_qs[-1] = -sim_res.pos[0, 2]+qs_flow_angle
        sim_res.C_npot[-1] = self._C_n_slope*(sim_res.alpha_qs[-1]-self._alpha_0n_inv) 
        sim_res.f_t[-1] = self._f_t(sim_res.alpha_qs[-1])
        sim_res.f_n[-1] = self._f_n(sim_res.alpha_qs[-1])

    def _BL_openFAST_Cl_disc(
            self, 
            sim_res: SimulationResults,
            i: int,
            A1: float, A2: float, b1: float, b2: float,
            chord: float=1, density: float=1.225, pitching_around: float=0.5, alpha_at: float=0.75,
            T_p: float=1.5, T_f: float=6):
        lcs = {param: value for param, value in locals().items() if param != "self"}
        lcs["C_slope"] = self._C_l_slope
        lcs["alpha_0"] = self._alpha_0l_visc
        return self._BL_openFAST_disc(**lcs)
    
    def _init_BL_openFAST_Cl_disc(
            self,
            sim_res: SimulationResults,
            chord: float=1,
            pitching_around: float=0.25,
            alpha_at: float=0.75,
            resolution: int=1000,
            characteristics_aoa_range: tuple[float, float]=(-10, 15),
            **kwargs
            ):
        # define alpha_0l_inv, C_l_slope   
        dir_BL_data = helper.create_dir(join(self.dir_polar, "preparation", "BL_openFAST_Cl_disc"))[0]
        ff_aero_characteristics = join(dir_BL_data, "aero_characteristics.json")

        if not isfile(ff_aero_characteristics):
            aoa_range = np.deg2rad(np.asarray(characteristics_aoa_range))
            ids = np.logical_and(self._raoa_given>=aoa_range[0], self._raoa_given<=aoa_range[1])
            self._save_aero_characteristics(ff_aero_characteristics, self._raoa_given[ids],
                                            {"C_l_visc": self.C_l_polar(self._raoa_given[ids])},
                                            {"C_d": self.C_d_polar(self._raoa_given[ids])})
        
        with open(ff_aero_characteristics, "r") as ff_aero:
            aero_characteristics = json.load(ff_aero)

        self._alpha_0l_visc = aero_characteristics["C_l_visc_root"]
        self._C_l_slope = aero_characteristics["C_l_visc_max_slope"]
        self._C_d0 = aero_characteristics["C_d_min"]

        # define f_l
        ff_sep_points = join(dir_BL_data, "f_l.dat")
        if not isfile(ff_sep_points):
            def sqrt_of_f_l(alpha):
                return 2*np.sqrt(self.C_l_polar(alpha)/(self._C_l_slope*(alpha-self._alpha_0l_visc)))-1
            
            alpha_interp = np.linspace(self._raoa_given.min(), self._raoa_given.max(), resolution)
            alpha_f_l, f_l = self._adjust_f(alpha_interp, sqrt_of_f_l, adjust_attached=False)
            df_sep = pd.DataFrame({"alpha_l": alpha_f_l, "f_l": f_l})
            df_sep.to_csv(ff_sep_points, index=None)
        else:
            df_sep = pd.read_csv(ff_sep_points)
        self._f_l = interpolate.interp1d(df_sep["alpha_l"], df_sep["f_l"])

        # define C_lfs
        ff_C_l_fs = join(dir_BL_data, "C_l_fs.dat")
        if not isfile(ff_C_l_fs):
            alpha_fs, C_l_fs = self._get_C_fs(alpha_f_l, f_l, alpha_interp, self.C_l_polar(alpha_interp), 
                                              self._alpha_0l_visc, self._C_l_slope)
            pd.DataFrame({"alpha_fs": alpha_fs, "C_l_fs": C_l_fs}).to_csv(ff_C_l_fs, index=None)
        else:
            df_C_l_fs = pd.read_csv(ff_C_l_fs)
            alpha_fs = df_C_l_fs["alpha_fs"].to_numpy().flatten()
            C_l_fs = df_C_l_fs["C_l_fs"].to_numpy().flatten()
        self._C_fs = interpolate.interp1d(alpha_fs, C_l_fs)

        # currently does not support an initial non-zero velocity of the airfoil
        alpha_steady = -sim_res.pos[0, 2]+sim_res.inflow_angle[0]
        qs_flow_angle, _, _ = self._quasi_steady_flow_angle(sim_res.vel[0, :], sim_res.pos[0, :], 
                                                                sim_res.inflow[0, :], chord, pitching_around, alpha_at)
        alpha_qs = -sim_res.pos[0, 2]+qs_flow_angle
        sim_res.alpha_qs[-1] = alpha_qs
        sim_res.x1[-1] = alpha_qs*kwargs["A1"]
        sim_res.x2[-1] = alpha_qs*kwargs["A2"]
        sim_res.C_lpot[-1] = self._C_l_slope*(alpha_qs-self._alpha_0l_visc)
        sim_res.x3[-1] = self._C_l_slope*(alpha_qs-self._alpha_0l_visc)
        sim_res.x4[-1] = self._f_l(alpha_steady)
        sim_res.f_steady[-1] = self._f_l(alpha_steady)

    def _BL_openFAST_Cn_disc(
            self, 
            sim_res: SimulationResults,
            i: int,
            A1: float, A2: float, b1: float, b2: float,
            chord: float=1, density: float=1.225, pitching_around: float=0.5, alpha_at: float=0.75,
            T_p: float=1.5, T_f: float=6):
        #todo this is not really correct. It just switches Cl and Cn, but does that even work (looking at Cd and Ct).
        lcs = {param: value for param, value in locals().items() if param != "self"}
        lcs["C_slope"] = self._C_n_slope
        lcs["alpha_0"] = self._alpha_0n
        return self._BL_openFAST_disc(**lcs)

    def _BL_openFAST_disc(
            self, 
            C_slope: float, alpha_0: float,
            sim_res: SimulationResults,
            i: int,
            A1: float, A2: float, b1: float, b2: float,
            chord: float=1, density: float=1.225, pitching_around: float=0.5, alpha_at: float=0.75,
            T_p: float=1.5, T_f: float=6):
        sim_res.alpha_steady[i] = -sim_res.pos[i, 2]+sim_res.inflow_angle[i]
        qs_flow_angle, v_x, v_y = self._quasi_steady_flow_angle(sim_res.vel[i, :], sim_res.pos[i, :], 
                                                                sim_res.inflow[i, :], chord, pitching_around, alpha_at)
        rel_inflow_speed = np.sqrt(v_x**2+v_y**2)
        sim_res.alpha_qs[i] = -sim_res.pos[i, 2]+qs_flow_angle
        T_u_current = 0.5*chord/rel_inflow_speed
        
        _, v_x_last, v_y_last = self._quasi_steady_flow_angle(sim_res.vel[i-1, :], sim_res.pos[i-1, :], 
                                                              sim_res.inflow[i-1, :], chord, pitching_around, alpha_at)
        T_u_last = 0.5*chord/np.sqrt(v_x_last**2+v_y_last**2)  #todo not just the wind velocity in the denominator?
        tmp1 = np.exp(-sim_res.dt[i-1]*b1/T_u_last) 
        tmp2 = np.exp(-sim_res.dt[i-1]*b2/T_u_last)
        alpha_qs_avg = 0.5*(sim_res.alpha_qs[i-1]+sim_res.alpha_qs[i])
        sim_res.x1[i] = sim_res.x1[i-1]*tmp1+alpha_qs_avg*A1*(1-tmp1)  # this discretisation causes the failure in 
        sim_res.x2[i] = sim_res.x2[i-1]*tmp2+alpha_qs_avg*A2*(1-tmp2)  # reconstructing the static polar exactly

        sim_res.alpha_eff[i] = sim_res.alpha_qs[i]*(1-A1-A2)+sim_res.x1[i]+sim_res.x2[i]
        sim_res.C_lpot[i] = C_slope*(sim_res.alpha_eff[i]-alpha_0)-np.pi*T_u_current*sim_res.vel[i, 2]

        tmp3 = np.exp(-sim_res.dt[i-1]/(T_u_last*T_p))
        sim_res.x3[i] = sim_res.x3[i-1]*tmp3+0.5*(sim_res.C_lpot[i-1]+sim_res.C_lpot[i])*(1-tmp3)
        sim_res.alpha_eq[i] = sim_res.x3[i]/C_slope+alpha_0

        tmp4 = np.exp(-sim_res.dt[i-1]/(T_u_last*T_f))
        sim_res.f_steady[i] = self._f_l(sim_res.alpha_eq[i])
        sim_res.x4[i] = sim_res.x4[i-1]*tmp4+0.5*(sim_res.f_steady[i-1]+sim_res.f_steady[i])*(1-tmp4)

        sim_res.C_lc[i] = sim_res.x4[i]*C_slope*(sim_res.alpha_eff[i]-alpha_0)
        sim_res.C_lc[i] += (1-sim_res.x4[i])*self._C_fs(sim_res.alpha_eff[i])
        sim_res.C_lnc[i] = -np.pi*T_u_current*sim_res.vel[i, 2]
        sim_res.C_lus[i] = sim_res.C_lc[i]+sim_res.C_lnc[i]

        sim_res.C_ds[i] = self.C_d_polar(sim_res.alpha_eff[i])
        tmp = (np.sqrt(sim_res.f_steady[i])-np.sqrt(sim_res.x4[i]))/2-(sim_res.f_steady[i]-sim_res.x4[i])/4
        sim_res.C_dsep[i] = (sim_res.C_ds[i]-self._C_d0)*tmp
        sim_res.C_dc[i] = (sim_res.alpha_qs[i]-sim_res.alpha_eff[i]-T_u_current*sim_res.vel[i, 2])*sim_res.C_lc[i]
        sim_res.C_dus[i] = sim_res.C_ds[i]+sim_res.C_dc[i]+sim_res.C_dsep[i]

        sim_res.C_ms[i] = self.C_m_polar(sim_res.alpha_eff[i])
        sim_res.C_mnc[i] = 0.5*np.pi*T_u_current*sim_res.vel[i, 2]
        sim_res.C_mus[i] = sim_res.C_ms[i]+sim_res.C_mnc[i]

        # --------- Combining everything
        coeffs = np.asarray([sim_res.C_dus[i], sim_res.C_lus[i], sim_res.C_mus[i]])

        # for return of [C_d, C_l, C_m]
        # return coeffs

        # for return of [f_x, f_y, mom]
        dynamic_pressure = density/2*rel_inflow_speed**2
        rot = self.passive_3D_planar(-sim_res.alpha_eff[i]-sim_res.pos[i, 2])
        return dynamic_pressure*np.asarray([chord, chord, -chord**2])*rot@coeffs
    
    def _BL_openFAST_Cl_disc_f_scaled(
            self, 
            sim_res: SimulationResults,
            i: int,
            A1: float, A2: float, b1: float, b2: float,
            chord: float=1, density: float=1.225, pitching_around: float=0.5, alpha_at: float=0.75,
            T_p: float=1.5, T_f: float=6):
        lcs = {param: value for param, value in locals().items() if param != "self"}
        lcs["C_slope"] = self._C_l_slope
        lcs["alpha_0"] = self._alpha_0l_visc
        return self._BL_openFAST_disc_f_scaled(**lcs)
    
    def _init_BL_openFAST_f_scaled(
            self,
            sim_res: SimulationResults,
            chord: float=1,
            pitching_around: float=0.25,
            alpha_at: float=0.75,
            resolution: int=1000,
            characteristics_aoa_range: tuple[float, float]=(-10, 20),
            **kwargs
            ):
        # define alpha_0l_inv, C_l_slope   
        dir_BL_data = helper.create_dir(join(self.dir_polar, "preparation", "BL_openFAST_Cl_disc_f_scaled"))[0]
        ff_aero_characteristics = join(dir_BL_data, "aero_characteristics.json")

        if not isfile(ff_aero_characteristics):
            aoa_range = np.deg2rad(np.asarray(characteristics_aoa_range))
            ids = np.logical_and(self._raoa_given>=aoa_range[0], self._raoa_given<=aoa_range[1])
            self._save_aero_characteristics(ff_aero_characteristics, self._raoa_given[ids],
                                            {"C_l_visc": self.C_l_polar(self._raoa_given[ids])},
                                            {"C_d": self.C_d_polar(self._raoa_given[ids])})
        
        with open(ff_aero_characteristics, "r") as ff_aero:
            aero_characteristics = json.load(ff_aero)

        self._alpha_0l_visc = aero_characteristics["C_l_visc_root"]
        self._C_l_slope = aero_characteristics["C_l_visc_max_slope"]
        self._C_d0 = aero_characteristics["C_d_min"]

        # define f_l
        ff_sep_points = join(dir_BL_data, "f_l.dat")
        if not isfile(ff_sep_points):
            def sqrt_of_f_l(alpha):
                return 2*np.sqrt(self.C_l_polar(alpha)/(self._C_l_slope*(alpha-self._alpha_0l_visc)))-1
            
            alpha_interp = np.linspace(self._raoa_given.min(), self._raoa_given.max(), resolution)
            alpha_f_l, f_l = self._adjust_f(alpha_interp, sqrt_of_f_l, adjust_attached=False)
            df_sep = pd.DataFrame({"alpha_l": alpha_f_l, "f_l": f_l})
            df_sep.to_csv(ff_sep_points, index=None)
        else:
            df_sep = pd.read_csv(ff_sep_points)
        self._f_l = interpolate.interp1d(df_sep["alpha_l"], df_sep["f_l"])

        # define C_lfs
        ff_C_l_fs = join(dir_BL_data, "C_l_fs.dat")
        if not isfile(ff_C_l_fs):
            alpha_fs, C_l_fs = self._get_C_fs(alpha_f_l, f_l, alpha_interp, self.C_l_polar(alpha_interp), 
                                              self._alpha_0l_visc, self._C_l_slope)
            pd.DataFrame({"alpha_fs": alpha_fs, "C_l_fs": C_l_fs}).to_csv(ff_C_l_fs, index=None)
        else:
            df_C_l_fs = pd.read_csv(ff_C_l_fs)
            alpha_fs = df_C_l_fs["alpha_fs"].to_numpy().flatten()
            C_l_fs = df_C_l_fs["C_l_fs"].to_numpy().flatten()
        self._C_fs = interpolate.interp1d(alpha_fs, C_l_fs)

        # currently does not support an initial non-zero velocity of the airfoil
        alpha_steady = -sim_res.pos[0, 2]+sim_res.inflow_angle[0]
        qs_flow_angle, _, _ = self._quasi_steady_flow_angle(sim_res.vel[0, :], sim_res.pos[0, :], sim_res.inflow[0, :], 
                                                            chord, pitching_around, alpha_at)
        alpha_qs = -sim_res.pos[0, 2]+qs_flow_angle
        sim_res.alpha_qs[-1] = alpha_qs
        sim_res.x1[-1] = 0
        sim_res.x2[-1] = 0
        sim_res.rel_inflow_speed[-1] = np.sqrt(sim_res.inflow[0, :]@sim_res.inflow[0, :].T)
        sim_res.dt[-1] = sim_res.dt[0]
        sim_res.C_lpot[-1] = self._C_l_slope*(alpha_qs-self._alpha_0l_visc)
        sim_res.x3[-1] = self._C_l_slope*(alpha_qs-self._alpha_0l_visc)
        sim_res.x4[-1] = self._f_l(alpha_steady)
        sim_res.f_steady[-1] = self._f_l(alpha_steady)
        sim_res.T_u[-1] = 2*sim_res.rel_inflow_speed[-1]*sim_res.dt[-1]/chord
  
    def _BL_openFAST_disc_f_scaled(
            self, 
            C_slope: float, alpha_0: float,
            sim_res: SimulationResults,
            i: int,
            A1: float, A2: float, b1: float, b2: float,
            chord: float=1, density: float=1.225, pitching_around: float=0.5, alpha_at: float=0.75,
            T_p: float=1.5, T_f: float=6):
        sim_res.alpha_steady[i] = -sim_res.pos[i, 2]+sim_res.inflow_angle[i]
        qs_flow_angle, v_x, v_y = self._quasi_steady_flow_angle(sim_res.vel[i, :], sim_res.pos[i, :], 
                                                                sim_res.inflow[i, :], chord, pitching_around, alpha_at)
        sim_res.alpha_qs[i] = -sim_res.pos[i, 2]+qs_flow_angle
        sim_res.rel_inflow_speed[i] = np.sqrt(v_x**2+v_y**2)
        sim_res.T_u[i] = 0.5*chord/sim_res.rel_inflow_speed[i] #todo not just the wind velocity in the denominator right? Just the wind velocity doesn't really make sense

        tmp1 = np.exp(-sim_res.dt[i-1]*b1/sim_res.T_u[i-1]) 
        tmp2 = np.exp(-sim_res.dt[i-1]*b2/sim_res.T_u[i-1])
        tmp_t = sim_res.T_u[i-1]/sim_res.dt[i-1]
        d_downwash = sim_res.alpha_qs[i]*sim_res.rel_inflow_speed[i]-sim_res.alpha_qs[i-1]*sim_res.rel_inflow_speed[i-1]
        sim_res.x1[i] = sim_res.x1[i-1]*tmp1+d_downwash*A1/b1*tmp_t*(1-tmp1)*sim_res.x4[i-1]
        sim_res.x2[i] = sim_res.x2[i-1]*tmp2+d_downwash*A2/b2*tmp_t*(1-tmp2)*sim_res.x4[i-1]

        sim_res.alpha_eff[i] = sim_res.alpha_qs[i]-(sim_res.x1[i]+sim_res.x2[i])/sim_res.rel_inflow_speed[i]
        sim_res.C_lpot[i] = C_slope*(sim_res.alpha_eff[i]-alpha_0)-np.pi*sim_res.T_u[i]*sim_res.vel[i, 2]

        tmp3 = np.exp(-sim_res.dt[i-1]/(sim_res.T_u[i-1]*T_p))
        sim_res.x3[i] = sim_res.x3[i-1]*tmp3+0.5*(sim_res.C_lpot[i-1]+sim_res.C_lpot[i])*(1-tmp3)
        sim_res.alpha_eq[i] = sim_res.x3[i]/C_slope+alpha_0

        tmp4 = np.exp(-sim_res.dt[i-1]/(sim_res.T_u[i-1]*T_f))
        sim_res.f_steady[i] = self._f_l(sim_res.alpha_eq[i])
        sim_res.x4[i] = sim_res.x4[i-1]*tmp4+0.5*(sim_res.f_steady[i-1]+sim_res.f_steady[i])*(1-tmp4)

        sim_res.C_lc[i] = sim_res.x4[i]*C_slope*(sim_res.alpha_eff[i]-alpha_0)
        sim_res.C_lc[i] += (1-sim_res.x4[i])*self._C_fs(sim_res.alpha_eff[i])
        sim_res.C_lnc[i] = -np.pi*sim_res.T_u[i]*sim_res.vel[i, 2]
        sim_res.C_lus[i] = sim_res.C_lc[i]+sim_res.C_lnc[i]

        sim_res.C_ds[i] = self.C_d_polar(sim_res.alpha_eff[i])
        tmp = (np.sqrt(sim_res.f_steady[i])-np.sqrt(sim_res.x4[i]))/2-(sim_res.f_steady[i]-sim_res.x4[i])/4
        sim_res.C_dsep[i] = (sim_res.C_ds[i]-self._C_d0)*tmp
        sim_res.C_dc[i] = (sim_res.alpha_qs[i]-sim_res.alpha_eff[i]-sim_res.T_u[i]*sim_res.vel[i, 2])*sim_res.C_lc[i]
        sim_res.C_dus[i] = sim_res.C_ds[i]+sim_res.C_dc[i]+sim_res.C_dsep[i]

        sim_res.C_ms[i] = self.C_m_polar(sim_res.alpha_eff[i])
        sim_res.C_mnc[i] = 0.5*np.pi*sim_res.T_u[i]*sim_res.vel[i, 2]
        sim_res.C_mus[i] = sim_res.C_ms[i]+sim_res.C_mnc[i]

        # --------- Combining everything
        coeffs = np.asarray([sim_res.C_dus[i], sim_res.C_lus[i], sim_res.C_mus[i]])

        # for return of [C_d, C_l, C_m]
        # return coeffs

        # for return of [f_x, f_y, mom]
        dynamic_pressure = density/2*sim_res.rel_inflow_speed[i]**2
        rot = self.passive_3D_planar(-sim_res.alpha_eff[i]-sim_res.pos[i, 2])
        return dynamic_pressure*np.asarray([chord, chord, -chord**2])*rot@coeffs

    def _BL_Staeblein(
            self, 
            sim_res: SimulationResults,
            i: int,
            A1: float, A2: float, b1: float, b2: float, 
            chord: float=1, density: float=1.225, pitching_around: float=0.5, alpha_at: float=0.75):
        # get inflow conditions
        sim_res.alpha_steady[i] = -sim_res.pos[i, 2]+sim_res.inflow_angle[i]
        qs_flow_angle, v_x, v_y = self._quasi_steady_flow_angle(sim_res.vel[i, :], sim_res.pos[i, :], 
                                                                sim_res.inflow[i, :], chord, pitching_around, alpha_at)
        T_u_current = 0.5*chord/np.sqrt(v_x**2+v_y**2)

        sim_res.alpha_qs[i] = -sim_res.pos[i, 2]+qs_flow_angle
        sim_res.rel_inflow_speed[i] = np.sqrt((sim_res.inflow[i, 0]-sim_res.vel[i, 0])**2+
                                              (sim_res.inflow[i, 1]-sim_res.vel[i, 1])**2)
        sim_res.rel_inflow_accel[i] = (sim_res.rel_inflow_speed[i]-sim_res.rel_inflow_speed[i-1])/sim_res.dt[i-1]

        # get circulatory lift coefficient
        tmp1 = (sim_res.rel_inflow_speed[i]+sim_res.rel_inflow_speed[i-1])/chord
        tmp2 = sim_res.rel_inflow_accel[i]+sim_res.rel_inflow_accel[i-1]
        tmp2 /= sim_res.rel_inflow_speed[i]+sim_res.rel_inflow_speed[i-1]
        tmp3 = sim_res.rel_inflow_speed[i-1]*sim_res.alpha_qs[i-1]+sim_res.rel_inflow_speed[i]*sim_res.alpha_qs[i]

        avg_P1 = b1*tmp1+tmp2
        avg_P2 = b2*tmp1+tmp2
        avg_Q1 = b1*A1/chord*tmp3
        avg_Q2 = b2*A2/chord*tmp3

        C1 = np.exp(-avg_P1*sim_res.dt[i-1])
        C2 = np.exp(-avg_P2*sim_res.dt[i-1])

        I1 = avg_Q1/avg_P1*(1-C1)
        I2 = avg_Q2/avg_P2*(1-C2)
        
        sim_res.x1[i] = sim_res.x1[i-1]*C1+I1
        sim_res.x2[i] = sim_res.x2[i-1]*C2+I2

        sim_res.alpha_eff[i] = sim_res.alpha_qs[i]*(1-A1-A2)+sim_res.x1[i]+sim_res.x2[i]
        sim_res.C_lc[i] = self._C_l_slope*(sim_res.alpha_eff[i]-self._alpha_0l_inv)

        # get circulatory lift
        dynamic_pressure = density/2*sim_res.rel_inflow_speed[i]**2
        base_force = dynamic_pressure*chord
        L_c = base_force*sim_res.C_lc[i]

        # get inertial lift contribution
        # L_iner is the apparent force times the vertical acceleration at the semi-chord position
        m_apparent = density*np.pi*chord**2/4
        # L_iner = -m_apparent*(sim_res.accel[i, 1]+
        #                       (1/2-pitching_around)*chord*sim_res.accel[i, 2]*np.cos(sim_res.pos[i,2]))
        L_iner = -m_apparent*(sim_res.accel[i, 1]+(1/2-pitching_around)*chord*sim_res.accel[i, 2])
        
        # get centrifugal lift contribution
        # minus because the torsional direction here is opposite to that of the paper but the lift direction is the same
        L_cent = -m_apparent*sim_res.rel_inflow_speed[i]*sim_res.vel[i, 2]
        
        # get drag
        D_s = base_force*self.C_d_polar(sim_res.alpha_eff[i])
        D_ind = L_c*(sim_res.alpha_qs[i]-sim_res.alpha_eff[i]-T_u_current*sim_res.vel[i, 2])
        # alpha_quasi_geometric = np.arctan2(sim_res.inflow[i, 1]-sim_res.vel[i, 1],
        #                                    sim_res.inflow[i, 0]-sim_res.vel[i, 0])-sim_res.pos[i, 2]
        # D_ind = L_c*(alpha_quasi_geometric-sim_res.alpha_eff[i]-T_u_current*sim_res.vel[i, 2])
        D = D_s+D_ind

        # get moment
        M_s = -base_force*chord*self.C_m_polar(sim_res.alpha_eff[i])  # minus because a positive C_m means nose
        # up by definition, but in the CS used here nose down -> include minus to correct direction
        M_lift = chord/2*(L_iner/2+L_cent)  # moment caused by L_iner and L_cent w.r.t. the quarter-chord
        M_iner = m_apparent*chord**2/32*sim_res.accel[i, 2]  # this minus stays because the acceleration term is on the 
        M = M_s+M_lift-M_iner
        # same axes as the moment

        # save coefficients
        sim_res.C_liner[i] = L_iner/base_force
        sim_res.C_lcent[i] = L_cent/base_force
        sim_res.C_l[i] = sim_res.C_liner[i]+sim_res.C_lcent[i]+sim_res.C_lc[i]
        sim_res.C_ds[i] = D_s/base_force
        sim_res.C_dind[i] = D_ind/base_force
        sim_res.C_ms[i] = -M_s/(base_force*chord)
        sim_res.C_lift[i] = -M_lift/(base_force*chord)
        sim_res.C_miner[i] = -M_iner/(base_force*chord)

        # combine everything
        f_aero = np.asarray([D, L_c+L_iner+L_cent, M])

        # for return of [C_d, C_l, C_m] uncomment the three following lines
        coeffs = f_aero/base_force
        coeffs[2] /= -chord
        sim_res.C_dus[i] = coeffs[0]
        sim_res.C_lus[i] = coeffs[1]
        sim_res.C_mus[i] = coeffs[2]
        # return coeffs

        # for return of [f_x, f_y, mom] uncomment the two following lines 
        rot = self.passive_3D_planar(-sim_res.alpha_eff[i]-sim_res.pos[i, 2])
        return rot@f_aero

    def _init_BL_Staeblein(
            self,
            sim_res: SimulationResults,
            chord: float=1,
            pitching_around: float=0.25,
            alpha_at: float=0.75,
            **kwargs
    ):
        self._C_l_slope = 7.15
        self._alpha_0l_inv = 0
        # currently does not support an initial non-zero velocity of the airfoil
        init_inflow_vel = np.linalg.norm(sim_res.inflow[0, :])
        next_inflow_vel = np.linalg.norm(sim_res.inflow[1, :])
        init_inflow_accel = (next_inflow_vel-init_inflow_vel)/sim_res.dt[0]
        sim_res.rel_inflow_accel[-1] = init_inflow_accel
        sim_res.rel_inflow_speed[-1] = init_inflow_vel-init_inflow_accel*sim_res.dt[0]

        qs_flow_angle, _, _ = self._quasi_steady_flow_angle(sim_res.vel[0, :], sim_res.pos[0, :],
                                                            sim_res.inflow[0, :], chord, pitching_around, alpha_at)
        alpha_qs = -sim_res.pos[0, 2]+qs_flow_angle
        sim_res.alpha_qs[-1] = alpha_qs
        sim_res.x1[-1] = alpha_qs*kwargs["A1"]
        sim_res.x2[-1] = alpha_qs*kwargs["A2"]

    def _save_aero_characteristics(
            self,
            save_to: str,
            alpha: np.ndarray,
            coefficients: dict[str, np.ndarray],
            min_of: dict[str, np.ndarray]={}
    ):
        aero = {}
        for coeff, values in coefficients.items():
            root, root_slope, max_slope = self._get_zero_crossing_and_max_slope(alpha, values)
            aero[coeff+"_root"] = root
            aero[coeff+"_root_slope"] = root_slope
            aero[coeff+"_max_slope"] = max_slope
        
        for coeff, values in min_of.items():
            aero[coeff+"_min"] = values.min()
        
        with open(save_to, "w") as ff_aero:
            json.dump(aero, ff_aero, indent=4)

    @staticmethod
    def _quasi_steady_flow_angle(
        velocity: np.ndarray,
        position: np.ndarray,
        flow: np.ndarray,
        chord: float,
        pitching_around: float,  # as factor of chord!
        alpha_at: float  #  as factor of chord!
        ):
        pitching_speed = velocity[2]*chord*(alpha_at-pitching_around)
        v_pitching_x = np.sin(-position[2])*pitching_speed  # x velocity of the point
        v_pitching_y = np.cos(position[2])*pitching_speed  # y velocity of the point

        v_x = flow[0]-velocity[0]-v_pitching_x
        v_y = flow[1]-velocity[1]-v_pitching_y
        return np.arctan2(v_y, v_x), v_x, v_y
    
    @staticmethod
    def _get_zero_crossing_and_max_slope(x: np.ndarray, y: np.ndarray, gradient: str="pos") -> tuple[float, float]:
        """Returns the root, the slope at root, and the maximum slope from the root to any other point.
        Could currently only work for functions with positive gradients at the root.
        :param x: _description_
        :type x: np.ndarray
        :param y: _description_
        :type y: np.ndarray
        :param gradient: _description_, defaults to "pos"
        :type gradient: str, optional
        :return: _description_
        :rtype: tuple[float, float]
        """
        idx_greater_0 = np.argmax(y > 0) if gradient == "pos" else np.argmax(y < 0)
        dy_zero = y[idx_greater_0]-y[idx_greater_0-1]
        dx_zero = x[idx_greater_0]-x[idx_greater_0-1]
        x_0 = x[idx_greater_0]-y[idx_greater_0]*dx_zero/dy_zero
        max_dy_dx = 0
        for i in range(x.size-1):
            max_dy_dx = max(max_dy_dx, abs(y[i]/(x_0-x[i])))
        return x_0, dy_zero/dx_zero, max_dy_dx
    
    @staticmethod
    def _adjust_f(
            alpha: np.ndarray,
            sqrt_of_f: Callable,
            adjust_attached: bool=True,
            adjust_separated: bool=True,
            keep_sign: bool=True,
            attached_region: tuple=(-1.05, 1.05)  # approx from -60deg to 60deg
    ) -> pd.DataFrame:
        def resiude_separated(alpha):
            return sqrt_of_f(alpha)

        def residue_attached(alpha):
            return resiude_separated(alpha)-1
        
        def handle_sign(sqrt_f: np.ndarray):
            return sqrt_f**2*np.sign(sqrt_f) if keep_sign else sqrt_f**2
        
        if not adjust_attached and not adjust_separated:
            return alpha, handle_sign(sqrt_of_f(alpha))
        
        alpha_sub = alpha[np.logical_and(alpha>=attached_region[0], alpha<=attached_region[1])]
        res_sep = resiude_separated(alpha_sub)
        res_att = res_sep-1

        if adjust_separated:
            id_begin_attachement = (res_sep>0).argmax()
            begin_attachement = brentq(resiude_separated, alpha_sub[id_begin_attachement-2], 
                                    alpha_sub[id_begin_attachement])
            id_end_attched = res_att.size-(res_sep[::-1]>0).argmax()-1
            end_attachement = brentq(resiude_separated, alpha_sub[id_end_attched], alpha_sub[id_end_attched+2])
            
        if adjust_attached:
            id_fully_attached = (res_att>0).argmax()
            begin_fully_attached = brentq(residue_attached, alpha_sub[id_fully_attached-2], 
                                          alpha_sub[id_fully_attached])

            id_end_fully_attached = res_att.size-(res_att[::-1]>0).argmax()-1
            end_fully_attached = brentq(residue_attached, alpha_sub[id_end_fully_attached], 
                                        alpha_sub[id_end_fully_attached+2])

        if adjust_separated and adjust_attached:
            alpha_begins_attaching = alpha_sub[np.logical_and(alpha_sub>begin_attachement, 
                                                              alpha_sub<begin_fully_attached)]
            alpha_ends_attached = alpha_sub[np.logical_and(alpha_sub>end_fully_attached, alpha_sub<end_attachement)]
            f = np.r_[0, 0, handle_sign(resiude_separated(alpha_begins_attaching)), 1, 1, 
                      handle_sign(resiude_separated(alpha_ends_attached)), 0, 0]
            alpha = np.r_[alpha.min(), begin_attachement, alpha_begins_attaching, begin_fully_attached, 
                          end_fully_attached, alpha_ends_attached, end_attachement, alpha.max()]
        elif adjust_attached:
            alpha_below_fully_attached = alpha[alpha<begin_fully_attached]
            alpha_above_fully_attached = alpha[alpha>end_fully_attached]
            f = np.r_[handle_sign(resiude_separated(alpha_below_fully_attached)), 1, 1, 
                      handle_sign(resiude_separated(alpha_above_fully_attached))]
            alpha = np.r_[alpha_below_fully_attached, begin_fully_attached, end_fully_attached, 
                          alpha_above_fully_attached]
        elif adjust_separated:
            alpha_attached = alpha[np.logical_and(alpha>begin_attachement, alpha<end_attachement)]
            f = np.r_[0, handle_sign(resiude_separated(alpha_attached)), 0]
            alpha = np.r_[alpha.min(), alpha_attached, alpha.max()]
        return alpha, f
    
    @staticmethod
    def _get_C_fs(
            alpha_f: np.ndarray,
            f: np.ndarray,
            alpha_coeff: np.ndarray,
            coeff: np.ndarray,
            alpha_0: float,
            slope: float
    ):
        i_attach = (f>0).argmax()
        alpha_begin_attachement = alpha_f[i_attach]
        alpha = alpha_coeff[alpha_coeff<alpha_begin_attachement]
        C_fs = coeff[alpha_coeff<alpha_begin_attachement]
        if (f==1).sum() > 1:
            # at: attach process
            i_fully_attached = (f==1).argmax()
            alpha_at = alpha_f[i_attach:i_fully_attached]
            f_at = f[i_attach:i_fully_attached]
            coeff_at = coeff[np.logical_and(alpha_coeff>=alpha_at[0], alpha_coeff<=alpha_at[-1])]
            alpha = np.r_[alpha, alpha_at]
            C_fs = np.r_[C_fs, (coeff_at-slope*(alpha_at-alpha_0)*f_at)/(1-f_at)]
            
            # fa: fully attached
            i_end_fully_attached = (f[i_fully_attached:]<1).argmax()+i_fully_attached
            alpha_begin_fully_attached = alpha_f[i_fully_attached]
            alpha_end_fully_attached = alpha_f[i_end_fully_attached] 
            ids_coeff_fully_attached = np.logical_and(alpha_coeff>=alpha_begin_fully_attached,
                                                    alpha_coeff<=alpha_end_fully_attached)
            alpha = np.r_[alpha, alpha_coeff[ids_coeff_fully_attached]]
            C_fs = np.r_[C_fs, coeff[ids_coeff_fully_attached]/2]

            # dt: detach process
            i_end_attachement = (f[i_end_fully_attached:]==0).argmax()+i_end_fully_attached
            alpha_dt = alpha_f[i_attach+1:i_end_attachement]
            f_dt = f[i_end_fully_attached+1:i_end_attachement]
            coeff_dt = coeff[np.logical_and(alpha_coeff>=alpha_dt[0], alpha_coeff<=alpha_dt[-1])]
            alpha = np.r_[alpha, alpha_dt]
            C_fs = np.r_[C_fs, (coeff_dt-slope*(alpha_dt-alpha_0)*f_dt)/(1-f_dt)]
        else:
            i_end_attachement = (f[i_attach+2:]==0).argmax()+i_attach+2
            alpha_dt = alpha_f[i_attach+1:i_end_attachement]
            f_dt = f[i_attach+1:i_end_attachement]
            coeff_dt = coeff[np.logical_and(alpha_coeff>=alpha_dt[0], alpha_coeff<=alpha_dt[-1])]
            alpha = np.r_[alpha, alpha_dt]
            C_fs = np.r_[C_fs, (coeff_dt-slope*(alpha_dt-alpha_0)*f_dt)/(1-f_dt)]
        
        alpha_end_attachement = alpha_f[i_end_attachement]
        ids_fully_separated = alpha_coeff>=alpha_end_attachement
        alpha_fs = alpha_coeff[ids_fully_separated]
        alpha = np.r_[alpha, alpha_fs]
        C_fs = np.r_[C_fs, coeff[ids_fully_separated]]
        return alpha, C_fs

    
class StructForce(SimulationSubRoutine, Rotations):
    """Class implementing differnt ways to calculate the structural stiffness and damping forces depending on the
    state of the system.
    """
    
    _implemented_schemes = ["linear_xy"]

    _scheme_settings_must = {
        "linear_xy": ["stiffness", "damping", "coordinate_system"],
    }

    _scheme_settings_can = {
        "linear_xy": [],
    }

    _sim_params_required = {
        "linear_xy": [],
    }

    _copy_schemes = {
        "linear_xy": ["linear_ef"]
    }

    def __init__(self, verbose: bool=True) -> None:
        SimulationSubRoutine.__init__(self, verbose=verbose)
    
        self.stiffness = None
        self.damping = None
    
    def get_scheme_method(self, scheme: str) -> Callable:
        scheme_methods = {
            "linear_xy": self._linear_xy,
            "linear_ef": self._linear_ef,
        }
        return scheme_methods[scheme]
    
    def get_scheme_init_method(self, scheme: str) -> Callable:
        scheme_methods = {
            "linear_xy": self._init_linear,
            "linear_ef": self._init_linear,
        }
        return scheme_methods[scheme]

    def _linear_xy(
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
        return self.damping@sim_res.vel[i, :], self.stiffness@sim_res.pos[i, :]
    
    def _linear_ef(
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
        K = rot.T@self.stiffness@rot+self._K_apparent(*np.diag(self.stiffness)[:2], *sim_res.pos[i, :]) 
        C = rot.T@self.damping@rot+self._C_apparent(*np.diag(self.damping)[:2], *sim_res.pos[i, :]) 
        return C@sim_res.vel[i, :], K@sim_res.pos[i, :]
        
    def _init_linear(
            self, 
            sim_res: SimulationResults,
            stiffness: np.ndarray|list,
            damping: np.ndarray|list,
            **kwargs):
        stiffness = stiffness if isinstance(stiffness, np.ndarray) else np.asarray(stiffness)
        damping = damping if isinstance(damping, np.ndarray) else np.asarray(damping)
        if stiffness.shape != damping.shape:
            raise ValueError("Initialising the structural force calculation 'linear' failed because there are "
                             f"a different number of stiffness {stiffness.shape} and damping {damping.shape} values "
                             "given.")
        self.stiffness = stiffness if len(stiffness.shape) > 1 else np.diag(stiffness.flatten())
        self.damping = damping if len(damping.shape) > 1 else np.diag(damping.flatten())

    @staticmethod
    def _K_apparent(k_0: float, k_1: float, x_0: float, x_1: float, x_tors: float) -> float:
        """Calculates the apparent stiffness matrix that the is seen in the inertial x-y coordinate system when rotating
        a body that has its stiffnesses given w.r.t. its rotated coordiante system.

        :param k_0: The stiffness of the first translational degree of freedom in the body-fixed coordinate system. 
        "First" refers to the axis that corresponds to the coordinate of the first entry in a position array.  
        :param k_1: The stiffness of the second translational degree of freedom in the body-fixed coordinate system. 
        "Second" refers to the axis that corresponds to the coordinate of the second entry in a position array. 
        :param x_0: Coordinate on first axis in the inertial x-y coordinate system of where the body is.
        :type x_0: float
        :param x_1: Coordinate on first axis in the inertial x-y coordinate system of where the body is.
        :type x_1: float
        :param x_tors: Coordinate on rotational axis in the inertial x-y coordinate system of where the body is.
        :type x_tors: float
        :return: Apparent stiffness moment
        :rtype: float
        """
        c = np.cos(x_tors)
        s = np.sin(x_tors)
        K_app = np.zeros((3, 3))
        K_app[2, 0] = (k_0-k_1)*(-c*s*x_0+x_1*(c**2-s**2))
        K_app[2, 1] = (k_0-k_1)*c*s*x_1
        return K_app

    @staticmethod
    def _C_apparent(c_0, c_1, x_0: float, x_1: float, x_tors: float) -> float:
        """Calculates the apparent damping matrix that the is seen in the inertial x-y coordinate system when rotating
        a body that has its damping given w.r.t. its rotated coordiante system. 

        :param c_0: The damping of the first translational degree of freedom in the body-fixed coordinate system. 
        "First" refers to the axis that corresponds to the coordinate of the first entry in a position array.  
        :param c_1: The damping of the second translational degree of freedom in the body-fixed coordinate system. 
        "Second" refers to the axis that corresponds to the coordinate of the second entry in a position array. 
        :param x_0: Coordinate on first axis in the inertial x-y coordinate system of where the body is.
        :type x_0: float
        :param x_1: Coordinate on first axis in the inertial x-y coordinate system of where the body is.
        :type x_1: float
        :param x_tors: Coordinate on rotational axis in the inertial x-y coordinate system of where the body is.
        :type x_tors: float
        :param tors_rate: Rotational velocity of the body in the x-y coordinate system.
        :type tors_rate: float
        :return: Apparent damping moment
        :rtype: float
        """
        c = np.cos(x_tors)
        s = np.sin(x_tors)
        tmp_1 = -c*x_0-s*x_1
        tmp_2 = -s*x_0+c*x_1
        C_app = np.zeros((3, 3))
        C_app[0, 2] = c_0*c*tmp_2-c_1*s*tmp_1
        C_app[1, 2] = c_0*s*tmp_2+c_1*c*tmp_1
        C_app[2, 0] = C_app[0, 2]
        C_app[2, 1] = C_app[1, 2]
        C_app[2, 2] = c_0*tmp_2**2+c_1*tmp_1**2
        return C_app


class TimeIntegration(SimulationSubRoutine, Rotations):
    """Class implementing differnt ways to calculate the position, velocity, and acceleration of the system at the
     next time step.
    """
    
    _implemented_schemes = ["eE", "HHT-alpha-xy", "HHT-alpha-xy-adapted"]    
    # eE: explicit Euler
    # HHT_alpha: algorithm as given in #todo add paper

    _scheme_settings_must = {
        "eE": [],
        "HHT-alpha-xy": ["dt", "alpha", "inertia", "damping", "stiffness"],
    }

    _scheme_settings_can = {
        "eE": [],
        "HHT-alpha-xy": [],
    }

    _sim_params_required = {
        "eE": [],
        "HHT-alpha-xy": [],
    } 

    _copy_schemes = {
        "HHT-alpha-xy": ["HHT-alpha-xy-adapted"]
    }

    def __init__(self, verbose: bool=True) -> None:
        self.coordinate_system = "ef"  # this needs to be defined before the __init__ call of the SubRoutines
        SimulationSubRoutine.__init__(self, verbose=verbose)

        self._inertia = None

        self._M_next = None
        self._M_current = None
        self._C_current = None
        self._K_current = None
        self._external_next = None
        self._external_current = None
        self._beta = None
        self._gamma = None
        self._dt = None
        
    def get_scheme_method(self, scheme: str) -> Callable:
        scheme_methods = {
            "eE": self._eE,
            "HHT-alpha-xy": self._HHT_alpha_xy,
            "HHT-alpha-xy-adapted": self._HHT_alpha_xy_adapated,
            "HHT-alpha-increment": self._HHT_alpha_increment,
        }
        return scheme_methods[scheme]
    
    def get_scheme_init_method(self, scheme: str) -> Callable:
        scheme_methods = {
            "eE": self._init_eE,
            "HHT-alpha-xy": self._init_HHT_alpha,
            "HHT-alpha-xy-adapted": self._init_HHT_alpha,
            "HHT-alpha-increment": self._init_HHT_alpha,
        }
        return scheme_methods[scheme]
    
    def _eE(
        self,
        sim_res: ThreeDOFsAirfoil, 
        i: int,
        **kwargs  # the kwargs are needed because of the call in simulate()
    ):
        """Calculates the system's current acceleration and the next position and velocity. Since the loop in 
        ThreeDOFsAirfoil.simulate() assumes the time integration scheme to return the next acceleration, position, and 
        velocity, the current acceleration is already set here. The next acceleration returned to simulate() is np.nan.
        This acceleration value is overwritten in the next iteration loop. 

        :param sim_res: Instance of ThreeDOFsAirfoil. This instance has the state of the system as its attributes.
        :type sim_res: ThreeDOFsAirfoilBase
        :param i: index of current time step
        :type i: int
        :param dt: time step duration
        :type: float
        :return: next position, velocity, and acceleration
        :rtype: np.ndarray
        """
        sim_res.accel[i, :] = self._get_accel(sim_res, i)  # the
        # sim_res.accel[i, :] = [0, 0, sim_res.accel[i, 2]]  # to keep airfoil at a certain (x, y)
        next_vel = sim_res.vel[i, :]+sim_res.accel[i, :]*sim_res.dt[i]
        next_pos = sim_res.pos[i, :]+sim_res.vel[i, :]*sim_res.dt[i]+sim_res.accel[i, :]/2*sim_res.dt[i]**2
        return next_pos, next_vel, np.nan
    
    def _HHT_alpha_xy(
            self,
            sim_res: SimulationResults, 
            i: int,
            dt: float,
            **kwargs):  # the kwargs are needed because of the call in simulate():
        rhs = -self._M_current@sim_res.accel[i, :]-self._C_current@sim_res.vel[i, :]-self._K_current@sim_res.pos[i, :]
        rhs += self._external_next*sim_res.aero[i+1, :]+self._external_current*sim_res.aero[i, :]
        
        accel = solve(self._M_next, rhs, assume_a="sym") 
        vel = sim_res.vel[i, :]+dt*((1-self._gamma)*sim_res.accel[i, :]+self._gamma*accel)
        pos = sim_res.pos[i, :]+dt*sim_res.vel[i, :]+dt**2*((0.5-self._beta)*sim_res.accel[i, :]+self._beta*accel)
        return pos, vel, accel

    def _HHT_alpha_xy_adapated(
            self,
            sim_res: SimulationResults, 
            i: int,
            dt: float,
            **kwargs):  # the kwargs are needed because of the call in simulate():
        rhs = -self._M_current@sim_res.accel[i, :]-self._C_current@sim_res.vel[i, :]-self._K_current@sim_res.pos[i, :]
        rhs += self._external_next*sim_res.aero[i, :]+self._external_current*sim_res.aero[i-1, :]
        
        accel = solve(self._M_next, rhs, assume_a="sym") 
        vel = sim_res.vel[i, :]+dt*((1-self._gamma)*sim_res.accel[i, :]+self._gamma*accel)
        pos = sim_res.pos[i, :]+dt*sim_res.vel[i, :]+dt**2*((0.5-self._beta)*sim_res.accel[i, :]+self._beta*accel)
        return pos, vel, accel

    def _HHT_alpha_increment(
            self,
            sim_res: SimulationResults, 
            i: int,
            dt: float,
            **kwargs):  # the kwargs are needed because of the call in simulate():
        rot = self.passive_3D_planar(sim_res.pos[i, 2])
        rot_t = rot.T

        rot_last = self.passive_3D_planar(sim_res.pos[i-1, 2])
        rot_t_last = rot.T

        self._M_current = 1/(2*self._beta)*self.M-(1-self._alpha)*self.dt*(1-self._gamma/(2*self._beta))*self.C
        self._M_current = rot_t@self._M_current@rot

        self._C_current = 1/(self._beta*self._dt)*self.M+(1-self._alpha)*self._gamma/self._beta*self.C
        self._C_current = rot_t@self._C_current@rot

        M_last = self._alpha*self.dt*(1-self._gamma/(2*self._beta))*self.C
        M_last = rot_t_last@M_last@rot_last

        C_last = self._alpha*self._gamma/self._beta*self.C
        C_last = rot_t_last@C_last@rot_last

        K_step = self._alpha*(self._gamma/(self._beta*self.dt)*self.C+self.K)
        
        d_external_next = sim_res.aero[i+1, :]-sim_res.aero[i, :]
        d_external_current = sim_res.aero[i, :]-sim_res.aero[i-1, :]

        A = 1/(self._dt**2*self._beta)*self.M+(1-self._alpha)*self._gamma/(self._dt*self._beta)*self.C
        A += (1-self._alpha)*self.K
        rhs = self._M_current@sim_res.accel[i, :]+self._C_current@sim_res.vel[i, :]-M_last@sim_res.accel[i-1, :]
        rhs += C_last@sim_res.vel[i, :]-K_step@(sim_res.pos[i, :]-sim_res.pos[i-1, :])
        rhs += self._external_next*d_external_next+self._external_current*d_external_current

        d_x = solve(A, rhs)
        pos = sim_res.pos[i, :]+d_x
        vel = (1-self._gamma/self._beta)*sim_res.vel[i, :]+self._gamma/(self._beta*self._dt)*d_x 

    def _init_eE(
            self,
            sim_res: ThreeDOFsAirfoil,
            inertia: np.ndarray|list,
            **kwargs):
        self._inertia = inertia if isinstance(inertia, np.ndarray) else np.asarray(inertia)
        
    def _init_HHT_alpha(
            self,
            sim_res: ThreeDOFsAirfoil,
            alpha: float,
            dt: float,
            inertia: np.ndarray|list[float],
            damping: np.ndarray|list[float],
            stiffness: np.ndarray|list[float],
            **kwargs): #todo implies constant dt!!!
        self._alpha = alpha
        self._dt = dt
        self._beta = (1+alpha)**2/4
        self._gamma = 0.5+alpha

        inertia = inertia if isinstance(inertia, np.ndarray) else np.asarray(inertia)
        damping = damping if isinstance(damping, np.ndarray) else np.asarray(damping)
        stiffness = stiffness if isinstance(stiffness, np.ndarray) else np.asarray(stiffness)
        
        M = inertia if len(inertia.shape) > 1 else np.diag(inertia.flatten())
        C = damping if len(damping.shape) > 1 else np.diag(damping.flatten())
        K = stiffness if len(stiffness.shape) > 1 else np.diag(stiffness.flatten())

        if K.shape != C.shape:
            raise ValueError("Initialising the time integration for a subclass of an HHT-alpha failed because there "
                             f"are a different number of stiffness {stiffness.shape} and damping {damping.shape} "
                             "values given.")
        if M.shape != C.shape:
            raise ValueError("Initialising the time integration for a subclass of an HHT-alpha failed because there "
                             f"are a different number of inertial {inertia.shape} and damping {damping.shape} "
                             "values given.")
        
        self._M_next = M+dt*(1-alpha)*self._gamma*C+dt**2*(1-alpha)*self._beta*K
        self._M_current = dt*(1-alpha)*(1-self._gamma)*C+dt**2*(1-alpha)*(0.5-self._beta)*K
        self._C_current = C+dt*(1-alpha)*K
        self._K_current = K
        self._external_next = 1-alpha
        self._external_current = alpha

        self.M = M
        self.C = C
        self.K = K
        
    def _get_accel(self, sim_res: ThreeDOFsAirfoil, i: int) -> np.ndarray:
        """Calculates the acceleration for the current time step based on the forces acting on the system.

        :param sim_res: Instance of ThreeDOFsAirfoil. This instance has the state of the system as its attributes.
        :type sim_res: ThreeDOFsAirfoilBase
        :param i: index of current time step
        :type i: int
        :return: acceleration in [x, y, rot z] direction
        :rtype: np.ndarray
        """
        return (sim_res.aero[i, :]-sim_res.damp[i, :]-sim_res.stiff[i, :])/self._inertia


class Oscillations:
    def __init__(
            self,
            inertia: np.ndarray,
            natural_frequency: np.ndarray=None,
            damping_coefficient: np.ndarray=None,
            stiffness: np.ndarray=None,
            damping: np.ndarray=None
            ) -> None:
        self._M = inertia 
        if natural_frequency is not None and damping_coefficient is not None:
            self._K = natural_frequency**2*self._M
            self._C = damping_coefficient*2*np.sqrt(self._M*self._K)
        elif stiffness is not None and damping is not None:
            self._K = stiffness
            self._C = damping
        else:
            raise ValueError("Either 'natural frequency' and 'damping_coefficient' or 'stiffness' and 'damping' "
                             "has to be given.")

        self._zeta = self._C/(2*np.sqrt(self._M*self._K))
        self._omega_n = np.sqrt(self._K/self._M)
        self._omega_d = self._omega_n*np.sqrt(1-self._zeta**2)

    def undamped(self, t: np.ndarray, x_0: np.ndarray=1):
        return x_0*np.cos(self._omega_n*t)
    
    def damped(self, t: np.ndarray, x_0: np.ndarray=1, v_0: np.ndarray=0):
        # Theory of Vibration with Applications 2.6-16
        delta = self._zeta*self._omega_n
        return np.exp(-delta*t)*(x_0*np.cos(self._omega_d*t)+(v_0+delta*x_0)/self._omega_d*np.sin(self._omega_d*t))
    
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

    def rotation(self, T:float, dt: float, force: np.ndarray, x_0: np.ndarray, v_0: np.ndarray):
        t_sim = np.linspace(0, T, 10*int(T/dt))
        pos = np.zeros((t_sim.size, 3))
        vel = np.zeros((t_sim.size, 3))
        accel = np.zeros((t_sim.size, 3))
        # force = np.repeat([force], repeats=int(T/dt), axis=0)
        force = np.zeros((t_sim.size, 3))

        pos[0, :] = x_0
        vel[0, :] = v_0
        dt = t_sim[1]
        for i in range(int(t_sim.size-1)):
            f_stiffness = self._K@pos[i, :]
            f_damping = self._C@vel[i, :]
            accel[i, :] = (force[i, :]-f_stiffness-f_damping)/self._M
            vel[i+1, :] = vel[i, :]+accel[i, :]*dt
            pos[i+1, :] = pos[i, :]+(vel[i, :]+vel[i+1, :])/2*dt
        return t_sim, pos, vel, accel
