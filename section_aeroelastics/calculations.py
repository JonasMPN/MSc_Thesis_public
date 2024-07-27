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
                tors_angle = -base_force*chord*C_m(alpha)/stiffness[2]
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
            aoa = inflow_angle-tors_angle
            # aero forces in x and y
            rot = self.passive_2D(-inflow_angle)
            f_aero_xy = (rot@np.c_[[base_force*C_d(aoa)], [base_force*C_l(aoa)]].T).flatten()
            f_aero_xy[0] = f_aero_xy[0] if x else 0 
            f_aero_xy[1] = f_aero_xy[1] if y else 0 
            f_aero_tors = -base_force*chord*C_m(aoa) if torsion else 0
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
            rot = self.passive_2D(-inflow_angle)
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

        :param scheme: Existing schemes are "steady", "quasi_steady", and "BL_chinese". See details in class AeroForce., defaults to "quasi_steady"
        :type scheme: str, optional
        :raises NotImplementedError: If a scheme is wanted that is not implemented.
        """
        self._aero_scheme_settings = kwargs
        AF = AeroForce(dir_polar=dir_polar, file_polar=file_polar, verbose=self.verbose)
        self._added_sim_params[self._dfl_filenames["f_aero"]] = AF.get_additional_params(scheme)
        self._aero_force = AF.prepare_and_get_scheme(scheme, self, "set_aero_calc", **kwargs)
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
        self._struct_force = SF.prepare_and_get_scheme(scheme, self, "set_struct_calc", **kwargs)
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
        self._time_integration = TI.prepare_and_get_scheme(scheme, self, "set_time_integration", **kwargs)
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
    _implemented_schemes = ["quasi_steady", "BL_chinese", "BL_openFAST_Cl_disc", "BL_Staeblein", 
                            "BL_openFAST_Cl_disc_f_scaled"]

    _scheme_settings_must = {
        "quasi_steady": [],
        "BL_chinese": ["A1", "A2", "b1", "b2"],
        "BL_openFAST_Cl_disc": ["A1", "A2", "b1", "b2"],
        "BL_Staeblein": ["A1", "A2", "b1", "b2"],
        "BL_openFAST_Cl_disc_f_scaled": ["A1", "A2", "b1", "b2"],
    }

    _scheme_settings_can = {
        "quasi_steady": ["density", "chord", "pitching_around", "alpha_at"],
        "BL_chinese": ["density", "chord", "a", "K_alpha", "T_p", "T_bl", "Cn_vortex_detach", "tau_vortex_pure_decay", 
                       "T_v", "pitching_around", "alpha_at"],
        "BL_openFAST_Cl_disc": ["density", "chord", "T_p", "T_f", "pitching_around", "alpha_at"],
        "BL_Staeblein": ["density", "chord", "T_p", "T_f", "pitching_around", "alpha_at"],
        "BL_openFAST_Cl_disc_f_scaled": ["density", "chord", "T_p", "T_f", "pitching_around", "alpha_at"],
    }

    _sim_params_required = {
        "quasi_steady": ["alpha_qs"],
        "BL_chinese": ["s", "alpha_qs", "X_lag", "Y_lag", "alpha_eff", "C_nc", "D_i", "C_ni", "C_npot", "C_tpot", 
                       "D_p", "C_nsEq", "alpha_sEq", "f_t_Dp", "f_n_Dp", "D_bl_t", "D_bl_n", "f_t", "f_n", "C_nf", "C_tf", "tau_vortex", "C_nv_instant", "C_nv", "C_mqs", "C_mnc"],
        "BL_openFAST_Cl_disc": ["alpha_qs", "alpha_eff", "x1", "x2", "x3", "x4", "C_lpot", "C_lc", "C_lnc", "C_ds", 
                                "C_dc", "C_dsep", "C_ms", "C_mnc", "f_steady", "alpha_eq", "C_dus", "C_lus", "C_mus"],
        "BL_Staeblein": ["alpha_qs", "alpha_eff", "x1", "x2", "C_lc", "rel_inflow_speed", "rel_inflow_accel",
                "C_liner", "C_lcent", "C_ds", "C_dind", "C_ms", "C_lift", "C_miner", "C_l"],
        "BL_openFAST_Cl_disc_f_scaled": ["alpha_qs", "alpha_eff", "x1", "x2", "x3", "x4", "C_lpot", "C_lc", "C_lnc", 
                                      "C_ds", "C_dc", "C_dsep", "C_ms", "C_mnc", "f_steady", "T_u", "alpha_eq", 
                                      "C_dus", "C_lus", "C_mus", "rel_inflow_speed"],
    }

    # _copy_scheme = {
    #     "BL_chinese": ["BL_openFAST_Cl_disc"]
    # }
    
    def __init__(self, dir_polar: str, file_polar: str="polars.dat", verbose: bool=True) -> None:
        """Initialises an instance.

        :param file_polar_data: Path to the file from the current working directory containing the polar data.
        The file must contain the columns ["alpha", "C_d", "C_l", "C_m"].
        :type file_polar_data: str
        """
        SimulationSubRoutine.__init__(self, verbose=verbose)

        self.dir_polar = dir_polar
        self.df_polars = pd.read_csv(join(dir_polar, file_polar), delim_whitespace=True)
        self._C_d_0 = self.df_polars["C_d"].min()
        
        self.C_d = interpolate.interp1d(self.df_polars["alpha"], self.df_polars["C_d"])
        self.C_l = interpolate.interp1d(self.df_polars["alpha"], self.df_polars["C_l"])
        self.C_m = interpolate.interp1d(self.df_polars["alpha"], self.df_polars["C_m"])

        self._alpha_0L = None
        self._alpha_0N = None
        self._alpha_0D = self.df_polars["alpha"].iat[self.df_polars["C_d"].argmin()]

        self._C_l_slope = None
        self._C_n_slope = None

        self._f_n = None
        self._f_l = None
        self._f_t = None

        self._C_fs = None

        self._state_derivatives = None

    def get_scheme_method(self, scheme: str) -> Callable:
        scheme_methods = {
            "quasi_steady": self._quasi_steady,
            "BL_chinese": self._BL_chinese,
            "BL_openFAST_Cl_disc": self._BL_openFAST_Cl_disc,
            "BL_Staeblein": self._BL_Staeblein,
            "BL_openFAST_Cl_disc_f_scaled": self._BL_openFAST_Cl_disc_f_scaled,
        }
        return scheme_methods[scheme]
    
    def get_scheme_init_method(self, scheme: str) -> Callable:
        scheme_methods = {
            "quasi_steady": self._init_quasi_steady,
            "BL_chinese": self._init_BL_chinese,
            "BL_openFAST_Cl_disc": self._init_BL_openFAST_Cl_disc,
            "BL_Staeblein": self._init_BL_Staeblein,
            "BL_openFAST_Cl_disc_f_scaled": self._init_BL_openFAST_f_scaled,
        }
        return scheme_methods[scheme]

    def _pre_calculations(
            self, 
            scheme: str,
            resolution: int=1000, 
            sep_points_scheme: int=None,
            adjust_attached: bool=True,
            adjust_separated: bool=True,
            dir_save_to: str=None):
        match scheme:
            case "BL_chinese"|"BL_openFAST_Cl_disc"|"BL_openFAST_Cl_disc_f_scaled":
                # self._C_l_slope = 7.15
                # self._alpha_0L = 0
                # self._C_fs = lambda x: 0
                # self._f_l = lambda x: 1
                # return

                map_sep_point_scheme = {
                    "BL_chinese": 3,
                    "BL_openFAST_Cl_disc": 4,
                    "BL_openFAST_Cd": 5,
                    "BL_openFAST_Cl_disc_f_scaled": 4
                }
                if sep_points_scheme is None:
                    sep_points_scheme = map_sep_point_scheme[scheme]
                if dir_save_to is None:
                    dir_BL_data = helper.create_dir(join(self.dir_polar, "preparation", scheme))[0]
                else:
                    dir_BL_data = helper.create_dir(dir_save_to)[0]
                f_zero_data = join(dir_BL_data, "zero_data.json")
                
                alpha_given = self.df_polars["alpha"]
                #todo: if resolution<=polar resolution around alpha_0, C_l_slope and alpha_0L are not correct anymore;
                #todo if the polar is not resolved by equidistant points further care needs to be taken!
                alpha_interp = np.linspace(alpha_given.min(), alpha_given.max(), resolution)
                coefficients = np.c_[self.C_d(alpha_interp)-self._C_d_0, self.C_l(alpha_interp)]
                if not isfile(f_zero_data):
                    coeff_rot = self.project_2D(coefficients, -np.deg2rad(alpha_interp))
                    coeffs = {"C_l": coefficients[:, 1], "C_n": coeff_rot[:, 1]}
                    alphas = {"C_l": alpha_interp, "C_n": alpha_interp}
                    # coeffs = {"C_l": self.C_l(self.df_polars["alpha"].to_numpy()), "C_n": coeff_rot[:, 1]}
                    # alphas = {"C_l": self.df_polars["alpha"].to_numpy(), "C_n": alpha_interp}
                    add = {"alpha_0_d": self._alpha_0D}
                    zero_data = self._zero_forces(alpha=alphas, coeffs=coeffs, save_as=f_zero_data, add=add)
                else:
                    with open(f_zero_data, "r") as f:
                        zero_data = json.load(f)

                self._alpha_0L = np.deg2rad(zero_data["alpha_0_l"])
                self._alpha_0N = np.deg2rad(zero_data["alpha_0_n"])
                self._C_l_slope = np.rad2deg(zero_data["C_l_slope"])
                self._C_n_slope = np.rad2deg(zero_data["C_n_slope"])

                f_sep = join(dir_BL_data, f"sep_points_{sep_points_scheme}.dat")
                #todo could write something like the below; currently doesn't work because the file name is
                #todo apadated in _write_and_get_separation_points()
                sep_data = self._write_and_get_separation_points(f_sep, alpha_interp, sep_points_scheme,
                                                                 adjust_attached, adjust_separated)
                # if not isfile(f_sep):
                #     sep_data = self._write_and_get_separation_points(f_sep, alpha_interp, sep_points_scheme,
                #                                                      adjust_attached, adjust_separated)
                # else:
                #     sep_data = pd.read_csv(f_sep)

                if "f_n" in sep_data:
                    self._f_n = interpolate.interp1d(sep_data["alpha_n"], sep_data["f_n"])
                if "f_t" in sep_data:
                    self._f_t = interpolate.interp1d(sep_data["alpha_t"], sep_data["f_t"])
                if "f_l" in sep_data:
                    self._f_l = interpolate.interp1d(sep_data["alpha_l"], sep_data["f_l"])

                if "BL_openFAST" in scheme:  # calculate fully separated polar
                    for param in sep_data.keys():
                        direction = param[-1]
                        save_as = join(dir_BL_data, f"C_fs_{sep_points_scheme}_{direction}.dat")
                        # if isfile(save_as):
                        #     df_fs = pd.read_csv(save_as)
                        # else:
                        if True:
                            if "alpha" in param or param == "f_t":
                                continue
                            raoa = np.deg2rad(alpha_interp)
                            if param == "f_n":
                                coeff = self.C_l(alpha_interp)*np.cos(raoa)+(self.C_d(alpha_interp)-self._C_d_0)*np.sin(raoa)
                                alpha_0 = self._alpha_0N
                                slope = self._C_n_slope
                            elif param == "f_l":
                                coeff = self.C_l(alpha_interp)
                                alpha_0 = self._alpha_0L
                                slope = self._C_l_slope
                            else:
                                raise NotImplementedError(f"For param '{param}'")
                            df_fs = self._C_l_fs(sep_data[f"alpha_{direction}"], sep_data[param], alpha_interp, coeff,
                                                 alpha_0, slope, save_as)
                        self._C_fs = interpolate.interp1d(df_fs["alpha"], df_fs["C_fs"]) 
            case "BL_Staeblein":
                # self._C_l_slope = np.rad2deg(0.12493650014276078)
                # self._alpha_0L = np.deg2rad(-3.0253288636075264)
                self._C_l_slope = 7.15
                self._alpha_0L = 0

    def _write_and_get_separation_points(
            self,
            save_as: str,
            alpha_interp: np.ndarray,
            scheme: int=0,
            adjust_attached: bool=True,
            adjust_separated: bool=True
            ) -> pd.DataFrame:
        """_summary_

        :param alpha_0: _description_
        :type alpha_0: float
        :param save_as: _description_
        :type save_as: str
        :param res: _description_, defaults to 100
        :type res: int, optional
        :param scheme: 
        1: X. Liu, uses f_t and f_n, uses sin() in both
        2: Bangga, uses f_n to get C_n and C_t, uses linear a-a_0 
        3: Chinese paper, uses f_t and f_n, f_n based on linear, f_t based on linear*tan
        4: openFAST, HAWC2, f_l, based on linear, 
        , defaults to 0
        5: openFAST, HAWC2, f_n, based on linear
        :type scheme: int, optional
        :param limits: _description_, defaults to (None, None)
        :type limits: tuple, optional
        :return: _description_
        :rtype: pd.DataFrame
        """
        data = {}
        
        match scheme:
            case 1:
                def sqrt_of_f_t(alpha):
                    raoa = np.deg2rad(alpha)
                    C_t = self.C_l(alpha)*np.sin(raoa)-(self.C_d(alpha)-self._C_d_0)*np.cos(raoa)
                    return C_t/(self._C_n_slope*np.sin(raoa-self._alpha_0N)**2)
                
                def sqrt_of_f_n(alpha):
                    raoa = np.deg2rad(alpha)
                    C_n = self.C_l(alpha)*np.cos(raoa)+(self.C_d(alpha)-self._C_d_0)*np.sin(raoa)
                    return 2*np.sqrt(C_n/(self._C_n_slope*np.sin(raoa-self._alpha_0N)))-1

                data["alpha_t"], data["f_t"] = self._adjust_f(alpha_interp, sqrt_of_f_t, adjust_attached, False)
                data["alpha_n"], data["f_n"] = self._adjust_f(alpha_interp, sqrt_of_f_n, adjust_attached, 
                                                              adjust_separated)
            case 2:
                def sqrt_of_f_n(alpha):
                    raoa = np.deg2rad(alpha)
                    C_n = self.C_l(alpha)*np.cos(raoa)+(self.C_d(alpha)-self._C_d_0)*np.sin(raoa)
                    return 2*np.sqrt(C_n/(self._C_n_slope*(raoa-self._alpha_0N)))-1
                
                data["alpha_n"], data["f_n"] = self._adjust_f(alpha_interp, sqrt_of_f_n, adjust_attached, 
                                                              adjust_separated)
            case 3:
                def sqrt_of_f_t(alpha):
                    raoa = np.deg2rad(alpha)
                    C_t = self.C_l(alpha)*np.sin(raoa)-(self.C_d(alpha)-self._C_d_0)*np.cos(raoa)
                    return C_t/(self._C_n_slope*(raoa-self._alpha_0N)*np.tan(raoa))
                
                def sqrt_of_f_n(alpha):
                    raoa = np.deg2rad(alpha)
                    C_n = self.C_l(alpha)*np.cos(raoa)+(self.C_d(alpha)-self._C_d_0)*np.sin(raoa)
                    return 2*np.sqrt(C_n/(self._C_n_slope*(raoa-self._alpha_0N)))-1
                
                data["alpha_t"], data["f_t"] = self._adjust_f(alpha_interp, sqrt_of_f_t, adjust_attached, 
                                                              False)
                data["alpha_n"], data["f_n"] = self._adjust_f(alpha_interp, sqrt_of_f_n, adjust_attached, 
                                                              adjust_separated)
            case 4:
                def sqrt_of_f(alpha):
                    raoa = np.deg2rad(alpha)
                    return 2*np.sqrt(self.C_l(alpha)/(self._C_l_slope*(raoa-self._alpha_0L)))-1
                
                data["alpha_l"], data["f_l"] = self._adjust_f(alpha_interp, sqrt_of_f, adjust_attached, 
                                                              adjust_separated)
            case 5:
                def sqrt_of_f(alpha):
                    raoa = np.deg2rad(alpha)
                    C_n = self.C_l(alpha)*np.cos(raoa)+(self.C_d(alpha)-self._C_d_0)*np.sin(raoa)
                    return 2*np.sqrt(C_n/(self._C_n_slope*(raoa-self._alpha_0N)))-1
                data["alpha_n"], data["f_n"] = self._adjust_f(alpha_interp, sqrt_of_f, adjust_attached, 
                                                              adjust_separated)
        
        for param, values in data.items():
            if "alpha" in param or values is None:
                continue
            idx_last_dot = save_as.rfind(".")
            file_name = save_as[:idx_last_dot]+f"_{param}"+save_as[idx_last_dot:]
            direction = param[-1]
            df = pd.DataFrame({f"alpha_{direction}": data[f"alpha_{direction}"], f"{param}": values})
            df.to_csv(file_name, index=None)
        return data
    
    def _adjust_f(
            self,
            alpha: np.ndarray,
            sqrt_of_f: Callable,
            adjust_attached: bool=True,
            adjust_separated: bool=True,
            keep_sign: bool=True,
            attached_region: tuple=(-60, 60)
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
    def _C_l_fs(
            alpha_f: np.ndarray,
            f: np.ndarray,
            alpha_coeff: np.ndarray,
            coeff: np.ndarray,
            alpha_0: float,
            slope: float,
            save_as: str
    ):
        i_attach = (f>0).argmax()
        alpha_begin_attachement = alpha_f[i_attach]
        alpha = alpha_coeff[alpha_coeff<alpha_begin_attachement]
        C_fs = coeff[alpha_coeff<alpha_begin_attachement]
        
        # at: attach process
        i_fully_attached = (f==1).argmax()
        alpha_at = alpha_f[i_attach:i_fully_attached]
        f_at = f[i_attach:i_fully_attached]
        coeff_at = coeff[np.logical_and(alpha_coeff>=alpha_at[0], alpha_coeff<=alpha_at[-1])]
        alpha = np.r_[alpha, alpha_at]
        C_fs = np.r_[C_fs, (coeff_at-np.deg2rad(slope)*(alpha_at-np.rad2deg(alpha_0))*f_at)/(1-f_at)]
        
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
        alpha_dt = alpha_f[i_end_fully_attached+1:i_end_attachement]
        f_dt = f[i_end_fully_attached+1:i_end_attachement]
        coeff_dt = coeff[np.logical_and(alpha_coeff>=alpha_dt[0], alpha_coeff<=alpha_dt[-1])]
        alpha = np.r_[alpha, alpha_dt]
        C_fs = np.r_[C_fs, (coeff_dt-np.deg2rad(slope)*(alpha_dt-np.rad2deg(alpha_0))*f_dt)/(1-f_dt)]
        
        alpha_end_attachement = alpha_f[i_end_attachement]
        ids_fully_separated = alpha_coeff>=alpha_end_attachement
        alpha_fs = alpha_coeff[ids_fully_separated]
        alpha = np.r_[alpha, alpha_fs]
        C_fs = np.r_[C_fs, coeff[ids_fully_separated]]
        df = pd.DataFrame({"alpha": alpha, "C_fs": C_fs})
        df.to_csv(save_as, index=None)
        return df

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
        alpha_qs_deg = np.rad2deg(sim_res.alpha_qs[i])
        coefficients = np.asarray([self.C_d(alpha_qs_deg), self.C_l(alpha_qs_deg), -self.C_m(alpha_qs_deg)*chord])
        return dynamic_pressure*chord*rot@coefficients
    
    @staticmethod
    def _init_quasi_steady(sim_res:ThreeDOFsAirfoil, **kwargs):
        pass

    def _BL_chinese(self, 
            sim_res: ThreeDOFsAirfoil,
            i: int,
            A1: float, A2: float, b1: float, b2: float,
            chord: float=1, density: float=1.225, pitching_around: float=0.25, alpha_at: float=0.75,
            a: float=343, K_alpha: float=0.75,
            T_p: float=1.5, T_bl: float=5,
            Cn_vortex_detach: float=1.0093, tau_vortex_pure_decay: float=11, T_v: float=5):
        sim_res.alpha_steady[i] = -sim_res.pos[i, 2]+sim_res.inflow_angle[i]

        # --------- MODULE unsteady attached flow
        flow_vel_last = np.sqrt(sim_res.inflow[i-1, :]@sim_res.inflow[i-1, :].T)
        ds_last = 2*sim_res.dt[i-1]*flow_vel_last/chord  #todo this should probably be the total velocity
        sim_res.s[i] = sim_res.s[i-1]+ds_last
        flow_vel = np.sqrt(sim_res.inflow[i, :]@sim_res.inflow[i, :].T)

        qs_flow_angle, _, _ = self._quasi_steady_flow_angle(sim_res.vel[i, :], sim_res.pos[i, :], sim_res.inflow[i, :], 
                                                            chord, pitching_around, alpha_at)
        sim_res.alpha_qs[i] = -sim_res.pos[i, 2]+qs_flow_angle
        d_alpha_qs = sim_res.alpha_qs[i]-sim_res.alpha_qs[i-1]
        sim_res.X_lag[i] = sim_res.X_lag[i-1]*np.exp(-b1*ds_last)+d_alpha_qs*A1*np.exp(-0.5*b1*ds_last)
        sim_res.Y_lag[i] = sim_res.Y_lag[i-1]*np.exp(-b2*ds_last)+d_alpha_qs*A2*np.exp(-0.5*b2*ds_last)
        sim_res.alpha_eff[i] = sim_res.alpha_qs[i]-sim_res.X_lag[i]-sim_res.Y_lag[i]
        sim_res.C_nc[i] = self._C_n_slope*(sim_res.alpha_eff[i]-self._alpha_0N)  #todo check that this sin() is
        #todo also correct for high aoa in potential flow

        # impulsive (non-circulatory) normal force coefficient
        tmp = -a*sim_res.dt[i]/(K_alpha*chord)
        tmp_2 = -(sim_res.vel[i, 2]-sim_res.vel[i-1, 2])
        sim_res.D_i[i] = sim_res.D_i[i-1]*np.exp(tmp)+tmp_2*np.exp(0.5*tmp)
        sim_res.C_ni[i] = 4*K_alpha*chord/flow_vel*(-sim_res.vel[i, 2]-sim_res.D_i[i])

        # add circulatory and impulsive
        sim_res.C_npot[i] = sim_res.C_nc[i]+sim_res.C_ni[i]
        sim_res.C_tpot[i] = sim_res.C_npot[i]*np.tan(sim_res.alpha_eff[i])

        # --------- MODULE nonlinear trailing edge separation
        #todo why does the impulsive part of the potential flow solution go into the lag term?
        sim_res.D_p[i] = sim_res.D_p[i-1]*np.exp(-ds_last/T_p)
        sim_res.D_p[i] += (sim_res.C_npot[i]-sim_res.C_npot[i-1])*np.exp(-0.5*ds_last/T_p)
        if i == 0: 
            sim_res.D_p[i] = 0
        sim_res.C_nsEq[i] = sim_res.C_npot[i]-sim_res.D_p[i]
        sim_res.alpha_sEq[i] = sim_res.C_nsEq[i]/self._C_n_slope+self._alpha_0N
        
        # sim_res.f_t_Dp[i] = self._f_t(np.rad2deg(sim_res.alpha_sEq[i]))
        aoa = np.rad2deg(sim_res.alpha_sEq[i])
        coefficients = np.c_[self.C_d(aoa)-self._C_d_0, self.C_l(aoa)]

        rot = self.passive_2D(-sim_res.alpha_sEq[i])
        coeff_rot = rot@coefficients.T
        C_t, C_n = -coeff_rot[0], coeff_rot[1]

        sim_res.f_t_Dp[i] = self._f_t(np.rad2deg(sim_res.alpha_sEq[i]))
        sim_res.f_n_Dp[i] = self._f_n(np.rad2deg(sim_res.alpha_sEq[i]))

        sim_res.D_bl_t[i] = sim_res.D_bl_t[i-1]*np.exp(-ds_last/T_bl)+\
            (sim_res.f_t_Dp[i]-sim_res.f_t_Dp[i-1])*np.exp(-0.5*ds_last/T_bl)
        sim_res.D_bl_n[i] = sim_res.D_bl_n[i-1]*np.exp(-ds_last/T_bl)+\
            (sim_res.f_n_Dp[i]-sim_res.f_n_Dp[i-1])*np.exp(-0.5*ds_last/T_bl)
        if i == 0:
            sim_res.D_bl_t[i] = 0
            sim_res.D_bl_n[i] = 0
        
        sim_res.f_t[i] = sim_res.f_t_Dp[i]-sim_res.D_bl_t[i]
        sim_res.f_n[i] = sim_res.f_n_Dp[i]-sim_res.D_bl_n[i]

        C_nqs = sim_res.C_nc[i]*((1+np.sign(sim_res.f_n[i])*np.sqrt(np.abs(sim_res.f_n[i])))/2)**2
        sim_res.C_nf[i] = C_nqs+sim_res.C_ni[i]
        sim_res.C_tf[i] = self._C_n_slope*(sim_res.alpha_eff[i]-self._alpha_0N)*np.tan(sim_res.alpha_eff[i])*\
            np.sign(sim_res.f_t[i])*np.sqrt(np.abs(sim_res.f_t[i]))
        
        # --------- MODULE leading-edge vortex position
        if sim_res.C_nsEq[i] >= Cn_vortex_detach or d_alpha_qs < 0:
            sim_res.tau_vortex[i] = sim_res.tau_vortex[i-1]+0.45*ds_last
        else:
            sim_res.tau_vortex[i] = 0

        # --------- MODULE leading-edge vortex lift
        sim_res.C_nv_instant[i] = sim_res.C_nc[i]*(1-((1+np.sign(sim_res.f_n[i])*np.sqrt(np.abs(sim_res.f_n[i])))/2)**2)
        sim_res.C_nv[i] = sim_res.C_nv[i-1]*np.exp(-ds_last/T_v)
        if sim_res.tau_vortex[i] < tau_vortex_pure_decay:
            if i != 0:
                sim_res.C_nv[i] += (sim_res.C_nv_instant[i]-sim_res.C_nv_instant[i-1])*np.exp(-0.5*ds_last/T_v)
        

        # --------- MODULE moment coefficient
        rel_speed = np.sqrt((sim_res.inflow[i, 0]-sim_res.vel[i, 0])**2+
                            (sim_res.inflow[i, 1]-sim_res.vel[i, 1])**2)
        sim_res.C_mqs[i] = self.C_m(np.rad2deg(sim_res.alpha_eff[i]))
        sim_res.C_mnc[i] = -0.5*np.pi*chord*(alpha_at-pitching_around)/rel_speed*(-sim_res.vel[i, 2])
        C_m = sim_res.C_mqs[i]+sim_res.C_mnc[i]

        # --------- Combining everything
        # for return of [C_d, C_l, C_m]
        coefficients = np.asarray([sim_res.C_tf[i], sim_res.C_nf[i]+sim_res.C_nv[i], C_m])
        rot = self.passive_3D_planar(sim_res.pos[i, 2]+sim_res.inflow_angle[i])
        coeffs = rot@coefficients-np.asarray([self._C_d_0, 0, 0])
        coeffs[0] = -coeffs[0] 
        return coeffs

        # for return of [f_x, f_y, mom]
        # C_t and C_d point in opposite directions!
        # coefficients = np.asarray([sim_res.C_tf[i], sim_res.C_nf[i]+sim_res.C_nv[i], -C_m*chord])
        # rot = self.passive_3D_planar(-sim_res.pos[i, 2]) 
        # dynamic_pressure = density/2*rel_speed**2
        # forces = dynamic_pressure*chord*rot@coefficients  # for [-f_x, f_y, mom]
        # forces[0] *= -1
        # return forces  # for [f_x, f_y, mom]

    def _BL_openFAST_Cl_disc(
            self, 
            sim_res: SimulationResults,
            i: int,
            A1: float, A2: float, b1: float, b2: float,
            chord: float=1, density: float=1.225, pitching_around: float=0.5, alpha_at: float=0.75,
            T_p: float=1.5, T_f: float=6):
        lcs = {param: value for param, value in locals().items() if param != "self"}
        lcs["C_slope"] = self._C_l_slope
        lcs["alpha_0"] = self._alpha_0L
        return self._BL_openFAST_disc(**lcs)
    
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
        sim_res.alpha_qs[i] = -sim_res.pos[i, 2]+qs_flow_angle
        T_u_current = 0.5*chord/np.sqrt(v_x**2+v_y**2)
        
        _, v_x_last, v_y_last = self._quasi_steady_flow_angle(sim_res.vel[i-1, :], sim_res.pos[i-1, :], 
                                                              sim_res.inflow[i-1, :], chord, pitching_around, alpha_at)
        T_u_last = 0.5*chord/np.sqrt(v_x_last**2+v_y_last**2)  #todo not just the wind velocity in the denominator?
        tmp1 = np.exp(-sim_res.dt[i-1]*b1/T_u_last) 
        tmp2 = np.exp(-sim_res.dt[i-1]*b2/T_u_last)
        alpha_qs_avg = 0.5*(sim_res.alpha_qs[i-1]+sim_res.alpha_qs[i])
        sim_res.x1[i] = sim_res.x1[i-1]*tmp1+alpha_qs_avg*A1*(1-tmp1)
        sim_res.x2[i] = sim_res.x2[i-1]*tmp2+alpha_qs_avg*A2*(1-tmp2)

        sim_res.alpha_eff[i] = sim_res.alpha_qs[i]*(1-A1-A2)+sim_res.x1[i]+sim_res.x2[i]
        sim_res.C_lpot[i] = C_slope*(sim_res.alpha_eff[i]-alpha_0)-np.pi*T_u_current*sim_res.vel[i, 2]

        tmp3 = np.exp(-sim_res.dt[i-1]/(T_u_last*T_p))
        sim_res.x3[i] = sim_res.x3[i-1]*tmp3+0.5*(sim_res.C_lpot[i-1]+sim_res.C_lpot[i])*(1-tmp3)
        sim_res.alpha_eq[i] = sim_res.x3[i]/C_slope+alpha_0

        tmp4 = np.exp(-sim_res.dt[i-1]/(T_u_last*T_f))
        sim_res.f_steady[i] = self._f_l(np.rad2deg(sim_res.alpha_eq[i]))
        sim_res.x4[i] = sim_res.x4[i-1]*tmp4+0.5*(sim_res.f_steady[i-1]+sim_res.f_steady[i])*(1-tmp4)

        sim_res.C_lc[i] = sim_res.x4[i]*C_slope*(sim_res.alpha_eff[i]-alpha_0)
        sim_res.C_lc[i] += (1-sim_res.x4[i])*self._C_fs(np.rad2deg(sim_res.alpha_eff[i]))
        sim_res.C_lnc[i] = -np.pi*T_u_current*sim_res.vel[i, 2]
        sim_res.C_lus[i] = sim_res.C_lc[i]+sim_res.C_lnc[i]

        sim_res.C_ds[i] = self.C_d(np.rad2deg(sim_res.alpha_eff[i]))
        tmp = (np.sqrt(sim_res.f_steady[i])-np.sqrt(sim_res.x4[i]))/2-(sim_res.f_steady[i]-sim_res.x4[i])/4
        sim_res.C_dsep[i] = (sim_res.C_ds[i]-self._C_d_0)*tmp
        sim_res.C_dc[i] = (sim_res.alpha_qs[i]-sim_res.alpha_eff[i]-T_u_current*sim_res.vel[i, 2])*sim_res.C_lc[i]
        sim_res.C_dus[i] = sim_res.C_ds[i]+sim_res.C_dc[i]+sim_res.C_dsep[i]

        sim_res.C_ms[i] = self.C_m(np.rad2deg(sim_res.alpha_eff[i]))
        sim_res.C_mnc[i] = 0.5*np.pi*T_u_current*sim_res.vel[i, 2]
        sim_res.C_mus[i] = sim_res.C_ms[i]+sim_res.C_mnc[i]

        # --------- Combining everything
        coeffs = np.asarray([sim_res.C_dus[i], sim_res.C_lus[i], sim_res.C_mus[i]])

        # for return of [C_d, C_l, C_m]
        return coeffs

        # for return of [f_x, f_y, mom]
        # coeffs = np.asarray([sim_res.C_dus[i], sim_res.C_lus[i], -sim_res.C_mus[i]*chord])
        # rel_speed = np.sqrt((sim_res.inflow[i, 0]-sim_res.vel[i, 0])**2+
        #                     (sim_res.inflow[i, 1]-sim_res.vel[i, 1])**2)
        # dynamic_pressure = density/2*rel_speed**2
        # rot = self.passive_3D_planar(-sim_res.alpha_eff[i]-sim_res.pos[i, 2])
        # # print( rel_speed, dynamic_pressure*chord*rot@coeffs)
        # return dynamic_pressure*chord*rot@coeffs
    
    def _BL_openFAST_Cl_disc_f_scaled(
            self, 
            sim_res: SimulationResults,
            i: int,
            A1: float, A2: float, b1: float, b2: float,
            chord: float=1, density: float=1.225, pitching_around: float=0.5, alpha_at: float=0.75,
            T_p: float=1.5, T_f: float=6):
        lcs = {param: value for param, value in locals().items() if param != "self"}
        lcs["C_slope"] = self._C_l_slope
        lcs["alpha_0"] = self._alpha_0L
        return self._BL_openFAST_disc_f_scaled(**lcs)
    
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
        sim_res.f_steady[i] = self._f_l(np.rad2deg(sim_res.alpha_eq[i]))
        sim_res.x4[i] = sim_res.x4[i-1]*tmp4+0.5*(sim_res.f_steady[i-1]+sim_res.f_steady[i])*(1-tmp4)

        sim_res.C_lc[i] = sim_res.x4[i]*C_slope*(sim_res.alpha_eff[i]-alpha_0)
        sim_res.C_lc[i] += (1-sim_res.x4[i])*self._C_fs(np.rad2deg(sim_res.alpha_eff[i]))
        sim_res.C_lnc[i] = -np.pi*sim_res.T_u[i]*sim_res.vel[i, 2]
        sim_res.C_lus[i] = sim_res.C_lc[i]+sim_res.C_lnc[i]

        sim_res.C_ds[i] = self.C_d(np.rad2deg(sim_res.alpha_eff[i]))
        tmp = (np.sqrt(sim_res.f_steady[i])-np.sqrt(sim_res.x4[i]))/2-(sim_res.f_steady[i]-sim_res.x4[i])/4
        sim_res.C_dsep[i] = (sim_res.C_ds[i]-self._C_d_0)*tmp
        sim_res.C_dc[i] = (sim_res.alpha_qs[i]-sim_res.alpha_eff[i]-sim_res.T_u[i]*sim_res.vel[i, 2])*sim_res.C_lc[i]
        sim_res.C_dus[i] = sim_res.C_ds[i]+sim_res.C_dc[i]+sim_res.C_dsep[i]

        sim_res.C_ms[i] = self.C_m(np.rad2deg(sim_res.alpha_eff[i]))
        sim_res.C_mnc[i] = 0.5*np.pi*sim_res.T_u[i]*sim_res.vel[i, 2]
        sim_res.C_mus[i] = sim_res.C_ms[i]+sim_res.C_mnc[i]

        # --------- Combining everything
        coeffs = np.asarray([sim_res.C_dus[i], sim_res.C_lus[i], sim_res.C_mus[i]])

        # for return of [C_d, C_l, C_m]
        return coeffs

        # for return of [f_x, f_y, mom]
        # coeffs = np.asarray([sim_res.C_dus[i], sim_res.C_lus[i], -sim_res.C_mus[i]*chord])
        # rel_speed = np.sqrt((sim_res.inflow[i, 0]-sim_res.vel[i, 0])**2+
        #                     (sim_res.inflow[i, 1]-sim_res.vel[i, 1])**2)
        # dynamic_pressure = density/2*rel_speed**2
        # rot = self.passive_3D_planar(-sim_res.alpha_eff[i]-sim_res.pos[i, 2])
        # # print( rel_speed, dynamic_pressure*chord*rot@coeffs)
        # return dynamic_pressure*chord*rot@coeffs

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
        sim_res.C_lc[i] = self._C_l_slope*(sim_res.alpha_eff[i]-self._alpha_0L)

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
        D_s = base_force*self.C_d(np.rad2deg(sim_res.alpha_eff[i]))
        D_ind = L_c*(sim_res.alpha_qs[i]-sim_res.alpha_eff[i]-T_u_current*sim_res.vel[i, 2])
        # alpha_quasi_geometric = np.arctan2(sim_res.inflow[i, 1]-sim_res.vel[i, 1],
        #                                    sim_res.inflow[i, 0]-sim_res.vel[i, 0])-sim_res.pos[i, 2]
        # D_ind = L_c*(alpha_quasi_geometric-sim_res.alpha_eff[i]-T_u_current*sim_res.vel[i, 2])
        D = D_s+D_ind

        # get moment
        M_s = -base_force*chord*self.C_m(np.rad2deg(sim_res.alpha_eff[i]))  # minus because a positive C_m means nose
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

    def _init_BL_chinese(
            self,
            sim_res: SimulationResults,
            chord: float=1,
            pitching_around: float=0.25,
            alpha_at: float=0.75,
            **kwargs
            ):
        #todo make proper comment later, for now:
        # in the simulation loop, all values except for the position, velocity, and acceleration of the airfoil
        # are calculated for the first time step. However, in each simulation loop, the difference of some parameters
        # (the ones below) needs to be caluculated w.r.t. to the previous time step. At the initial time step, these
        # differences should be zero. Since the simulation calculates the below parameters for the first time step,
        # the values need to be set for the one before that. In numpy, this is the same as the value of the very last 
        # time step. Then, the simulation will calculate the difference of dparam = param[0]-param[-1] = 0.
        # Only if param[0] != 0 must param[-1] be initialised, since param[-1] = 0 already.
        qs_flow_angle, _, _ = self._quasi_steady_flow_angle(sim_res.vel[0, :], sim_res.pos[0, :], sim_res.inflow[0, :], 
                                                      chord, pitching_around, alpha_at)
        # sim_res.alpha_qs[-1] = -sim_res.pos[0, 2]+qs_flow_angle
        # sim_res.vel[-1, 2] = sim_res.vel[0, 2]

    def _init_BL_openFAST_Cl_disc(
            self,
            sim_res: SimulationResults,
            chord: float=1,
            pitching_around: float=0.25,
            alpha_at: float=0.75,
            **kwargs
            ):
        # currently does not support an initial non-zero velocity of the airfoil
        alpha_steady = -sim_res.pos[0, 2]+sim_res.inflow_angle[0]
        qs_flow_angle, _, _ = self._quasi_steady_flow_angle(sim_res.vel[0, :], sim_res.pos[0, :], 
                                                                sim_res.inflow[0, :], chord, pitching_around, alpha_at)
        alpha_qs = -sim_res.pos[0, 2]+qs_flow_angle
        sim_res.alpha_qs[-1] = alpha_qs
        sim_res.x1[-1] = alpha_qs*kwargs["A1"]
        sim_res.x2[-1] = alpha_qs*kwargs["A2"]
        sim_res.C_lpot[-1] = self._C_l_slope*(alpha_qs-self._alpha_0L)
        sim_res.x3[-1] = self._C_l_slope*(alpha_qs-self._alpha_0L)
        sim_res.x4[-1] = self._f_l(np.rad2deg(alpha_steady))
        sim_res.f_steady[-1] = self._f_l(np.rad2deg(alpha_steady))

    def _init_BL_Staeblein(
            self,
            sim_res: SimulationResults,
            chord: float=1,
            pitching_around: float=0.25,
            alpha_at: float=0.75,
            **kwargs
    ):
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

    def _init_BL_openFAST_f_scaled(
            self,
            sim_res: SimulationResults,
            chord: float=1,
            pitching_around: float=0.25,
            alpha_at: float=0.75,
            **kwargs
            ):
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
        sim_res.C_lpot[-1] = self._C_l_slope*(alpha_qs-self._alpha_0L)
        sim_res.x3[-1] = self._C_l_slope*(alpha_qs-self._alpha_0L)
        sim_res.x4[-1] = self._f_l(np.rad2deg(alpha_steady))
        sim_res.f_steady[-1] = self._f_l(np.rad2deg(alpha_steady))
        sim_res.T_u[-1] = 2*sim_res.rel_inflow_speed[-1]*sim_res.dt[-1]/chord
    
    def _zero_forces(self, alpha: dict[str, np.ndarray], coeffs: dict[str, np.ndarray], save_as: str, add: dict,
                     alpha0_in: tuple=(-10, 5)):
        zero_data = {}
        for coeff_name, coeff in coeffs.items():
            aoa = alpha[coeff_name]
            ids_subset = np.logical_and(aoa>=alpha0_in[0], aoa<=alpha0_in[1])
            alpha_subset = aoa[ids_subset]
            coeff_subset = coeff[ids_subset]
            find_sign_change = coeff_subset[1:]*coeff_subset[:-1]
            if np.all(find_sign_change>0):
                raise ValueError(f"The given '{coeff_name}' does not have a sign change in the current AoA interval of "
                                f"{alpha0_in} degree. Linear interpolation can thus not find alpha_0.")
            alpha_0, slope = self._get_zero_crossing(alpha_subset, coeff_subset)
            zero_data[f"alpha_0_{coeff_name.split("_")[-1]}"] = alpha_0
            zero_data[f"{coeff_name}_slope"] = slope
        with open(save_as, "w") as f:
            json.dump(zero_data|add, f, indent=4)
        return zero_data

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
    def _get_zero_crossing(x: np.ndarray, y: np.ndarray, gradient: str="pos") -> tuple[float, float]:
        idx_greater_0 = np.argmax(y > 0) if gradient == "pos" else np.argmax(y < 0)
        dy = y[idx_greater_0]-y[idx_greater_0-1]
        dx = x[idx_greater_0]-x[idx_greater_0-1]
        x_0 = x[idx_greater_0]-y[idx_greater_0]*dx/dy
        return x_0, dy/dx

    
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
