import numpy as np
from scipy.linalg import solve
from scipy.optimize import brentq
import pandas as pd
from os.path import join, isdir
from typing import Callable
import json
from helper_functions import Helper
from scipy import interpolate
from calculation_utils import SimulationResults, SimulationSubRoutine, Rotations
helper = Helper()


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
        file_polar: str="polars.dat",
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

        self.dir_polar = dir_polar
        self.file_polar_data = file_polar
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
  
    def set_aero_calc(self, scheme: str="quasi_steady", **kwargs):
        """Sets how the aerodynamic forces are calculated in the simulation. If the scheme is dependet on constants,
        these must be given in the kwargs. Which kwargs are necessary are defined in the match-case statements in 
        this function's implementation; they are denoted "must_haves". These are mandatory. The "can_haves" are 
        constants for which default values are set. They (the "can_haves") can, but do not have to, thus be set.

        :param scheme: Existing schemes are "steady", "quasi_steady", and "BL". See details in class AeroForce., defaults to "quasi_steady"
        :type scheme: str, optional
        :raises NotImplementedError: If a scheme is wanted that is not implemented.
        """
        self._aero_scheme_settings = kwargs
        AF = AeroForce(dir_polar=self.dir_polar, file_polar=self.file_polar_data, verbose=self.verbose)
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

    def simuluate(
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
        dt = np.r_[self.time[1:]-self.time[:-1], self.time[1]]  # todo hard coded additional dt at the end  
        self.set_param(inflow=inflow, inflow_angle=inflow_angle, dt=dt)

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
            init_func(self, **scheme_setting)

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
        files = files if files is not None else {filename: [] for filename in self._dfl_files.keys()}
        if use_default:
            for filename, params in self._added_sim_params.items():
                if params == None:
                    continue
                files[filename] = list(set(files[filename]+params))
        #todo add error if use_default==False and files==None in method call
        self._save(root, files, split, use_default)


class AeroForce(SimulationSubRoutine, Rotations):
    """Class implementing differnt ways to calculate the lift, drag, and moment depending on the
    state of the system.
    """
    _implemented_schemes = ["quasi_steady", "BL", "BL_openFAST"]

    _scheme_settings_must = {
        "quasi_steady": [],
        "BL": ["A1", "A2", "b1", "b2"],
        "BL_openFAST": ["A1", "A2", "b1", "b2"]
    }

    _scheme_settings_can = {
        "quasi_steady": ["density", "chord", "pitching_around", "alpha_at"],
        "BL": ["density", "chord", "a", "K_alpha", "T_p", "T_bl", "Cn_vortex_detach", "tau_vortex_pure_decay", "T_v", 
               "pitching_around", "alpha_at"],
        "BL_openFAST": ["density", "chord"]
    }

    _sim_params_required = {
        "quasi_steady": ["alpha_qs"],
        "BL": ["s", "alpha_qs", "X_lag", "Y_lag", "alpha_eff", "C_nc", "D_i", "C_ni", "C_npot", "C_tpot", "D_p", 
               "C_nsEq", "alpha_sEq", "f_t_Dp", "f_n_Dp", "D_bl_t", "D_bl_n", "f_t", "f_n", "C_nf", "C_tf",
               "tau_vortex", "C_nv_instant", "C_nv", "C_mqs", "C_mnc"],
        "BL_openFAST": []
    }
    
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

    def get_scheme_method(self, scheme: str) -> Callable:
        scheme_methods = {
            "quasi_steady": self._quasi_steady,
            "BL": self._BL,
            "BL_openFAST": self._BL_openFAST
        }
        return scheme_methods[scheme]
    
    def get_scheme_init_method(self, scheme: str) -> Callable:
        scheme_methods = {
            "quasi_steady": self._init_quasi_steady,
            "BL": self._init_BL,
            "BL_openFAST": self._init_BL
        }
        return scheme_methods[scheme]

    def _pre_calculations(self, scheme: str, resolution: int=100, sep_points_scheme: int=0):
        match scheme:
            case "BL"|"BL_openFAST":
                prep_dir = "Beddoes_Leishman_preparation"
                if scheme == "BL_openFAST":
                    prep_dir += "_openFAST"
                dir_BL_data = helper.create_dir(join(self.dir_polar, prep_dir))[0]
                f_zero_data = join(dir_BL_data, "zero_data.json")
                f_sep = join(dir_BL_data, f"sep_points_{sep_points_scheme}.dat")

                if not isfile(f_zero_data):
                    alpha_given = self.df_polars["alpha"]
                    alpha_interp = np.linspace(alpha_given.min(), alpha_given.max(), resolution)
                    coefficients = np.c_[self.C_d(alpha_interp)-self._C_d_0, self.C_l(alpha_interp)]
                    coeff_rot = self.project_2D(coefficients, -np.deg2rad(alpha_interp))
                    coeffs = {"C_l": coefficients[:, 1], "C_n": coeff_rot[:, 1]}
                    zero_data = self._zero_forces(alpha=alpha_interp, coeffs=coeffs, save_as=f_zero_data)
                else:
                    with open(f_zero_data, "r") as f:
                        zero_data = json.load(f)

                self._alpha_0L = np.deg2rad(zero_data["alpha_0_l"])
                self._alpha_0N = np.deg2rad(zero_data["alpha_0_n"])
                self._C_l_slope = np.rad2deg(zero_data["C_l_slope"])
                self._C_n_slope = np.rad2deg(zero_data["C_n_slope"])
                
                # if not isfile(f_sep):
                if True:
                    df_sep = self._write_and_get_separation_points(save_as=f_sep, scheme=sep_points_scheme,
                                                                   res=resolution)
                else:
                    with open(f_alpha_0, "r") as f:
                        self._alpha_0 = np.deg2rad(json.load(f)["alpha_0"])
                    df_sep = pd.read_csv(f_sep)
                
                if "f_n" in df_sep.columns:
                    self._f_n = interpolate.interp1d(df_sep["alpha"], df_sep["f_n"])
                if "f_t" in df_sep.columns:
                    self._f_t = interpolate.interp1d(df_sep["alpha"], df_sep["f_t"])
                if "f_l" in df_sep.columns:
                    self._f_l = interpolate.interp1d(df_sep["alpha"], df_sep["f_l"])

    def _write_and_get_separation_points(
            self,
            save_as: str,
            res: int=100,
            scheme: int=0,
            limits: tuple=(None, None)
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
        5: openFAST< HAWC2, f_n, based on linear
        :type scheme: int, optional
        :param limits: _description_, defaults to (None, None)
        :type limits: tuple, optional
        :return: _description_
        :rtype: pd.DataFrame
        """
        alpha_given = self.df_polars["alpha"]
        alpha_interp = np.linspace(alpha_given.min(), alpha_given.max(), res)
        coefficients = np.c_[self.C_d(alpha_interp)-self._C_d_0, self.C_l(alpha_interp)]
        data = {"alpha": alpha_interp, "f_n": None, "f_t": None, "f_l": None}
        alpha_interp = np.deg2rad(alpha_interp)
        
        match scheme:
            case 1:
                coeff_rot = self.project_2D(coefficients, -alpha_interp)
                C_t, C_n = -coeff_rot[:, 0], coeff_rot[:, 1]

                f_t = C_t/(self._C_n_slope*np.sin(alpha_interp-self._alpha_0N)**2)
                f_t = f_t**2*np.sign(f_t)

                f_n = (2*np.sqrt(C_n/(self._C_n_slope*np.sin(alpha_interp-self._alpha_0N)))-1)**2

                data["f_t"] = f_t
                data["f_n"] = f_n
            case 2:
                coeff_rot = self.project_2D(coefficients, -alpha_interp)
                C_n = coeff_rot[:, 1]

                f_n = (2*np.sqrt(C_n/(self._C_n_slope*(alpha_interp-self._alpha_0N)))-1)**2
                data["f_n"] = f_n
            case 3:
                coeff_rot = self.project_2D(coefficients, -alpha_interp)
                C_t, C_n = -coeff_rot[:, 0], coeff_rot[:, 1]

                f_t = C_t/(self._C_n_slope*np.sin(alpha_interp-self._alpha_0N)*np.tan(alpha_interp))
                f_t = f_t**2*np.sign(f_t)

                f_n = 2*np.sqrt(C_n/(self._C_n_slope*(alpha_interp-self._alpha_0N)))-1
                f_n = f_n**2*np.sign(f_n)

                data["f_t"] = f_t
                data["f_n"] = f_n
            case 4:
                data["alpha"], data["f_l"] = self._write_fn_openFAST("C_l", res)
            case 5:
                data["alpha"], data["f_n"] = self._write_fn_openFAST("C_n", res)

        for f_kind in ["f_n", "f_t", "f_l"]:
            if data[f_kind] is None:
                data.pop(f_kind)
                continue
            f = data[f_kind]
            if limits[0] is not None:
                f[f<=limits[0]] = limits[0]
            
            if limits[1] is not None:
                f[f>=limits[1]] = limits[1]
        
        df = pd.DataFrame(data)
        df.to_csv(save_as, index=None)
        return df
    
    def _write_fn_openFAST(
            self,
            coeff_direction: str,
            res: int=100
    ) -> pd.DataFrame:
        # currently expects the begin of the attachement in [alpha_0-60, alpha_0-0.1]
        # the begin of the fully attache region in [begin_attachement, alpha_0-3]
        # the end of the fully attached region in [alpha_0+3, alpha_0+20]
        # the begin of the fully separated region in [end_fully_attached, alpha_0+70]
        alpha_given = self.df_polars["alpha"].to_numpy()
        alpha_sub = alpha_given[np.logical_and(alpha_given>=-80, alpha_given<=80)]
        
        def res_separated(alpha):
            # the return value squared equals f_n
            raoa = np.deg2rad(alpha)
            if coeff_direction == "C_n":
                coeff = self.C_l(alpha)*np.cos(raoa)+(self.C_d(alpha)-self._C_d_0)*np.sin(raoa)
                return 2*np.sqrt(coeff/(self._C_n_slope*np.sin(raoa-self._alpha_0N)))-1
            elif coeff_direction == "C_l":
                coeff = self.C_l(alpha)
                return 2*np.sqrt(coeff/(self._C_l_slope*np.sin(raoa-self._alpha_0L)))-1
        
        def res_attached(alpha):
            return res_separated(alpha)-1
        
        # import matplotlib.pyplot as plt
        # alpha = np.linspace(-40, 50, 300)
        # fig, ax = plt.subplots()
        # # ax.plot(alpha, res_separated(alpha)**2, label="sep")
        # ax.plot(alpha, res_attached(alpha), label="attached")
        # ax.grid()
        # ax.legend()
        # ax.set_ylim((-0.3, 0.3))
        # fig, ax = plt.subplots()
        # alpha_sub = np.linspace(-40, 40, 300)
        # ax.plot(alpha_sub, self.C_l(alpha_sub), label="polar")
        # ax.plot(alpha_sub, np.deg2rad(self._C_l_slope)*(alpha_sub-np.rad2deg(self._alpha_0L)), label="linear")
        # ax.plot(alpha_sub, self._C_l_slope*np.sin(np.deg2rad(alpha_sub)-self._alpha_0L), label="sin")
        # ax.legend()
        # ax.grid()
        # ax.set_ylim((-2, 2))
        # plt.show()
        
        res_sep = res_separated(alpha_sub)
        id_begin_attachement = (res_sep>0).argmax()
        begin_attachement = brentq(res_separated, alpha_sub[id_begin_attachement-2], alpha_sub[id_begin_attachement])

        res_att = res_sep-1
        id_fully_attached = (res_att>0).argmax()
        begin_fully_attached = brentq(res_attached, alpha_sub[id_fully_attached-2], alpha_sub[id_fully_attached])

        id_end_fully_attached = res_att.size-(res_att[::-1]>0).argmax()-1
        end_fully_attached = brentq(res_attached, alpha_sub[id_end_fully_attached], alpha_sub[id_end_fully_attached+2])

        id_end_attched = res_att.size-(res_sep[::-1]>0).argmax()-1
        end_attachement = brentq(res_separated, alpha_sub[id_end_attched], alpha_sub[id_end_attched+2])

        alpha_begins_attaching = np.linspace(begin_attachement, begin_fully_attached, int(res/2))[:-1]
        alpha_ends_attached = np.linspace(end_fully_attached, end_attachement, int(res/2))[1:]
        f = np.r_[0, res_separated(alpha_begins_attaching)**2, 1, 1, res_separated(alpha_ends_attached)**2, 0]
        alpha = np.r_[alpha_given.min(), alpha_begins_attaching, begin_fully_attached, end_fully_attached, 
                    alpha_ends_attached, alpha_given.max()]
        return alpha, f
    
    def C_l_fs(
            self,
            alpha_n: np.ndarray,
            f_n: np.ndarray,
            alpha_l: np.ndarray,
            f_l: np.ndarray,
            save_as: str
    ):
        data = {}
        for alpha, f, kind in [(alpha_n, f_n, "n"), (alpha_l, f_l, "l")]:
            f_not_ones = f_n != 1
            f_not_zeros = f_n != 0
            ids_subset = np.logical_and(f_not_ones, f_not_zeros)
            alpha = alpha[ids_subset]
            f = f[ids_subset]
            if kind == "n":
                raoa = np.deg2rad(alpha-self._alpha_0L)
                coeff = self.C_l(alpha)*np.cos(raoa)+(self.C_d(alpha)-self._C_d_0)*np.sin(raoa)
            elif kind == "l":
                coeff = self.C_l(alpha)
            C_fs = (coeff-self._C_l_slope*np.sin(np.deg2rad(alpha-self._alpha_0L))*f)/(1-f)
            data[f"alpha_{kind}"] = alpha
            data[f"C_{kind}_fs"] = C_fs
        df = pd.DataFrame(data)
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
        qs_flow_angle = self._quasi_steady_flow_angle(sim_res.vel[i, :], sim_res.pos[i, :], sim_res.inflow[i, :],
                                                      chord, pitching_around, alpha_at)
        
        rot = self.passive_3D_planar(-qs_flow_angle)
        sim_res.alpha_qs[i] = -sim_res.pos[i, 2]+qs_flow_angle

        rel_speed = np.sqrt((sim_res.inflow[i, 0]-sim_res.vel[i, 0])**2+
                            (sim_res.inflow[i, 1]-sim_res.vel[i, 1])**2)

        dynamic_pressure = density/2*rel_speed
        alpha_qs_deg = np.rad2deg(sim_res.alpha_qs[i])
        coefficients = np.asarray([self.C_d(alpha_qs_deg), self.C_l(alpha_qs_deg), -self.C_m(alpha_qs_deg)])
        return dynamic_pressure*chord*rot@coefficients
    
    @staticmethod
    def _init_quasi_steady(sim_res:ThreeDOFsAirfoil, **kwargs):
        pass

    def _BL(self, 
            sim_res: ThreeDOFsAirfoil,
            i: int,
            A1: float, A2: float, b1: float, b2: float,
            chord: float=1, density: float=1.225, pitching_around: float=0.25, alpha_at: float=0.75,
            a: float=343, K_alpha: float=0.75,
            T_p: float=1.5, T_bl: float=5,
            Cn_vortex_detach: float=1.0093, tau_vortex_pure_decay: float=11, T_v: float=5):
        sim_res.alpha_steady[i] = -sim_res.pos[i, 2]+sim_res.inflow_angle[i]

        # --------- MODULE unsteady attached flow
        flow_vel = np.sqrt(sim_res.inflow[i, :]@sim_res.inflow[i, :].T)
        ds = 2*sim_res.dt[i]*flow_vel/chord
        sim_res.s[i] = sim_res.s[i-1]+ds
        qs_flow_angle = self._quasi_steady_flow_angle(sim_res.vel[i, :], sim_res.pos[i, :], sim_res.inflow[i, :], 
                                                      chord, pitching_around, alpha_at)
        sim_res.alpha_qs[i] = -sim_res.pos[i, 2]+qs_flow_angle
        d_alpha_qs = sim_res.alpha_qs[i]-sim_res.alpha_qs[i-1] if i != 0 else 0
        sim_res.X_lag[i] = sim_res.X_lag[i-1]*np.exp(-b1*ds)+d_alpha_qs*A1*np.exp(-0.5*b1*ds)
        sim_res.Y_lag[i] = sim_res.Y_lag[i-1]*np.exp(-b2*ds)+d_alpha_qs*A2*np.exp(-0.5*b2*ds)
        sim_res.alpha_eff[i] = sim_res.alpha_qs[i]-sim_res.X_lag[i]-sim_res.Y_lag[i]
        sim_res.C_nc[i] = self._C_l_slope*np.sin(sim_res.alpha_eff[i]-self._alpha_0L)  #todo check that this sin() is
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
        sim_res.D_p[i] = sim_res.D_p[i-1]*np.exp(-ds/T_p)+(sim_res.C_npot[i]-sim_res.C_npot[i-1])*np.exp(-0.5*ds/T_p)
        sim_res.C_nsEq[i] = sim_res.C_npot[i]-sim_res.D_p[i]
        sim_res.alpha_sEq[i] = np.arcsin(sim_res.C_nsEq[i]/self._C_l_slope)+self._alpha_0L  # introduce np.arcsin
        
        # sim_res.f_t_Dp[i] = self._f_t(np.rad2deg(sim_res.alpha_sEq[i]))
        aoa = np.rad2deg(sim_res.alpha_sEq[i])
        coefficients = np.c_[self.C_d(aoa)-self._C_d_0, self.C_l(aoa)]

        rot = self.passive_2D(-sim_res.alpha_sEq[i])
        coeff_rot = rot@coefficients.T
        C_t, C_n = -coeff_rot[0], coeff_rot[1]
        
        f_t = C_t/(self._C_l_slope*np.sin(sim_res.alpha_sEq[i]-self._alpha_0L)**2)
        sim_res.f_t_Dp[i] = f_t**2*np.sign(f_t)

        f_n = 2*np.sqrt(C_n/(self._C_l_slope*np.sin(sim_res.alpha_sEq[i]-self._alpha_0L)))-1
        sim_res.f_n_Dp[i] = f_n**2*np.sign(f_n)

        # sim_res.f_t_Dp[i] = self._f_t(np.rad2deg(sim_res.alpha_sEq[i]))
        # sim_res.f_n_Dp[i] = self._f_n(np.rad2deg(sim_res.alpha_sEq[i]))

        sim_res.D_bl_t[i] = sim_res.D_bl_t[i-1]*np.exp(-ds/T_bl)+\
            (sim_res.f_t_Dp[i]-sim_res.f_t_Dp[i-1])*np.exp(-0.5*ds/T_bl)
        sim_res.D_bl_n[i] = sim_res.D_bl_n[i-1]*np.exp(-ds/T_bl)+\
            (sim_res.f_n_Dp[i]-sim_res.f_n_Dp[i-1])*np.exp(-0.5*ds/T_bl)
        
        sim_res.f_t[i] = sim_res.f_t_Dp[i]-sim_res.D_bl_t[i]
        sim_res.f_n[i] = sim_res.f_n_Dp[i]-sim_res.D_bl_n[i]

        C_nqs = sim_res.C_nc[i]*((1+np.sign(sim_res.f_n[i])*np.sqrt(np.abs(sim_res.f_n[i])))/2)**2
        sim_res.C_nf[i] = C_nqs+sim_res.C_ni[i]
        sim_res.C_tf[i] = self._C_l_slope*np.sin(sim_res.alpha_eff[i]-self._alpha_0L)**2*\
            np.sign(sim_res.f_t[i])*np.sqrt(np.abs(sim_res.f_t[i]))
        
        # --------- MODULE leading-edge vortex position
        if sim_res.C_nsEq[i] >= Cn_vortex_detach or d_alpha_qs < 0:
            sim_res.tau_vortex[i] = sim_res.tau_vortex[i-1]+0.45*ds
        else:
            sim_res.tau_vortex[i] = 0

        # --------- MODULE leading-edge vortex lift
        sim_res.C_nv_instant[i] = sim_res.C_nc[i]*(1-((1+np.sign(sim_res.f_n[i])*np.sqrt(np.abs(sim_res.f_n[i])))/2)**2)
        sim_res.C_nv[i] = sim_res.C_nv[i-1]*np.exp(-ds/T_v)
        if sim_res.tau_vortex[i] < tau_vortex_pure_decay:
            sim_res.C_nv[i] += (sim_res.C_nv_instant[i]-sim_res.C_nv_instant[i-1])*np.exp(-0.5*ds/T_v)
        

        # --------- MODULE moment coefficient
        rel_speed = np.sqrt((sim_res.inflow[i, 0]-sim_res.vel[i, 0])**2+
                            (sim_res.inflow[i, 1]-sim_res.vel[i, 1])**2)
        sim_res.C_mqs[i] = self.C_m(np.rad2deg(sim_res.alpha_eff[i]))
        sim_res.C_mnc[i] = -0.5*np.pi*chord*(alpha_at-pitching_around)/rel_speed*(-sim_res.vel[i, 2])
        C_m = sim_res.C_mqs[i]+sim_res.C_mnc[i]

        # --------- Combining everything
        # for return of [-C_d, C_l, C_m]
        # coefficients = np.asarray([sim_res.C_tf[i], sim_res.C_nf[i]+sim_res.C_nv[i], C_m])
        # rot = self.passive_3D_planar(sim_res.pos[i, 2]+sim_res.inflow_angle[i])
        # return rot@coefficients-np.asarray([self._C_d_0, 0, 0])

        # for return of [f_x, f_y, mom]
        # C_t and C_d point in opposite directions!
        coefficients = np.asarray([sim_res.C_tf[i], sim_res.C_nf[i]+sim_res.C_nv[i], -C_m*chord])
        rot = self.passive_3D_planar(-sim_res.pos[i, 2]) 
        dynamic_pressure = density/2*rel_speed
        forces = dynamic_pressure*chord*rot@coefficients  # for [-f_x, f_y, mom]
        forces[0] *= -1
        # return forces
        return rot@coefficients  # for [-C_d, C_l, C_m]
    
    def _init_BL(
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
        qs_flow_angle = self._quasi_steady_flow_angle(sim_res.vel[0, :], sim_res.pos[0, :], sim_res.inflow[0, :], 
                                                      chord, pitching_around, alpha_at)
        # sim_res.alpha_qs[-1] = -sim_res.pos[0, 2]+qs_flow_angle
        # sim_res.vel[-1, 2] = sim_res.vel[0, 2]

    def _BL_openFAST(self):
        pass
    
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
    
    def _zero_forces(self, alpha: np.ndarray, coeffs: dict[str, np.ndarray], save_as: str, alpha0_in: tuple=(-10, 5)):
        #todo calc d_Cn/d_alpha as dC_l_dalpha at every alpha
        ids_subset = np.logical_and(alpha>=alpha0_in[0], alpha<=alpha0_in[1])
        alpha_subset = alpha[ids_subset]
        zero_data = {}
        for coeff_name, coeff in coeffs.items():
            coeff_subset = coeff[ids_subset]
            find_sign_change = coeff_subset[1:]*coeff_subset[:-1]
            if np.all(find_sign_change>0):
                raise ValueError("The given C_n does not have a sign change in the current AoA interval of "
                                f"{alpha0_in} degree. Linear interpolation can thus not find alpha_0.")
            alpha_0, slope = self._get_zero_crossing(alpha_subset, coeff_subset)
            zero_data[f"alpha_0_{coeff_name.split("_")[-1]}"] = alpha_0
            zero_data[f"{coeff_name}_slope"] = slope
        with open(save_as, "w") as f:
            json.dump(zero_data, f, indent=4)
        return zero_data
    
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
    
    _implemented_schemes = ["linear"]

    _scheme_settings_must = {
        "linear": ["stiffness_edge", "stiffness_flap", "stiffness_tors", "damping_edge", "damping_flap", "damping_tors"]
    }

    _scheme_settings_can = {
        "linear": []
    }

    _sim_params_required = {
        "linear": []
    }

    def __init__(self, verbose: bool=True) -> None:
        SimulationSubRoutine.__init__(self, verbose=verbose)

        self.stiffness = None
        self.damp = None
    
    def get_scheme_method(self, scheme: str) -> Callable:
        scheme_methods = {
            "linear": self._linear
        }
        return scheme_methods[scheme]
    
    def get_scheme_init_method(self, scheme: str) -> Callable:
        scheme_methods = {
            "linear": self._init_linear,
        }
        return scheme_methods[scheme]

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
    
    def _init_linear(
            self, 
            sim_res: SimulationResults,
            stiffness_edge: float,
            stiffness_flap: float,
            stiffness_tors: float,
            damping_edge: float,
            damping_flap: float,
            damping_tors: float,
            **kwargs):
        self.stiffness = np.diag([stiffness_edge, stiffness_flap, stiffness_tors])
        self.damp = np.diag([damping_edge, damping_flap, damping_tors])


class TimeIntegration(SimulationSubRoutine, Rotations):
    """Class implementing differnt ways to calculate the position, velocity, and acceleration of the system at the
     next time step.
    """
    
    _implemented_schemes = ["eE", "HHT-alpha", "HHT-alpha-adapted"]    
    # eE: explicit Euler
    # HHT_alpha: algorithm as given in #todo add paper

    _scheme_settings_must = {
        "eE": [],
        "HHT-alpha": ["dt", "alpha", "stiffness_edge", "stiffness_flap", "stiffness_tors", "damping_edge", 
                      "damping_flap", "damping_tors", "mass", "mom_inertia"],
        "HHT-alpha-adapted": ["dt", "alpha", "stiffness_edge", "stiffness_flap", "stiffness_tors", "damping_edge",
                              "damping_flap", "damping_tors", "mass", "mom_inertia"],
    }

    _scheme_settings_can = {
        "eE": [],
        "HHT-alpha": [],
        "HHT-alpha-adapted": []
    }

    _sim_params_required = {
        "eE": [],
        "HHT-alpha": [],
        "HHT-alpha-adapted": []
    } 

    def __init__(self, verbose: bool=True) -> None:
        SimulationSubRoutine.__init__(self, verbose=verbose)

        self._M_current = None
        self._M_last = None
        self._C_last = None
        self._K_last = None
        self._external_current = None
        self._external_last = None
        self._beta = None
        self._gamma = None
        self._dt = None
        
    def get_scheme_method(self, scheme: str) -> Callable:
        scheme_methods = {
            "eE": self._eE,
            "HHT-alpha": self._HHT_alpha,
            "HHT-alpha-adapted": self._HHT_alpha_adapated,
        }
        return scheme_methods[scheme]
    
    def get_scheme_init_method(self, scheme: str) -> Callable:
        scheme_methods = {
            "eE": self._skip,
            "HHT-alpha": self._init_HHT_alpha,
            "HHT-alpha-adapted": self._init_HHT_alpha,
        }
        return scheme_methods[scheme]
    
    def _eE(
        self,
        sim_res: ThreeDOFsAirfoil, 
        i: int,
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
        next_vel = sim_res.vel[i, :]+sim_res.accel[i, :]*sim_res.dt[i]
        next_pos = sim_res.pos[i, :]+(sim_res.vel[i, :]+next_vel)/2*sim_res.dt[i]
        return next_pos, next_vel, accel
    
    def _HHT_alpha(
            self,
            sim_res: SimulationResults, 
            i: int,
            dt: float,
            **kwargs):  # the kwargs are needed because of the call in simulate():
        rot = self.passive_3D_planar(sim_res.pos[i, 2])
        rot_t = rot.T
        M_last = rot_t@self._M_last@rot 
        C_last = rot_t@self._C_last@rot 
        K_last = rot_t@self._K_last@rot 
        M_current = rot_t@self._M_current@rot 

        rhs = -M_last@sim_res.accel[i, :]-C_last@sim_res.vel[i, :]-K_last@sim_res.pos[i, :]
        rhs += self._external_current*sim_res.aero[i+1, :]+self._external_last*sim_res.aero[i, :]
        #todo the following lines are in the EFRot coordinate system!!! Rotate into XYRot
        accel = solve(self._M_current, rhs, assume_a="sym") 
        vel = sim_res.vel[i, :]+dt*((1-self._gamma)*sim_res.accel[i, :]+self._gamma*accel)
        pos = sim_res.pos[i, :]+dt*sim_res.vel[i, :]+dt**2*((0.5-self._beta)*sim_res.accel[i, :]+self._beta*accel)
        return pos, vel, accel

    def _HHT_alpha_adapated(
            self,
            sim_res: SimulationResults, 
            i: int,
            dt: float,
            **kwargs):  # the kwargs are needed because of the call in simulate():
        rot = self.passive_3D_planar(sim_res.pos[i, 2])
        rot_t = rot.T
        M_last = rot_t@self._M_last@rot 
        C_last = rot_t@self._C_last@rot 
        K_last = rot_t@self._K_last@rot 
        M_current = rot_t@self._M_current@rot 

        rhs = -M_last@sim_res.accel[i, :]-C_last@sim_res.vel[i, :]-K_last@sim_res.pos[i, :]
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
            mass: float,
            mom_inertia: float,
            stiffness_edge: float,
            stiffness_flap: float,
            stiffness_tors: float,
            damping_edge: float,
            damping_flap: float,
            damping_tors: float,
            **kwargs): #todo implies constant dt!!!
        self._dt = dt
        self._beta = (1+alpha)**2/4
        self._gamma = 0.5+alpha
        
        M = np.diag([mass, mass, mom_inertia])
        C = np.diag([damping_edge, damping_flap, damping_tors])
        K = np.diag([stiffness_edge, stiffness_flap, stiffness_tors])
        
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


