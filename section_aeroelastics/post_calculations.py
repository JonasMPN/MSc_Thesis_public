import numpy as np
from calculations import Rotations
import pandas as pd
from os.path import join, isfile
from os import listdir
from scipy.signal import find_peaks
import json
from helper_functions import Helper
from defaults import DefaultsSimulation
from copy import copy
helper = Helper()


class PowerAndEnergy(Rotations):
    def __init__(
            self,
            inertia: np.ndarray,
            stiffness: np.ndarray,
            structural_rotation: np.ndarray,
            alpha_lift: np.ndarray,
            position: np.ndarray,
            velocity_xyrot: np.ndarray,
            velocity_damp: np.ndarray,
            velocity_e_kin: np.ndarray,
            f_aero_dlm: np.ndarray,
            f_damp: np.ndarray
            ) -> None:
        """Class to calculate the energy of a system and the power done by various forces.

        :param time: (n, ) array with the time
        :type inertia: np.ndarray
        :param inertia: (3,) array with (linear, linear, rotational) inertia
        :type inertia: np.ndarray
        :param stiffness: (3,) array with stiffness for motion in edgewise, flapwise, and around z
        :type stiffness: np.ndarray
        :param position_xyz: (n, 3) array with (x, y, rot z) position. Rot z in rad
        :type position: np.ndarray
        :param position_efz: (n, 3) array with (edge, flap, rot z) position. Rot z in rad
        :type position: np.ndarray
        :param velocity: (n, 3) array with (edgewise, flaptwise, rot z) velocities. Rot z velocity in rad/s
        :type velocity: np.ndarray
        :param f_aero: (n, 3) array with aerodynamic force in (drag, lift, moment around z)
        :type f_aero: np.ndarray
        :param alpha_lift: (n,) array with angle between lift and inflow. alpha_lift=0 and position[i, 2]=0 means
        that the inflow pointing along edgewise-axis. alpha_lift=pi/2 and position[i, 2]=0 means its pointing along
         flapwise.
        :type alpha_lift: np.ndarray
        :param f_damp: (n, 3) array with damping forces/moment (edgewise, flapwise, rot z)
        :type f_damp: np.ndarray
        """
        Rotations.__init__(self)
        self.inertia = inertia
        self.stiffness = stiffness

        self.position = position
        self.velocity_xyrot = velocity_xyrot
        self.velocity_damp = velocity_damp
        self.velocity_e_kin = velocity_e_kin

        self.alpha_lift = alpha_lift
        self.structural_rotation = structural_rotation
        
        self.f_aero_dlm = f_aero_dlm
        self.f_damp = f_damp

    def power(self) -> tuple[dict[str: dict[str: np.ndarray]], dict[str: np.ndarray]]:
        """Calculates the power done by the aerodynamic and structural damping forces. The powers are calculated by 
        means of dot products. Thus, the coordinate system of the velocities and the forces must be the same!

        :return: Dictionary containing the power done by the aerodynamic forces (key: "aero") and the structural
        damping forces (key: "damp").
        :rtype: dict[str: dict[str: np.ndarray]]
        """
        # the structural rotation is included in alpha_lift so it has to be substracted
        sep_f_aero = self.project_separate(self.f_aero_dlm, -self.alpha_lift-self.structural_rotation)
        # sep_f_aero is now (drag_x, drag_y, lift_x, lift_y, aero torque)
        sep_f_aero_mean = (sep_f_aero[:-1]+sep_f_aero[1:])/2
        vel_aero_mean = (self.velocity_xyrot[:-1, :]+self.velocity_xyrot[1:, :])/2
        # the following lines are dot products
        power_drag = (sep_f_aero_mean[:, :2]*vel_aero_mean[:, :2]).sum(axis=1)
        power_lift = (sep_f_aero_mean[:, 2:4]*vel_aero_mean[:, :2]).sum(axis=1)
        power_moment = (sep_f_aero_mean[:, 4]*vel_aero_mean[:, 2])

        # here it is just the edgewise/flapwise damping force times the respective velocity
        f_damp_mean = (self.f_damp[:-1, :]+self.f_damp[1:, :])/2
        vel_damp_mean = (self.velocity_damp[:-1, :]+self.velocity_damp[1:, :])/2
        power_damp = f_damp_mean*vel_damp_mean
        
        return {
            "aero": {
                "drag": power_drag,
                "lift": power_lift,
                "mom": power_moment
            },
            "damp": {
                0: power_damp[:, 0],
                1: power_damp[:, 1],
                2: power_damp[:, 2]
            }
        }
    
    def kinetic_energy(self) -> dict[str: np.ndarray]:
        """Calculates the kinetic energy of the moving airfoil in the edgewise, flapwise, and torsional direction. 
        This is different from the kinetic energy seen in the edgewise-flapwise coordinate system!

        :return: Dictionary containing the kinetic energies.
        :rtype: dict[str: np.ndarray]
        """
        e_kin = 0.5*self.inertia*self.velocity_e_kin**2
        return {
            0: e_kin[:, 0],
            1: e_kin[:, 1],
            2: e_kin[:, 2]
        }

    def potential_energy(self) -> dict[str: np.ndarray]:
        """Calculates the potential energy of the moving airfoil.

        :return: Dictionary containing the potential energies.
        :rtype: dict[str: np.ndarray]
        """
        e_pot = 0.5*self.stiffness*self.position**2
        return {
            0: e_pot[:, 0],
            1: e_pot[:, 1],
            2: e_pot[:, 2]
     }


class PostCaluculations(Rotations, DefaultsSimulation):
    """Class deriving parameter values from simulation results. These results have to follow the naming convention
    given by DefaultsSimulation. The new parameters are added to the simulation results files.
    """
    def __init__(self, dir_sim_res: str, alpha_lift: str, coordinate_system_structure: str) -> None:
        """Initialises class instance.

        :param dir_sim_res: Path to directory containing the results files from a simulation.
        :type dir_sim_res: str
        :param alpha_lift: Column name in the aerodynamic forces results file that specifies the angle of attack to 
        which the lift vector is normal to. 
        :type alpha_lift: str
        """
        Rotations.__init__(self)
        DefaultsSimulation.__init__(self)
        self.dir_in = dir_sim_res

        self.df_f_aero = pd.read_csv(join(dir_sim_res, self._dfl_filenames["f_aero"]))
        self.df_f_structural = pd.read_csv(join(dir_sim_res, self._dfl_filenames["f_structural"]))
        self.df_general = pd.read_csv(join(dir_sim_res, self._dfl_filenames["general"]))
        with open(join(dir_sim_res, self._dfl_filenames["section_data"]), "r") as f:
            section_data = json.load(f)
            
        self.inertia = np.asarray(section_data["inertia"])
        self.stiffness = np.asarray(section_data["stiffness"])
        self.damping = np.asarray(section_data["damping"])

        self.name_alpha_lift = alpha_lift
        self._cs_struc = coordinate_system_structure
        self._calc = None

        self._index_to_label = {0: "edge", 1: "flap", 2: "tors"}
        if self._cs_struc == "ef":
            self._index_to_label_pot = {0: "edge", 1: "flap", 2: "tors"}
        elif self._cs_struc == "xy":
            self._index_to_label_pot = {0: "x", 1: "y", 2: "tors"}
        else:
            raise ValueError(f"Unrecoginsed 'coordinate_system_structure'={self._cs_struc}")
        
    def check_angle_of_attack(self, **kwargs: float):
        exceeds_threshold = {}
        for aoa, threshold in kwargs.items():
            vals = self.df_f_aero[aoa].to_numpy()
            exceeds_threshold[aoa] = np.abs(vals) > np.deg2rad(threshold)
        pd.DataFrame(exceeds_threshold).to_csv(join(self.dir_in, self._dfl_filenames["aoa_threshold"]), index=None)

    def project_data(self) -> None:
        """The simulation data saved is purely in the xyz coordinate system. This function projects
         - the aerodynamic forces such that the forces are [drag, lift]. 
         - the structurel forces such that the forces are [edgewise, flapwise]
         - the position and velocity such that they are [edgewise, flapwise]
         as additional data.
        """
        # transform dataframes into numpy arrays
        position = np.asarray([self.df_general[col].to_numpy() for col in self._get_split("pos")]).T
        velocity = np.asarray([self.df_general[col].to_numpy() for col in self._get_split("vel")]).T

        f_aero = np.asarray([self.df_f_aero[col].to_numpy() for col in self._get_split("aero")]).T
        alpha_lift = self.df_f_aero[self.name_alpha_lift].to_numpy()
        f_damp = np.asarray([self.df_f_structural[col].to_numpy() for col in self._get_split("damp")]).T
        f_stiff = np.asarray([self.df_f_structural[col].to_numpy() for col in self._get_split("stiff")]).T

        f_aero_dl = self.project_2D(f_aero[:, :2], alpha_lift+position[:, 2])  # is now (drag, lift)
        f_aero_ef = self.project_2D(f_aero[:, :2], position[:, 2])  # is now (edge, flap)

        # projection
        # The projection is different whether the structural CS (meaning in which coordinate system the structural
        # parameters are constant) is "xy" or "ef". For example, the velocity in the edgewise direction is generally
        # different from the velocity along the edgewise axis in the "ef" CS. The damping force is also different. 
        # The linear stiffness forces are the same but for coding consistency are separated nonetheless. External
        # forces and positions are unaffected by different structural CS. The different projections are needed 
        # because the damping happens in the system the structural parameters are defined in.
        pos_ef = self.project_2D(position[:, :2], position[:, 2])  # is now in (edge, flap)
        vel_ef_xy = self.project_2D(velocity[:, :2], position[:, 2])  # vel (edge, flap) as seen in xy
        vel_ef_ef = self._v_ef(velocity, position, "ef")  # vel (edge, flap) as seen in ef
        # if self._cs_struc == "xy":
        if True:
            f_damp_ef = self.project_2D(f_damp[:, :2], position[:, 2])  # is now (edge, flap) in xy
            f_stiff_ef = self.project_2D(f_stiff[:, :2], position[:, 2])  # is now (edge, flap) in xy
        elif self._cs_struc == "ef":
            f_damp_ef = self.damping[:2]*vel_ef_ef  # is now (edge, flap) in ef
            f_stiff_ef = self.stiffness[:2]*pos_ef  # is now (edge, flap) in ef
        
        # define new column names
        for i in range(2):
            self.df_general["pos_"+self._dfl_split["pos_projected"][i]] = pos_ef[:, i]
            self.df_general["vel_"+self._dfl_split["vel_projected_xy"][i]] = vel_ef_xy[:, i]
            self.df_general["vel_"+self._dfl_split["vel_projected_ef"][i]] = vel_ef_ef[:, i]

            self.df_f_aero["aero_"+self._dfl_split["aero_projected_dl"][i]] = f_aero_dl[:, i]
            self.df_f_aero["aero_"+self._dfl_split["aero_projected_ef"][i]] = f_aero_ef[:, i]

            self.df_f_structural["damp_"+self._dfl_split["damp_projected"][i]] = f_damp_ef[:, i]
            self.df_f_structural["stiff_"+self._dfl_split["stiff_projected"][i]] = f_stiff_ef[:, i]

        # save projected data
        for df_name, df in zip(["f_aero", "f_structural", "general"], 
                               [self.df_f_aero, self.df_f_structural, self.df_general]):
            columns = df.columns.tolist()
            columns.sort(key=str.lower)
            df.to_csv(join(self.dir_in, self._dfl_filenames[df_name]), index=None, columns=columns)

    def write_peaks(self, cols: dict[str, list[str]]={"general": ["pos_x"]}):
        peaks = {}
        time = pd.read_csv(join(self.dir_in, self._dfl_filenames["general"]))["time"].to_numpy().flatten()
        ff_peaks = join(self.dir_in, self._dfl_filenames["peaks"])
        for file, cols in cols.items():
            df = pd.read_csv(join(self.dir_in, self._dfl_filenames[file]))
            for col in cols:
                if col in peaks:
                    raise NotImplementedError(f"Column '{col}' exists in multiple result files; the peaks "
                                              f"saved to {ff_peaks} would overwrite one another.")
                # p_peaks = find_peaks(df[col].to_numpy().flatten())[0]
                # n_peaks = find_peaks(-df[col].to_numpy().flatten())[0] 
                # peaks["p_"+col] = [(peak, t) for peak, t in zip([0]+p_peaks.tolist(), np.r_[0, time[p_peaks]].tolist())]
                # peaks["n_"+col] = [(peak, t) for peak, t in zip([0]+n_peaks.tolist(), np.r_[0, time[n_peaks]].tolist())]
                peaks["p_"+col] = np.r_[0, find_peaks(df[col].to_numpy().flatten())[0]].tolist()
                peaks["n_"+col] = np.r_[0, find_peaks(-df[col].to_numpy().flatten())[0]].tolist()
        with open(ff_peaks, "w") as f_peaks:
            json.dump(peaks, f_peaks, indent=4)

    def _init_calc(func) -> callable:
        """Decorator initialising a PowerAndEnergy instance if it is not already initialised.

        :param func: A function needing self._calc, type(self._calc) = PowerAndEnergy to be initialised
        :type func: callable
        """
        def wrapper(self):
            #todo the way self._calc is handled is silly. It's meant to prevent PostCalculations from inheriting
            #todo all PowerAndEnergy methods; instead self._calc is a PowerAndEnergy instance
            if self._calc is not None:  # if self._calc is already initialised
                func(self)
                return
            # prepare inputs for __init__ of PowerAndEnergy
            position_xyz = self.df_general[["pos_x", "pos_y", "pos_tors"]].to_numpy()
            previously_projected_data = [
                (self.df_general, ["pos_edge", "pos_flap", "vel_edge_ef", "vel_flap_ef", "vel_edge_xy", "vel_flap_xy"]),
                (self.df_f_aero, ["aero_drag", "aero_lift"]),
                (self.df_f_structural, ["damp_edge", "damp_flap"])
            ]
            # check whether project() was called already.
            for df, must_haves in previously_projected_data:
                has_columns = df.columns
                for must_have in must_haves:
                    if must_have not in has_columns:
                        raise ValueError(f"Results dataframe with columns {has_columns} does not have a column "
                                         f"{must_have} but it must. This likely results from forgetting to call "
                                         f"'project()' before {func.__name__}.")
            vel_xyrot = self.df_general[["vel_x", "vel_y", "vel_tors"]].to_numpy()
            # if self._cs_struc == "xy":
            #     pos = self.df_general[["pos_x", "pos_y", "pos_tors"]].to_numpy()
            #     vel_damp = vel_xyrot
            #     vel_e_kin = vel_xyrot
            #     f_damp = self.df_f_structural[["damp_x", "damp_y", "damp_tors"]].to_numpy()
            # elif self._cs_struc == "ef":
            #     pos = self.df_general[["pos_edge", "pos_flap", "pos_tors"]].to_numpy()
            #     vel_damp = self.df_general[["vel_edge_ef", "vel_flap_ef", "vel_tors"]].to_numpy()
            #     vel_e_kin = self.df_general[["vel_edge_xy", "vel_flap_xy", "vel_tors"]].to_numpy()
            #     f_damp = self.df_f_structural[["damp_edge", "damp_flap", "damp_tors"]].to_numpy()
            if self._cs_struc == "xy":
                pos = self.df_general[["pos_x", "pos_y", "pos_tors"]].to_numpy()
            elif self._cs_struc == "ef":
                pos = self.df_general[["pos_edge", "pos_flap", "pos_tors"]].to_numpy()
            vel_e_kin = self.df_general[["vel_edge_xy", "vel_flap_xy", "vel_tors"]].to_numpy()
            vel_damp = vel_e_kin
            f_damp = self.df_f_structural[["damp_edge", "damp_flap", "damp_tors"]].to_numpy()

            f_aero_dlm = self.df_f_aero[["aero_drag", "aero_lift", "aero_mom"]].to_numpy()
            alpha_lift = self.df_f_aero[self.name_alpha_lift].to_numpy()
            self._calc = PowerAndEnergy(self.inertia, self.stiffness, position_xyz[:, 2], alpha_lift,
                                        pos, vel_xyrot, vel_damp, vel_e_kin, f_aero_dlm, f_damp)
            func(self)
        return wrapper
    
    @_init_calc
    def power(self):
        """Wrapper for PowerAndEnergy.power(). See details of the implementation there. "project()" needs to be called
        before power().
        """
        power_res = self._calc.power()
        power = {}
        for category, forces in power_res.items():
            for force, values in forces.items():
                if category == "damp":
                    force = self._index_to_label[force]
                power[category+"_"+force] = values
        pd.DataFrame(power).to_csv(join(self.dir_in, self._dfl_filenames["power"]), index=None)
    
    @_init_calc
    def kinetic_energy(self):
        """Wrapper for PowerAndEnergy.kinetic_energy(). See details of the implementation there. "project()" needs to be 
        called before kinetic_energy().
        """
        e_kin_res = self._calc.kinetic_energy()
        e_kin_res = {self._index_to_label[axis]: values for axis, values in e_kin_res.items()}
        df = pd.DataFrame(e_kin_res)
        df["total"] = df.sum(axis=1)
        df.to_csv(join(self.dir_in, self._dfl_filenames["e_kin"]), index=None)
    
    @_init_calc
    def potential_energy(self):
        """Wrapper for PowerAndEnergy.potential_energy(). See details of the implementation there. "project()" needs to 
        be called before potential_energy().
        """
        e_pot_res = self._calc.potential_energy()
        e_pot_res = {self._index_to_label_pot[axis]: values for axis, values in e_pot_res.items()}
        df = pd.DataFrame(e_pot_res)
        df["total"] = df.sum(axis=1)
        df.to_csv(join(self.dir_in, self._dfl_filenames["e_pot"]), index=None)

    def work_per_cycle(self, peaks: np.ndarray|str="p_pos_x", constant_timestep: bool=True):
        t = self.df_general["time"].to_numpy()
        dt = t[1:]-t[:-1]
        if isinstance(peaks, str):
            ff_peaks = join(self.dir_in, self._dfl_filenames["peaks"])
            if not isfile(ff_peaks):
                raise RuntimeError("When using 'peaks' as a column indicator for the 'peaks.json' file, "
                                   "such a file needs to exist. Probably a call to PostCalculations().write_peaks() "
                                   "is missing.")
            with open(ff_peaks, "r") as f_peaks:
                peaks = json.load(f_peaks)[peaks]

        df_power = pd.read_csv(join(self.dir_in, self._dfl_filenames["power"]))
        period_power = {}
        for col in df_power.columns:
            timeseries = df_power[col].to_numpy()
            period_power[col] = [(timeseries[peak_begin:peak_end]*dt[peak_begin:peak_end]).sum() for 
                                 peak_begin, peak_end in zip(peaks[:-1], peaks[1:])]
        
        df_exceeded = None
        ff_aoa_thresholds = join(self.dir_in, self._dfl_filenames["aoa_threshold"])
        if isfile(ff_aoa_thresholds):
            df_thresholds = pd.read_csv(ff_aoa_thresholds)
            aoas = df_thresholds.columns
            exceeds_threshold_per_cycle = []
            periods = np.zeros(len(peaks)-1)
            if constant_timestep:
                for i_period, (i_begin, i_end) in enumerate(zip(peaks[:-1], peaks[1:])):
                    period = dt[0]*(i_end-i_begin)
                    periods[i_period] = period
                    exceeds_threshold_per_cycle.append(df_thresholds.iloc[i_begin:i_end].sum()/period)
            else:
                raise NotImplementedError
            columns = {aoa: aoa+"_exceeds_threshold" for aoa in aoas}
            df_exceeded = pd.DataFrame(exceeds_threshold_per_cycle)
            df_exceeded = df_exceeded.rename(columns=columns)
        data =  {"cycle_no": np.arange(len(peaks)-1), "T": periods} | period_power
        
        df = pd.DataFrame(data)
        if df_exceeded is not None:
            df = pd.concat((df, df_exceeded), axis=1)
        df.to_csv(join(self.dir_in, self._dfl_filenames["period_work"]), index=None)
        
    def _v_ef(self, velocity, position, coordinate_system: str):
        if coordinate_system == "ef":
            return self._v_ef_calc(velocity[:, 0], velocity[:, 1], velocity[:, 2], 
                                position[:, 0], position[:, 1], position[:, 2])
        elif coordinate_system == "xy":
            return self.project_2D(velocity, position[:, 2])
        else:
            raise ValueError(f"Unsupported coordinate system '{coordinate_system}'.")

    @staticmethod
    def _v_ef_calc(v_0, v_1, v_2, x_0, x_1, x_2):
        c = np.cos(x_2)
        s = np.sin(x_2)
        return np.c_[c*v_0+s*v_1+(-s*x_0+c*x_1)*v_2, -s*v_0+c*v_1+(-c*x_0-s*x_1)*v_2]
        

class PostHHT_alpha:
    @staticmethod
    def amplitude_and_period(root_dir: str, first_peak_at=0):
        for dir_scheme in listdir(root_dir):
            if dir_scheme == "plots":
                continue
            for directory in listdir(join(root_dir, dir_scheme)):
                current_dir = join(root_dir, dir_scheme, directory)
                df_sol = pd.read_csv(join(current_dir, "analytical_sol.dat"))
                df_sim = pd.read_csv(join(current_dir, "general.dat"))
                sol = df_sol["solution"].to_numpy()
                sim = df_sim["pos_x"].to_numpy()
                time_sol = df_sol["time"].to_numpy()
                time_sim = df_sim["time"].to_numpy()
                if np.any(time_sol-time_sim):
                    raise ValueError("The same 'time' time series has to be used for both the analytical solution and "
                                    "the simulation.")
                
                peaks_sim = np.r_[first_peak_at, find_peaks(sim)[0]]
                peaks_sol = np.r_[first_peak_at, find_peaks(sol)[0]]
                if peaks_sim[1]/peaks_sol[1] < 0.7:
                    peaks_sim = peaks_sim[1:]
                n_peaks = min(peaks_sim.size, peaks_sol.size)
                peaks_sim = peaks_sim[:n_peaks-1]
                peaks_sol = peaks_sol[:n_peaks-1]

                period_sim = time_sim[peaks_sim][1:]-time_sim[peaks_sim][:-1]
                freq_sim = 2*np.pi/period_sim
                ampl_sim = sim[peaks_sim]

                period_sol = time_sol[peaks_sol][1:]-time_sol[peaks_sol][:-1]
                freq_sol = 2*np.pi/period_sol
                ampl_sol = sol[peaks_sol]
                
                err_freq_per_period = freq_sol-freq_sim
                err_freq = np.cumsum(err_freq_per_period)
                rel_err_freq = err_freq/freq_sol

                err_ampl = ampl_sol-ampl_sim
                i_start = 0 if ampl_sim[0] != 0 else 1  # if the solution starts at zero doge a division by zero
                rel_err_ampl = err_ampl[i_start:]/ampl_sol[i_start:]
                if i_start == 1:
                    rel_err_ampl = np.r_[0, rel_err_ampl]

                res = {
                    "err_freq": np.r_[err_freq, np.nan],
                    "rel_err_freq": np.r_[rel_err_freq, np.nan],
                    "err_ampl": err_ampl,
                    "rel_err_ampl": rel_err_ampl,
                }
                pd.DataFrame(res).to_csv(join(current_dir, "errors.dat"), index=None)

