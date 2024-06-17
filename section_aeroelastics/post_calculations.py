import numpy as np
from calculations import Rotations
import pandas as pd
from os.path import join
from os import listdir
from scipy.signal import find_peaks
import json
from helper_functions import Helper
from defaults import DefaultsSimulation
helper = Helper()


class PowerAndEnergy(Rotations):
    def __init__(
            self,
            inertia: np.ndarray,
            stiffness: np.ndarray,
            position_xyz: np.ndarray,
            velocity_xyz: np.ndarray,
            position_ef: np.ndarray,
            velocity_ef: np.ndarray,
            f_aero_dl: np.ndarray,
            alpha_lift: np.ndarray,
            f_damp_ef: np.ndarray) -> None:
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

        self.position_xyz = position_xyz
        self.velocity_xyz = velocity_xyz
        self.position_ef = position_ef
        self.velocity_ef = velocity_ef
        
        self.f_aero_dl = f_aero_dl
        self.alpha_lift = alpha_lift
        self.f_damp_ef = f_damp_ef

    def power(self) -> dict[str: dict[str: np.ndarray]]:
        """Calculates the power done by the aerodynamic and structural damping forces.

        :return: Dictionary containing the power done by the aerodynamic forces (key: "aero") and the structural
        damping forces (key: "damp").
        :rtype: dict[str: dict[str: np.ndarray]]
        """
        sep_f_aero = self.project_separate(self.f_aero_dlm, -self.alpha_lift-self.position_xyz[:, 2])
        sep_f_aero_mean = (sep_f_aero[:-1]+sep_f_aero[1:])/2
        vel_xyz_mean = (self.velocity_xyz[:-1, :]+self.velocity_xyz[1:, :])/2
        power_drag = (sep_f_aero_mean[:, :2]*vel_xyz_mean[:, :2]).sum(axis=1)
        power_lift = (sep_f_aero_mean[:, 2:4]*vel_xyz_mean[:, :2]).sum(axis=1)
        power_moment = (sep_f_aero_mean[:, 4]*vel_xyz_mean[:, 2])

        mean_vel_efz = self.project(mean_vel_xyz, self.position_xyz[:-1, 2])
        # mean_vel_efz_test = self.project((self.position_xyz[1:, :]-self.position_xyz[:-1, :])/0.01, 
        #                                  self.position_xyz[:-1, 2])
        # print(mean_vel_efz-mean_vel_efz_test) 
        # print((mean_vel_efz-mean_vel_efz_test).sum(axis=0)) 

        f_damp_xyz = self.project(self.f_damp_efm, -self.position_xyz[:, 2])
        mean_f_damp_xyz = (f_damp_xyz[:-1, :]+f_damp_xyz[1:, :])/2
        mean_f_damp_efz = self.project(mean_f_damp_xyz, self.position_xyz[:-1, 2])
        power_damp = mean_f_damp_efz*mean_vel_efz
        # power_damp = mean_f_damp_efz*mean_vel_efz_test
        
        return {
            "aero": {
                "drag": power_drag,
                "lift": power_lift,
                "mom": power_moment
            },
            "damp": {
                "edge": power_damp[:, 0],
                "flap": power_damp[:, 1],
                "tors": self.velocity_xyz[:, 2]*self.f_damp_ef
            }
        }
    
    def kinetic_energy(self) -> dict[str: np.ndarray]:
        """Calculates the kinetic energy of the moving airfoil in the edgewise, flapwise, and torsional direction.

        :return: Dictionary containing the kinetic energies.
        :rtype: dict[str: np.ndarray]
        """
        # indices are (edge, flap, torsional)
        e_kin = 0.5*self.inertia*self.velocity_efz**2
        return {
            "edge": e_kin[:, 0],
            "flap": e_kin[:, 1],
            "tors": e_kin[:, 2]
        }

    def potential_energy(self) -> dict[str: np.ndarray]:
        """Calculates the potential energy of the moving airfoil in the edgewise, flapwise, and torsional direction.

        :return: Dictionary containing the potential energies.
        :rtype: dict[str: np.ndarray]
        """
        # indices are (edge, flap, torsional)
        e_pot = 0.5*self.stiffness*self.position_efz**2
        return {
            "edge": e_pot[:, 0],
            "flap": e_pot[:, 1],
            "tors":e_pot[:, 2]
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
            
            self.inertia = np.asarray([section_data["mass"], 
                                       section_data["mass"],
                                       section_data["mom_inertia"]])
            self.stiffness = np.asarray([section_data["stiffness_edge"], 
                                         section_data["stiffness_flap"], 
                                         section_data["stiffness_tors"]])
            self.damping = np.asarray([section_data["damping_edge"], 
                                       section_data["damping_flap"], 
                                       section_data["damping_tors"]])

        self.name_alpha_lift = alpha_lift
        self._cs_struc = coordinate_system_structure
        self._calc = None

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

        f_aero_dl = self.project_2D(f_aero, alpha_lift+position[:, 2])  # is now (drag, lift)
        f_aero_ef = self.project_2D(f_aero, position[:, 2])  # is now (edge, flap)

        # projection
        # The projection is different whether the structural CS (meaning in which coordinate system the structural
        # parameters are constant) is "xy" or "ef". For example, the velocity in the edgewise direction is generally
        # different from the velocity along the edgewise axis in the "ef" CS. The damping force is also different. 
        # The linear stiffness forces are the same but for coding consistency are separated nonetheless. External
        # forces and positions are unaffected by different structural CS.
        pos_ef = self.project_2D(position, position[:, 2])  # is now in (edge, flap)
        if self._cs_struc == "ef":
            # now vel_ef is the velocity along the edgewise and flapwise axis.
            vel_ef = self._v_ef(velocity, position, self._cs_struc)  # is now in (x_edge, x_flap)
            f_damp_ef = self.damping[:2]*vel_ef
            f_stiff_ef = self.stiffness[:2]*pos_ef
        if self._cs_struc == "xy":
            # now vel_ef is the velocity in the edgewise and flapwise direction.
            vel_ef = self._v_ef(velocity, position, self._cs_struc)  # is now in (edge, flap)
            f_damp_ef = self.project_2D(f_damp, position[:, 2])  # is now (edge, flap)
            f_stiff_ef = self.project_2D(f_stiff, position[:, 2])  # is now (edge, flap)
        
        # define new column names
        for i in range(2):
            self.df_general["pos_"+self._dfl_split["pos_projected"][i]] = pos_ef[:, i]
            self.df_general["vel_"+self._dfl_split["vel_projected"][i]] = vel_ef[:, i]

            self.df_f_aero["aero_"+self._dfl_split["aero_projected"][i]] = f_aero_dl[:, i]
            self.df_f_aero["aero_"+self._dfl_split["damp_projected"][i]] = f_aero_ef[:, i]

            self.df_f_structural["damp_"+self._dfl_split["damp_projected"][i]] = f_damp_ef[:, i]
            self.df_f_structural["stiff_"+self._dfl_split["stiff_projected"][i]] = f_stiff_ef[:, i]

        # save projected data
        for df_name, df in zip(["f_aero", "f_structural", "general"], 
                               [self.df_f_aero, self.df_f_structural, self.df_general]):
            df.to_csv(join(self.dir_in, self._dfl_filenames[df_name]), index=None)

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
            velocity_xyz = self.df_general[["vel_x", "vel_y", "vel_tors"]].to_numpy()
            previously_projected_data = [
                (self.df_general, ["pos_edge", "pos_flap", "pos_tors", "vel_edge", "vel_flap", "vel_tors"]),
                (self.df_f_aero, ["aero_drag", "aero_lift", "aero_mom"]),
                (self.df_f_structural, ["damp_edge", "damp_flap", "damp_tors"])
            ]
            # check wether project() was called already.
            for df, must_haves in previously_projected_data:
                has_columns = df.columns
                for must_have in must_haves:
                    if must_have not in has_columns:
                        raise ValueError(f"Results dataframe with columns {has_columns} does not have a column "
                                         f"{must_have} but it must. This likely results from forgetting to call "
                                         f"'project()' before {func.__name__}.")
            position_efz = self.df_general[["pos_edge", "pos_flap", "pos_tors"]].to_numpy()
            velocity_efz = self.df_general[["vel_edge", "vel_flap", "vel_tors"]].to_numpy()
            f_aero_dlm = self.df_f_aero[["aero_drag", "aero_lift", "aero_mom"]].to_numpy()
            alpha_lift = self.df_f_aero[self.name_alpha_lift].to_numpy()
            f_damp_efm = self.df_f_structural[["damp_edge", "damp_flap", "damp_tors"]].to_numpy()
            self._calc = PowerAndEnergy(self.inertia, self.stiffness, position_xyz, velocity_xyz, position_efz,
                                        velocity_efz, f_aero_dlm, alpha_lift, f_damp_efm)
            func(self)
        return wrapper
    
    @_init_calc
    def power(self):
        """Wrapper for PowerAndEnergy.power(). See details of the implementation there. "project()" needs to be called
        before power().
        """
        power_res = self._calc.power()
        df = pd.DataFrame()
        for category, forces in power_res.items():
            for force, values in forces.items():
                df[category+"_"+force] = values
        df.to_csv(join(self.dir_in, self._dfl_filenames["power"]), index=None)
    
    @_init_calc
    def kinetic_energy(self):
        """Wrapper for PowerAndEnergy.kinetic_energy(). See details of the implementation there. "project()" needs to be 
        called before kinetic_energy().
        """
        if self._calc is None:
            self._init_calc()
        e_kin_res = self._calc.kinetic_energy()
        df = pd.DataFrame()
        for category, values in e_kin_res.items():
            df[category] = values
        df.to_csv(join(self.dir_in, self._dfl_filenames["e_kin"]), index=None)
    
    @_init_calc
    def potential_energy(self):
        """Wrapper for PowerAndEnergy.potential_energy(). See details of the implementation there. "project()" needs to 
        be called before potential_energy().
        """
        if self._calc is None:
            self._init_calc()
        e_pot_res = self._calc.potential_energy()
        df = pd.DataFrame()
        for category, values in e_pot_res.items():
            df[category] = values
        df.to_csv(join(self.dir_in, self._dfl_filenames["e_pot"]), index=None)

    def work_per_cycle(self, peaks: np.ndarray=None):
        t = self.df_general["time"].to_numpy()
        dt = t[1:]-t[:-1]
        if peaks is None:
            pos_x = self.df_general["pos_x"].to_numpy()
            peaks = find_peaks(pos_x)[0]
        
        df_power = pd.read_csv(join(self.dir_in, self._dfl_filenames["power"]))
        period_power = {}
        for col in df_power.columns:
            timeseries = df_power[col].to_numpy()
            period_power[col] = [(timeseries[i_begin:i_end]*dt[i_begin:i_end]).sum() for 
                                 i_begin, i_end in zip(peaks[:-1], peaks[1:])]
        
        data =  {"cycle_no": np.arange(len(peaks)-1), "T": t[peaks[1:]]-t[peaks[:-1]]} | period_power
        df = pd.DataFrame(data)
        df.to_csv(join(self.dir_in, "period_work.dat"), index=None)

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
                    "rel_err_freq": np.r_[rel_err_freq/1e2, np.nan],
                    "err_ampl": err_ampl,
                    "rel_err_ampl": rel_err_ampl/1e2,
                }
                pd.DataFrame(res).to_csv(join(current_dir, "errors.dat"), index=None)

