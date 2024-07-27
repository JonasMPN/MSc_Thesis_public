from calculations import ThreeDOFsAirfoil, AeroForce, TimeIntegration, StructForce, Rotations, Oscillations
import pandas as pd
import numpy as np
from os import listdir
from os.path import join, isfile
from post_calculations import PostCaluculations, PostHHT_alpha
from utils_plot import PlotHandler
import numpy as np
from plotting import HHTalphaPlotter, BLValidationPlotter
from defaults import DefaultPlot, DefaultStructure
from helper_functions import Helper
import json
import matplotlib.pyplot as plt
from plotting import Plotter
helper = Helper()


def sep_point_calc(dir_profile: str, BL_scheme: str, resolution: int=100):
    aero = AeroForce(dir_polar=dir_profile)
    save_to = join(dir_profile, "sep_point_tests", str(resolution))
    for scheme in range(1, 6):
        aero._pre_calculations(BL_scheme, resolution, scheme, adjust_attached=True, adjust_separated=True,
                               dir_save_to=save_to)

def plot_sep_points(dir_sep_points: str, alpha_limits: tuple=(-30, 30)):
    dir_plots = helper.create_dir(join(dir_sep_points, "plots"))[0]
    directions = []
    for file in listdir(dir_sep_points):
        if not "sep_points" in file:
            continue
        file_name = file.split(".")[0]
        directions.append(file_name[-1])

    figs = {}
    axs = {}
    for direction in directions:
        axs[direction] = {}
        figs[direction] = {}
        for extend in ["full", "limited"]:
            fig, ax = plt.subplots()
            figs[direction][extend] = fig
            axs[direction][extend] = ax
    
    colours = {1: "peru", 2: "forestgreen", 3: "mediumblue", 4: "coral", 5: "cadetblue"}
    for file in listdir(dir_sep_points):
        if not "sep_points" in file:
            continue
        df = pd.read_csv(join(dir_sep_points, file))
        file_name = file.split(".")[0]
        tmp = file_name.split("_")
        scheme_id = int(tmp[2])
        direction = tmp[-1]
        alpha = df["alpha"]
        axs[direction]["full"].plot(alpha, df[f"f_{direction}"], label=scheme_id, color=colours[scheme_id])
        alpha_data = alpha.loc[(alpha>=alpha_limits[0]) & (alpha<=alpha_limits[1])]
        axs[direction]["limited"].plot(alpha_data.values, df[f"f_{direction}"].loc[alpha_data.index], label=scheme_id,
                                       color=colours[scheme_id])

    for direction in set(directions):
        for extend in ["full", "limited"]:
            handler = PlotHandler(figs[direction][extend], axs[direction][extend])
            y_lims = [-1.2, 1.2] if extend == "full" else [-0.2, 1.2]
            handler.update(x_labels=r"$\alpha$ (°)", y_labels=rf"$f_{direction}$", legend=True, grid=True,
                           y_lims=y_lims)
            handler.save(join(dir_plots, f"{extend}_{direction}.pdf"))

#todo compare f distributions for different resolutions
#todo compare recreation of coefficients for different resolutions

def run_BeddoesLeishman(
        dir_profile: str, BL_scheme: str, validation_against: str, index_unsteady_data: int,
        k: float, inflow_speed: float, chord: float, amplitude: float, mean: float, 
        period_res: int=1000, file_polar: str="polars_new.dat",
        # A1: float=0.3, A2: float=0.7, b1: float=0.14, b2: float=0.53
        A1: float=0.165, A2: float=0.335, b1: float=0.0445, b2: float=0.3
        ):
    n_periods = 8
    overall_res = n_periods*period_res
    omega = 2*k*inflow_speed/chord
    T = 2*np.pi/omega
    dt = T/period_res
    t = dt*np.arange(overall_res)

    airfoil = ThreeDOFsAirfoil(t)

    alpha = np.deg2rad(mean+amplitude*np.sin(omega*t))
    alpha_speed = np.deg2rad(amplitude*omega*np.cos(omega*t))
    airfoil.pos = np.c_[np.zeros((overall_res, 2)), -alpha]
    airfoil.vel = np.c_[np.zeros((overall_res, 2)), -alpha_speed]
    airfoil.inflow = inflow_speed*np.c_[np.ones_like(t), np.zeros_like(t)]
    airfoil.inflow_angle = np.zeros_like(t)
    airfoil.dt = np.r_[t[1:]-t[:-1], t[1]]
    airfoil.density = 1
    
    pitch_around = 0.5
    alpha_at = 0.75
    airfoil.set_aero_calc(dir_polar=dir_airfoil, file_polar=file_polar, scheme=BL_scheme, A1=A1, A2=A2, b1=b1, 
                          b2=b2, pitching_around=pitch_around, alpha_at=alpha_at)
    airfoil._init_aero_force(airfoil, chord=chord, pitching_around=pitch_around, alpha_at=alpha_at, A1=A1, A2=A2)
    coeffs = np.zeros((t.size, 3))
    for i in range(overall_res):
        coeffs[i, :] = airfoil._aero_force(airfoil, i, A1=A1, A2=A2, b1=b1, b2=b2, chord=chord, 
                                           pitching_around=pitch_around, alpha_at=alpha_at)
    airfoil.vel = np.c_[np.zeros((overall_res, 2)), -alpha_speed]  # because it's changed in aero._init_BL()

    dir_res = helper.create_dir(join(dir_profile, "validation", BL_scheme, validation_against,
                                     str(index_unsteady_data)))[0]
    airfoil.save(dir_res)
    f_f_aero = join(dir_res, "f_aero.dat")
    df = pd.read_csv(f_f_aero)
    df["C_d"] = coeffs[:, 0]  
    df["C_l"] = coeffs[:, 1]
    df["C_m"] = coeffs[:, 2]
    df.to_csv(f_f_aero, index=None)
    

class HHTalphaValidator(DefaultPlot, DefaultStructure, Rotations):
    def __init__(self, dir_polar: str, dir_root_results: str) -> None:
        DefaultPlot.__init__(self)
        self.dir_polar = dir_polar
        self.dir_root_results = helper.create_dir(dir_root_results)[0]
        self._struc_params = None
        self._response_wanted = None

    def set_structure(
            self,
            chord: float,
            mass: float=None,
            mom_inertia: float=None,
            stiffness_edge: float=None,
            stiffness_flap: float=None,
            stiffness_tors: float=None,
            damping_edge: float=None,
            damping_flap: float=None,
            damping_tors: float=None,
    ):
        skip = ["self", "chord", "mass", "mom_inertia"]
        struc_params = {param: val for param, val in locals().items() if param not in ["skip"]+skip}
        response_wanted = {i: [] for i in range(3)}
        for i, (param, val) in enumerate(struc_params.items()):
            if val == None:
                struc_params[param] = 0
                response_wanted[i%3].append(False)
            else:
                response_wanted[i%3].append(True)
        self._response_wanted = {i: any(val_give) for i, val_give in response_wanted.items()}
        if (self._response_wanted[0] or self._response_wanted[1]) and mass is None:
            raise ValueError("'mass' must be set when calculating the response in the edgewise or flapwise direction.")
        if self._response_wanted[2] and mom_inertia is None:
            raise ValueError("'mom_inertia' must be set when calculating the response in the torsional direction.")
        
        self._struc_params = struc_params|{"chord": chord, 
                                           "mass": mass if mass is not None else 1, 
                                           "mom_inertia": mom_inertia if mom_inertia is not None else 1}
        self._inertia = np.asarray([mass, mass, mom_inertia])
        self._struct_force = StructForce()

    def simulate_case(
            self,
            case_name: str,
            alpha_HHT: float,
            time: np.ndarray,
            init_pos: np.ndarray,
            init_vel: np.ndarray,
            init_accel: np.ndarray,
            f_edge: np.ndarray=None,
            f_flap: np.ndarray=None,
            f_tors: np.ndarray=None,
            scheme: str="HHT-alpha-xy"
    ):
        f_ext = np.zeros((time.size, 3))
        f_response_wanted = {i: True for i in range(3)}
        for i, val in enumerate([f_edge, f_flap, f_tors]):
            if val is None:
                f_response_wanted[i] = False
            else:
                f_ext[:, i] = val
        for i, direction in enumerate(["edgewise", "flapwise", "torsional"]):
            if self._response_wanted[i] and not f_response_wanted[i]:
                raise ValueError(f"{direction} force time series is missing.")
            elif not self._response_wanted[i] and f_response_wanted[i]:
                raise ValueError(f"{direction} structural parameter(s) is(are) missing.")
            
        dir_save = helper.create_dir(join(self.dir_root_results, case_name))[0]
        airfoil = self._simulate(time=time, f_external=f_ext, init_pos=init_pos, init_vel=init_vel,
                                 init_accel=init_accel, alpha_HHT=alpha_HHT, scheme=scheme)
        airfoil._save(dir_save)
        with open(join(dir_save, "section_data.json"), "w") as f:
            json.dump(self._struc_params, f, indent=4)

        post_calc = PostCaluculations(dir_save, "alpha_steady")
        post_calc.project_data()
        post_calc.work()
        post_calc.kinetic_energy()
        post_calc.potential_energy()

    def simulate_undamped(
            self,
            n_steps_per_oscillation: list|np.ndarray,
            nat_frequency: list|np.ndarray,
            n_oscillations: float,
            x_0: float = 1,
            scheme: str="HHT-alpha-xy"
            ):
        counter = 0
        for n_dt in n_steps_per_oscillation:
            for omega_n in nat_frequency:
                t = np.linspace(0, n_oscillations*2*np.pi/omega_n, n_dt*n_oscillations+1)
                self.set_structure(1, 1, stiffness_edge=omega_n**2)
                results = self._simulate(time=t, f_external=np.zeros((t.size, 3)), init_pos=[x_0, 0, 0], 
                                         init_vel=0, scheme=scheme)

                dir_save = helper.create_dir(join(self.dir_root_results, scheme, f"{counter}"))[0]
                results._save(dir_save)
                with open(join(dir_save, "info.json"), "w") as f:
                    json.dump({"omega_n": omega_n, "n_dt": n_dt}, f)

                osci = Oscillations(1, omega_n, 0)
                df_analytical = pd.DataFrame({"time": t, "solution": osci.undamped(t, x_0)})
                df_analytical.to_csv(join(dir_save, "analytical_sol.dat"), index=None)
                counter += 1

    def simulate_damped(
            self,
            natural_frequencies: list|np.ndarray,
            damping_coefficients: list|np.ndarray,
            n_steps_per_oscillation: list|np.ndarray,
            n_oscillations: float,
            x_0: float=1,
            v_0: float=1,
            scheme: str="HHT-alpha-xy"
            ):
        M = 1
        counter = 0
        for n_dt in n_steps_per_oscillation:
            for omega_n in natural_frequencies:
                for damping_coeff in damping_coefficients:
                    K = omega_n**2*M
                    C = damping_coeff*2*np.sqrt(M*K)
                    omega_d = omega_n*np.sqrt(1-damping_coeff**2)
                    delta = damping_coeff*omega_n
                    accel_0 = 2*delta*v_0-(delta**2+omega_d**2)*x_0
                    t = np.linspace(0, n_oscillations*2*np.pi/omega_d, n_dt*n_oscillations+1)
                    self.set_structure(1, mass=M, stiffness_edge=K, damping_edge=C)
                    results = self._simulate(time=t, f_external=np.zeros((t.size, 3)), init_pos=[x_0, 0, 0], 
                                             init_vel=[v_0, 0, 0], init_accel=[accel_0, 0 ,0], scheme=scheme)

                    dir_save = helper.create_dir(join(self.dir_root_results, scheme, f"{counter}"))[0]
                    results._save(dir_save)
                    with open(join(dir_save, "info.json"), "w") as f:
                        json.dump({"omega_n": omega_n, "n_dt": n_dt, "damping_coeff": damping_coeff}, f)

                    osci = Oscillations(1, omega_n, damping_coeff)
                    df_analytical = pd.DataFrame({"time": t, "solution": osci.damped(t, x_0, v_0)})
                    df_analytical.to_csv(join(dir_save, "analytical_sol.dat"), index=None)
                    counter += 1
    
    def simulate_forced(
            self,
            damping_coefficients: list|np.ndarray,
            n_steps_per_oscillation: list|np.ndarray,
            force_frequencies:list|np.ndarray,
            n_oscillations: float,
            x_0: float=1,
            v_0: float=1,
            scheme: str="HHT-alpha-xy"
            ):
        M = 1
        counter = 0
        omega_n = 1
        for n_dt in n_steps_per_oscillation:
            for damping_coeff in damping_coefficients:
                K = omega_n**2*M
                C = damping_coeff*2*np.sqrt(M*K)
                omega_d = omega_n*np.sqrt(1-damping_coeff**2)
                t = np.linspace(0, n_oscillations*2*np.pi/omega_d, n_dt*n_oscillations+1)
                self.set_structure(1, mass=M, stiffness_edge=K, damping_edge=C)
                for force_freq in force_frequencies:
                    K = omega_n**2*M
                    C = damping_coeff*2*np.sqrt(M*K)
                    omega_d = omega_n*np.sqrt(1-damping_coeff**2)
                    delta = damping_coeff*omega_n
                    accel_0 = 2*delta*v_0-(delta**2+omega_d**2)*x_0
                    t = np.linspace(0, n_oscillations*2*np.pi/omega_d, n_dt*n_oscillations+1)
                    f = np.c_[np.sin(force_freq*t), np.zeros((t.size, 2))]
                    results = self._simulate(time=t, f_external=f, init_pos=[x_0, 0, 0], init_vel=[v_0, 0, 0],
                                              init_accel=[accel_0, 0 ,0], scheme=scheme)

                    dir_save = helper.create_dir(join(self.dir_root_results, scheme, f"{counter}"))[0]
                    results._save(dir_save)
                    with open(join(dir_save, "info.json"), "w") as f:
                        json.dump({"omega_n": omega_n, "n_dt": n_dt, "damping_coeff": damping_coeff, 
                                    "f_amplitude": 1, "f_freq": force_freq}, f)

                    osci = Oscillations(1, omega_n, damping_coeff)
                    force_freq = np.asarray([force_freq])
                    force_ampl = np.ones(1)
                    df_analytical = pd.DataFrame({"time": t, "solution": osci.forced(t, force_ampl, force_freq, x_0,
                                                                                     v_0)})
                    df_analytical.to_csv(join(dir_save, "analytical_sol.dat"), index=None)
                    counter += 1

    def simulate_forced_composite(
            self,
            damping_coefficient: float,
            n_steps_per_oscillation: list|np.ndarray,
            force_frequencies: list|np.ndarray,
            force_amplitude: list|np.ndarray,
            n_oscillations: list|np.ndarray,
            x_0: float = 1,
            v_0: float = 1,
            scheme: str="HHT-alpha-xy"
            ):
        if force_amplitude.size != force_frequencies.size:
            raise ValueError("The same number of amplitudes and frequencies for the external excitations have to be "
                             "given.")
        M = 1
        counter = 0
        omega_n = 1
        K = omega_n**2*M
        C = damping_coefficient*2*np.sqrt(M*K)
        omega_d = omega_n*np.sqrt(1-damping_coefficient**2)
        self.set_structure(1, mass=M, stiffness_edge=K, damping_edge=C)
        for n_dt in n_steps_per_oscillation:
            t = np.linspace(0, n_oscillations*2*np.pi/omega_d, n_dt*n_oscillations+1)[:, np.newaxis]
            f = np.c_[(force_amplitude*np.sin(force_frequencies*t)).sum(axis=1), np.zeros((t.size, 2))]
            t = t.squeeze()
            results = self._simulate(time=t, f_external=f, init_pos=[x_0, 0, 0], 
                                        init_vel=[v_0, 0, 0], init_accel=[-x_0*omega_n**2, 0 ,0], scheme=scheme)

            dir_save = helper.create_dir(join(self.dir_root_results, scheme, f"{counter}"))[0]
            results._save(dir_save)
            with open(join(dir_save, "info.json"), "w") as f:
                json.dump({"omega_n": omega_n, "n_dt": n_dt, "damping_coeff": damping_coefficient, 
                            "f_amplitude": force_amplitude.tolist(), "f_freq": force_frequencies.tolist()}, f)

            osci = Oscillations(1, omega_n, damping_coeff)
            df_analytical = pd.DataFrame({"time": t, "solution": osci.forced(t, force_amplitude, force_frequencies, 
                                                                             x_0, v_0)})
            df_analytical.to_csv(join(dir_save, "analytical_sol.dat"), index=None)
            counter += 1
        
    def simulate_step(
            self,
            natural_frequencies: list|np.ndarray,
            damping_coefficients: list|np.ndarray,
            n_steps_per_oscillation: list|np.ndarray,
            n_oscillations: float,
            scheme: str="HHT-alpha-xy"
            ):
        M = 1
        counter = 0
        for n_dt in n_steps_per_oscillation:
            for omega_n in natural_frequencies:
                for damping_coeff in damping_coefficients:
                    K = omega_n**2*M
                    C = damping_coeff*2*np.sqrt(M*K)
                    t = np.linspace(0, n_oscillations*2*np.pi/omega_n, n_dt*n_oscillations+1)
                    self.set_structure(1, mass=M, stiffness_edge=K, damping_edge=C)

                    f = np.c_[np.ones_like(t), np.zeros((t.size, 2))]
                    results = self._simulate(time=t, f_external=f, init_pos=0, init_vel=0, 
                                             init_accel=[1/M, 0 ,0], scheme=scheme)

                    dir_save = helper.create_dir(join(self.dir_root_results, scheme, f"{counter}"))[0]
                    results._save(dir_save)
                    with open(join(dir_save, "info.json"), "w") as f:
                        json.dump({"omega_n": omega_n, "n_dt": n_dt, "damping_coeff": damping_coeff, 
                                   "step_amplitude": 1}, f)

                        osci = Oscillations(1, omega_n, damping_coeff)
                        df_analytical = pd.DataFrame({"time": t, "solution": osci.step(t, 1)})
                        df_analytical.to_csv(join(dir_save, "analytical_sol.dat"), index=None)
                        counter += 1

    def simulate_rotation(
            self,
            case_info: str,
            inertia: np.ndarray,
            stiffness: np.ndarray,
            damping: np.ndarray,
            t: np.ndarray,
            force: np.ndarray,
            x_0: np.ndarray=np.zeros(3),
            v_0: np.ndarray=np.zeros(3),
            case_id: int=None,
            scheme: str="HHT-alpha-xy"
            ):
            self.set_structure(1, mass=inertia[0], mom_inertia=inertia[2], 
                               stiffness_edge=stiffness[0], stiffness_flap=stiffness[1], stiffness_tors=stiffness[2],
                               damping_edge=damping[0], damping_flap=damping[1], damping_tors=damping[2])

            results = self._simulate(time=t, f_external=force, init_pos=x_0, init_vel=v_0, scheme=scheme)
            dir_res = helper.create_dir(join(self.dir_root_results, scheme))[0]
            ids = [int(index) for index in listdir(dir_res)]
            case_id = max(ids)+1 if case_id is None else case_id
            dir_save = helper.create_dir(join(self.dir_root_results, scheme, str(case_id)))[0]
            with open(join(dir_save, case_info+".txt"), "w") as f:
                f.write(case_info)
            results.save(dir_save)

            osci = Oscillations(inertia, stiffness=stiffness, damping=damping)
            t_sim, pos, vel, accel = osci.rotation(t[-1], t[1], force, x_0, v_0)

            df_analytical = pd.DataFrame({"time": t_sim, 
                                          "pos_x": pos[:, 0], "pos_y": pos[:, 1], "pos_tors": pos[:, 2],
                                          "vel_x": vel[:, 0], "vel_y": vel[:, 1], "vel_tors": vel[:, 2]})
            df_analytical.to_csv(join(dir_save, "analytical_sol.dat"), index=None)

    def _simulate(
        self,
        time: np.ndarray,
        f_external: np.ndarray,
        init_pos: np.ndarray,
        init_vel: np.ndarray,
        init_accel: np.ndarray=None,
        scheme: str="HHT-alpha-xy",
        alpha_HHT: float=0.1,
        ) -> ThreeDOFsAirfoil:
        airfoil = ThreeDOFsAirfoil(time)
        airfoil.pos[0, :] = init_pos
        airfoil.vel[0, :] = init_vel
        
        airfoil.aero = f_external
        time_integrator = TimeIntegration()
        inertia = [self._struc_params["mass"], self._struc_params["mass"], self._struc_params["mom_inertia"]]
        damping = [self._struc_params["damping_edge"], self._struc_params["damping_flap"], 
                   self._struc_params["damping_tors"]]
        stiffness = [self._struc_params["stiffness_edge"], self._struc_params["stiffness_flap"], 
                     self._struc_params["stiffness_tors"]]
        time_integrator._init_HHT_alpha(airfoil, alpha_HHT, time[1], inertia=inertia, damping=damping, 
                                        stiffness=stiffness)
        integration_func = time_integrator.get_scheme_method(scheme)

        self._struct_force._init_linear(airfoil, damping=damping, stiffness=stiffness)
        if init_accel is not None:
            airfoil.accel[0, :] = init_accel
        elif "xy" in scheme:
            airfoil.accel[0, :] = (f_external[0]-self._struct_force.stiffness@init_pos)/self._inertia
        elif "ef" in scheme:
            rot = self.passive_3D_planar(init_pos[2])
            init_stiff = rot.T@self._struct_force.stiffness@rot
            airfoil.accel[0, :] = (f_external[0]-init_stiff@init_pos)/self._inertia
        
        airfoil.damp = np.zeros((time.size, 3))
        airfoil.stiff = np.zeros((time.size, 3))
        dt = time[1]
        for i in range(time.size-1):
            pos, vel, accel = integration_func(airfoil, i, dt)
            airfoil.pos[i+1, :] = pos
            airfoil.vel[i+1, :] = vel
            airfoil.accel[i+1, :] = accel

            #todo check coordinate sytsem under rotation (also check for force)
            airfoil.damp[i, :], airfoil.stiff[i, :] = self._struct_force._linear_xy(airfoil, i)
            # if not dt*(i+1)%10:
            #     print(dt*i)
            
        i = time.size-1
        airfoil.alpha_steady = -airfoil.pos[:, 2]
        airfoil.damp[i, :], airfoil.stiff[i, :] = self._struct_force._linear_xy(airfoil, i)
        airfoil.inflow = np.zeros((time.size, 2))
        return airfoil
  

if __name__ == "__main__":
    do = {
        "separation_point_calculation": False,
        "separation_point_plots": False,
        "BL": False,
        "plot_BL_results_meas": False,
        "plot_BL_results_polar": True,
        "calc_HHT_alpha_response": False,
        "plot_HHT_alpha_response": False,
        "HHT_alpha_undamped": False,
        "HHT_alpha_damped": False,
        "HHT_alpha_forced": False,
        "HHT_alpha_forced_composite": False,
        "HHT_alpha_step": False,
        "HHT_alpha_rotation": False,
        "ef_vs_xy": False
    }
    dir_airfoil = "data/FFA_WA3_221"
    dir_HHT_alpha_validation = "data/HHT_alpha_validation"
    # BL_scheme = "BL_openFAST_Cl_disc"
    # BL_scheme = "BL_chinese"
    BL_scheme = "BL_openFAST_Cl_disc_f_scaled"
    # BL_scheme = "BL_Staeblein"
    HHT_alpha_case = "test"
    sep_point_test_res = 202
    period_res = 400

    if do["separation_point_calculation"]:
        sep_point_calc(dir_airfoil, BL_scheme, resolution=sep_point_test_res)

    if do["separation_point_plots"]:
        plot_sep_points(join(dir_airfoil, "sep_point_tests", str(sep_point_test_res)), alpha_limits=(-30, 30))

    if do["BL"]:
        # validation_against = "measurement"
        # cases_defined_in = "cases.dat"
        validation_against = "polar"
        cases_defined_in = "cases_polar.dat"
        df_cases = pd.read_csv(join(dir_airfoil, "unsteady", cases_defined_in))
        for i, row in df_cases.iterrows():
            run_BeddoesLeishman(dir_airfoil, BL_scheme, validation_against, index_unsteady_data=i,
                                k=row["k"], inflow_speed=row["inflow_speed"], chord=row["chord"], 
                                amplitude=row["amplitude"], mean=row["mean"], period_res=period_res)

    if do["plot_BL_results_meas"]:
        plotter = BLValidationPlotter()
        # plotter.plot_preparation(join(dir_airfoil, "preparation", BL_scheme), join(dir_airfoil, "polars_new.dat"))
        dir_validations = join(dir_airfoil, "validation", BL_scheme, "measurement")
        dir_unsteady = join(dir_airfoil, "unsteady")
        df_cases = pd.read_csv(join(dir_unsteady, "cases.dat"))
        df_meas_files = pd.read_csv(join(dir_unsteady, "cases_results.dat"))
        for case_id, row in df_meas_files.iterrows():
            dir_case = join(dir_validations, str(case_id))
            # plotter.plot_model_comparison(dir_case)
            plotter.plot_meas_comparison(join(dir_unsteady, "unsteady_data"), row.to_dict(), dir_case, 
                                         period_res=period_res)
            
    if do["plot_BL_results_polar"]:
        file_polar = join(dir_airfoil, "polars_new.dat")
        plotter = BLValidationPlotter()
        # plotter.plot_preparation(join(dir_airfoil, "preparation", BL_scheme), join(dir_airfoil, "polars.dat"))
        dir_validations = join(dir_airfoil, "validation", BL_scheme, "polar")
        df_cases = pd.read_csv(join(dir_airfoil, "unsteady", "cases_polar.dat"))
        for case_name in listdir(dir_validations):
            dir_case = join(dir_validations, case_name)
            plotter.plot_over_polar(file_polar, dir_case, period_res, df_cases.iloc[int(case_name)])
  
    if do["calc_HHT_alpha_response"] or do["plot_HHT_alpha_response"]:
        T = 1
        dt = 0.1
        t = dt*np.arange(T/dt+1)
        alpha_HHT = 0.1

        init_pos = (1, 0, 0)
        init_vel = (0, 0, 0)
        init_accel = (0, 0, 0)

        force_edge = np.zeros_like(t)
        force_flap = None
        force_tors = None

        chord = 1
        mass = .1
        mom_inertia = None

        stiffness_edge = 1
        stiffness_flap = None
        stiffness_tors = None

        damping_edge = 1
        damping_flap = None
        damping_tors = None

    if do["calc_HHT_alpha_response"]:
        validator = HHTalphaValidator(dir_airfoil, dir_HHT_alpha_validation)
        validator.set_structure(chord=chord,mass=mass,mom_inertia=mom_inertia,stiffness_edge=stiffness_edge,
                                stiffness_flap=stiffness_flap,stiffness_tors=stiffness_tors,damping_edge=damping_edge,
                                damping_flap=damping_flap,damping_tors=damping_tors)   
        validator.simulate_case(case_name=HHT_alpha_case, time=t, alpha_HHT=alpha_HHT, init_pos=init_pos,
                                init_vel=init_vel, init_accel=init_accel, f_edge=force_edge, f_flap=force_flap, 
                                f_tors=force_tors)
        
    if do["plot_HHT_alpha_response"]:
        plotter = HHTalphaPlotter()
        plotter.plot_case(case_dir=join(dir_HHT_alpha_validation, HHT_alpha_case))

    if do["HHT_alpha_undamped"]:
        n_oscillations = 10
        n_t_per_oscillation = [80]  # should be a multiple of 4 to get the analytical peaks right
        omega_n = [1, 2, 3, 4]
        root_dir = join(dir_HHT_alpha_validation, "undamped")
        validator = HHTalphaValidator(dir_airfoil, root_dir)
        validator.simulate_undamped(n_t_per_oscillation, omega_n, n_oscillations, scheme="HHT-alpha-xy")
        PostHHT_alpha().amplitude_and_period(root_dir)
        HHTalphaPlotter().sol_and_sim(root_dir, ["ampl", "freq"])

    if do["HHT_alpha_damped"]:
        n_oscillations = 10
        n_t_per_oscillation = [32]  # should be a multiple of 4 to get the analytical peaks right
        omega_n = [1, 2, 3, 4]
        damping_coeffs = [0.1, 0.2, 0.3]
        root_dir = join(dir_HHT_alpha_validation, "damped")
        validator = HHTalphaValidator(dir_airfoil, root_dir)
        validator.simulate_damped(omega_n, damping_coeffs, n_t_per_oscillation, n_oscillations, 
                                  scheme="HHT-alpha-xy-adapted")
        PostHHT_alpha().amplitude_and_period(root_dir)
        HHTalphaPlotter().sol_and_sim(root_dir, ["ampl", "freq"])
    
    if do["HHT_alpha_forced"]:
        n_oscillations = 20
        n_t_per_oscillation = [12]  # should be a multiple of 4 to get the analytical peaks right
        force_freq = [0.6, 0.8, 1, 1.2, 1.4]
        damping_coeffs = [0.05, 0.1, 0.2]
        root_dir = join(dir_HHT_alpha_validation, "forced")
        # root_dir = join(dir_HHT_alpha_validation, "forced_force_term_adapted")
        validator = HHTalphaValidator(dir_airfoil, root_dir)
        validator.simulate_forced(damping_coeffs, n_t_per_oscillation, force_freq, n_oscillations, 
                                  scheme="HHT-alpha-xy")
        validator.simulate_forced(damping_coeffs, n_t_per_oscillation, force_freq, n_oscillations,
                                   scheme="HHT-alpha-xy-adapted")
        PostHHT_alpha().amplitude_and_period(root_dir)
        HHTalphaPlotter().sol_and_sim(root_dir, ["ampl", "freq"])
    
    if do["HHT_alpha_forced_composite"]:
        n_oscillations = 10
        n_t_per_oscillation = [12, 24, 64]  # should be a multiple of 4 to get the analytical peaks right
        force_freq = np.asarray([0.6, 0.8, 1, 1.2, 1.4])  # natural frequency is 1
        force_ampl = np.asarray([1, 1, 1, 1, 1])
        damping_coeff = 0.1
        x_0 = 1
        v_0 = -4
        root_dir = join(dir_HHT_alpha_validation, "forced_composite")
        # root_dir = join(dir_HHT_alpha_validation, "forced_force_term_adapted")
        validator = HHTalphaValidator(dir_airfoil, root_dir)
        validator.simulate_forced_composite(damping_coeff, n_t_per_oscillation, force_freq, force_ampl,
                                            n_oscillations, x_0, v_0, scheme="HHT-alpha-xy")
        validator.simulate_forced_composite(damping_coeff, n_t_per_oscillation, force_freq, force_ampl,
                                            n_oscillations, x_0, v_0, scheme="HHT-alpha-xy-adapted")
        PostHHT_alpha().amplitude_and_period(root_dir)
        HHTalphaPlotter().sol_and_sim(root_dir, ["ampl", "freq"])

    if do["HHT_alpha_step"]:
        n_oscillations = 10
        n_t_per_oscillation = [12]  # should be a multiple of 4 to get the analytical peaks right
        omega_n = [1, 2, 3, 4]
        damping_coeffs = [0.1, 0.2, 0.3]
        root_dir = join(dir_HHT_alpha_validation, "step")
        validator = HHTalphaValidator(dir_airfoil, root_dir)
        validator.simulate_step(omega_n, damping_coeffs, n_t_per_oscillation, n_oscillations, scheme="HHT-alpha-xy")
        PostHHT_alpha().amplitude_and_period(root_dir)
        HHTalphaPlotter().sol_and_sim(root_dir, ["ampl", "freq"])

    if do["HHT_alpha_rotation"]:
        root_dir = join(dir_HHT_alpha_validation, "rotation")
        validator = HHTalphaValidator(dir_airfoil, root_dir)
        coordinate_system = "xy"
        scheme = f"HHT-alpha-{coordinate_system}-adapted"
        alpha_lift = "alpha_steady"
        # case_info = "[0,0,0]_to_[1,0,0]_to_[1,0,90]_to_[0,1,0]_to_[0,0,0]"
        case_info = "[0,0,0]_to_[1,1,0]"
        # case_info = "[0,0,0]_to_[1,0,0]_to_[2,0,0]_to_[-1,0,0]_to[0,0,0]"
        case_id = 1

        # x_0 = [1, 0.5, 2.*np.pi]
        x_0 = [0, 0, 0.5*np.pi]
        # x_0 = np.zeros(3)
        inertia = np.ones(3)
        # stiffness = np.asarray([2, 0.5, 1])
        stiffness = np.asarray([2, 5., 1])
        # damping = 1*np.ones_like(inertia)
        damping = np.asarray([3, 1, 1.])
        # damping = np.zeros(3)
        dt = 0.01
        # T = 100
        T = 20
        t = np.linspace(0, T, int(T/dt)+1)

        f = np.repeat(np.c_[2, 5, 0.*np.pi], repeats=int(20/dt)+1, axis=0)
        # f = np.r_[f, np.repeat(np.c_[5, 0, np.pi/2], repeats=int(20/dt), axis=0)]
        # t_change = np.linspace(0, np.pi/2, int(20/dt))
        # f_change = np.c_[5*np.cos(t_change), 5*np.sin(t_change), np.pi/2*np.cos(t_change)]
        # f = np.r_[f, f_change]
        # f = np.r_[f, np.repeat(np.c_[0, 5, 0], repeats=int(20/dt), axis=0)]
        # f = np.r_[f, np.zeros((int(20/dt)+1, 3))]  # the +1 is needed if scheme="HHT-alpha-xy"

        # f = np.repeat(np.c_[0.5, 0, 0.5*np.pi], repeats=int(20/dt), axis=0)
        # # print(f.shape)
        # # print(int(20/dt)-1)
        # f = np.r_[f, np.repeat(np.c_[2, 0, 0], repeats=int(20/dt), axis=0)]
        # f = np.r_[f, np.repeat(np.c_[-1, 0, 0], repeats=int(20/dt), axis=0)]
        # f = np.r_[f, np.repeat(np.c_[0, 0, 0], repeats=int(20/dt), axis=0)]

        # f = np.c_[0.*np.ones_like(t), 0*np.ones_like(t), 0.*np.pi*np.ones_like(t)]
        # f_ramp = np.r_[np.linspace(0, 1, 500), np.ones(1501)]
        # f_ramp = np.r_[np.zeros(500), np.ones(1501)]
        # f = np.c_[f_ramp, 0*np.ones_like(t), 0*np.pi*np.ones_like(t)]
        
        validator.simulate_rotation(case_info, inertia, stiffness, damping, t, f, x_0, case_id=case_id, scheme=scheme)
        # HHTalphaPlotter().rotation(root_dir, case_id)

        # do postcalc?  
        post_calc = True
        if post_calc:
            dir_case = join(root_dir, scheme, str(case_id))
            with open(join(dir_case, "section_data.json"), "w") as f:
                section_data = {
                    "chord": 1,
                    "inertia": [inertia[0], inertia[1], inertia[2]],
                    "damping": [damping[0], damping[1], damping[2]],
                    "stiffness": [stiffness[0], stiffness[1], stiffness[2]],
                }
                json.dump(section_data, f, indent=4)
            
            post_calc = PostCaluculations(dir_case, alpha_lift, coordinate_system)
            post_calc.project_data()
            post_calc.power() 
            post_calc.kinetic_energy() 
            post_calc.potential_energy()  
            # ids_cycles = [0, int(20/dt)-1, int(40/dt)-1, int(60/dt)-1, int(80/dt)-1, int(100/dt)-1]
            # # ids_cycles = [0, int(20/dt)-1, int(40/dt)-1, int(60/dt)-1, int(80/dt)-1]
            ids_cycles = [0, t.size-1]
            post_calc.work_per_cycle(ids_cycles)
            plotter = Plotter("data/FFA_WA3_221/profile.dat", dir_case, join(dir_case, "plots"), coordinate_system) 
            plotter.force()  # plot various forces of the simulation
            plotter.energy()  # plot various energies and work done by forces of the simulation
    
    if do["ef_vs_xy"]:
        root_dir = helper.create_dir(join("data", "ef_vs_xy"))[0]
        case_id = "[1,0,small]_stable_ef"
        case_dir = helper.create_dir(join(root_dir, str(case_id)))[0]

        get_rot = Rotations()

        # x_0 = np.asarray([3, 4, 1*np.pi/2])  # in xy
        x_0 = np.asarray([1, 0, 0.01*np.pi/2])  # in xy
        # x_0 = np.asarray([0, 0, 0])  # in xy
        v_0 = np.zeros(3)  # in xy
        # v_0 = np.asarray([0, 0, 1])  # in xy
        k = np.asarray([1, 4, 1])  # in ef
        # c = np.asarray([2, 1, 1])  # in ef
        c = np.asarray([0, 0, 0])  # in ef
        inertia = np.asarray([1, 1, 1])

        section_data = {
            "inertia": inertia.tolist(),
            "damping": c.tolist(),
            "stiffness": k.tolist(),
            "chord": 1
        }

        K = np.diag(k)
        C = np.diag(c)
        M = np.diag(inertia)

        T = 20
        dt = 0.0005
        # dt = 0.01
        t = np.linspace(0, T, int(T/dt)+1)
        n_per_10 = (t<=10).sum()

        f = np.c_[0.*np.ones_like(t), 0*np.ones_like(t), 0.*np.pi*np.ones_like(t)]
        # f = np.c_[np.ones(n_per_10), np.zeros(n_per_10), np.zeros(n_per_10)]
        # phi = np.linspace(0, 40, t.size)
        # f = np.r_[f, np.c_[np.cos(phi), np.sin(phi), np.zeros_like(phi)]]
        # f = np.r_[np.c_[np.cos(phi), np.sin(phi), np.zeros_like(phi)]]
        # f = get_inflow(t, [(10, 10, 1, 0), (20, 20, 4, 45), (30, 30, 0, 0)], 1, 0)

        # f = get_inflow(t, [(10, 10, 1, 0), (20, 20, 4, 45), (30, 30, 0, 0)], 1, 0)
        # moment = np.r_[np.zeros(n_per_10-1), 3*np.pi/2*np.ones(n_per_10-1), np.zeros(n_per_10-1), np.zeros(n_per_10)]
        # f = np.c_[f, moment]

        sf = StructForce()
        cs = "xy"
        init = sf.get_scheme_init_method(f"linear_{cs}")
        init(None, k, c)
        f_struct = sf.get_scheme_method(f"linear_{cs}")

        ti = TimeIntegration()
        init = ti.get_scheme_init_method("eE")
        init(None, inertia)
        time_integrator = ti.get_scheme_method("eE")

        time_integrator = TimeIntegration()
        time_integrator._init_eE("_", inertia)
        integration_func = time_integrator.get_scheme_method("eE")

        sf = StructForce()
        sf._init_linear("_", k, c)
        sf_funcs = {"xy": sf._linear_xy, "ef": sf._linear_ef}
        
        for cs in ["xy", "ef"]:
            dir_res = helper.create_dir(join(case_dir, cs))[0]

            with open(join(dir_res, "section_data.json"), "w") as json_file:
                json.dump(section_data, json_file, indent=4)

            airfoil = ThreeDOFsAirfoil(t)
            airfoil.dt = dt*np.ones(t.size)
            airfoil.pos[0, :] = x_0
            airfoil.vel[0, :] = v_0

            airfoil.aero = f
            airfoil.damp = np.zeros((t.size, 3))
            airfoil.stiff = np.zeros((t.size, 3))

            sf_func = sf_funcs[cs]   
            full_rotations = 0
            was_below_0 = False
            for i in range(t.size-1):
                current_angle = airfoil.pos[i, 2]
                airfoil.damp[i, :], airfoil.stiff[i, :] = sf_func(airfoil, i)

                # airfoil.aero[i, :] = [-np.sin(current_angle), np.cos(current_angle), 0]
                pos, vel, accel = integration_func(airfoil, i)
                # if pos[1] > 0 and was_below_0:
                #     full_rotations += 1
                #     was_below_0 = False
                # if pos[1] < 0:
                #     was_below_0 = True
                # angle = np.arctan2(pos[1], pos[0])
                # angle = angle if angle > 0 else angle+2*np.pi
                # pos[2] = angle+full_rotations*2*np.pi
                # vel[2] = (airfoil.pos[i, 2]-airfoil.pos[i-1, 2])/dt if i>0 else 1
                airfoil.pos[i+1, :] = pos
                airfoil.vel[i+1, :] = vel
                airfoil.accel[i+1, :] = accel

            i = t.size-1
            airfoil.damp[i, :], airfoil.stiff[i, :] = sf_func(airfoil, i)
            airfoil.inflow = np.zeros((t.size, 2))
            airfoil._save(dir_res)

            post_calc = PostCaluculations(dir_res, "alpha_steady", cs)
            post_calc.project_data()
            post_calc.power()
            post_calc.kinetic_energy()
            post_calc.potential_energy()

            dir_plots = helper.create_dir(join(dir_res, "plots"))[0]
            plotter = Plotter("data/FFA_WA3_221/profile.dat", dir_res, dir_plots, cs, dt_res=500)
            plotter.force()
            plotter.energy()

        # # E_pot = E_pot[:, :2]
        # # E_kin = E_kin[:, :2]
        # E_tot = E_pot.sum(axis=1)+E_kin.sum(axis=1)
        
        # v_in_ef = _v_ef_calc(vel[:, 0], vel[:, 1], vel[:, 2], pos[:, 0], pos[:, 1], pos[:, 2])
        # f_damp_ef = -c*v_in_ef
        # p_damp_ef = f_damp_ef*v_in_ef

        # f_stiff = -k*ef_poss
        # p_stiff = f_stiff*v_in_ef
        
        # # p_damp_xy = (f_damps[1:]+f_damps[:-1])/2*(vel[1:]+vel[:-1])/2
        # p_damp_xy = f_damps*vel

        # w_damp_ef = (p_damp_ef*dt)
        # w_damp_xy = (p_damp_xy*dt)

        # # w_damp_ef = (p_damp_ef*dt).sum(axis=0).sum()
        # # w_damp_xy = (p_damp_xy*dt).sum(axis=0).sum()
        # # damping_errors[j] = np.abs((E_tot[0]+w_damp))/E_tot[0]
        # # energy_errors[j] = np.abs((E_tot[0]-E_tot[-2]))/E_tot[0]

        # w_stiff_ef = (p_stiff*dt)
        # dpos = pos[1:, :]-pos[:-1, :]
        # work_x = f_ext[:-1, 0]*dpos[:, 0]
        # work_y = f_ext[:-1, 1]*dpos[:, 1]
        # work_tors = f_ext[:-1, 2]*dpos[:, 2]
        # work = np.c_[work_x, work_y, work_tors]

        # work_e = (np.cos(pos[:-1, 2])*f_ext[:-1, 0]+np.sin(pos[:-1, 2])*f_ext[:-1, 1])
        # work_e *= (np.cos(pos[:-1, 2])*dpos[:, 0]+np.sin(pos[:-1, 2])*dpos[:, 1])

        # work_f = (-np.sin(pos[:-1, 2])*f_ext[:-1, 0]+np.cos(pos[:-1, 2])*f_ext[:-1, 1])
        # work_f *= (-np.sin(pos[:-1, 2])*dpos[:, 0]+np.cos(pos[:-1, 2])*dpos[:, 1])

        # work_ef = np.c_[work_e, work_f]

        # print(E_tot[0], E_tot[-1], w_damp_xy.sum(axis=0).sum(), w_damp_ef.sum(axis=0).sum(), work.sum(axis=0).sum())
        # print(E_pot[0, :])
        # print(w_damp_xy.sum(axis=0))
        # print(w_damp_ef.sum(axis=0))
        # print(work.sum(axis=0))
        # print(work_ef.sum(axis=0))