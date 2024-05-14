from calculations import AeroForce, Oscillations
import pandas as pd
import numpy as np
from os import listdir
from os.path import join, isfile
from calculations import ThreeDOFsAirfoil, AeroForce, TimeIntegration, StructForce
from post_calculations import PostCaluculations, PostHHT_alpha
from utils_plot import MosaicHandler
import numpy as np
from plotting import HHTalphaPlotter, BLValidationPlotter
from defaults import DefaultPlot, DefaultStructure
from helper_functions import Helper
import json
import matplotlib.pyplot as plt
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
            handler = MosaicHandler(figs[direction][extend], axs[direction][extend])
            y_lims = [-1.2, 1.2] if extend == "full" else [-0.2, 1.2]
            handler.update(x_labels=r"$\alpha$ (°)", y_labels=rf"$f_{direction}$", legend=True, grid=True,
                           y_lims=y_lims)
            handler.save(join(dir_plots, f"{extend}_{direction}.pdf"))

#todo compare f distributions for different resolutions
#todo compare recreation of coefficients for different resolutions

def run_BeddoesLeishman(
        dir_profile: str, BL_scheme: str, validation_against: str, index_unsteady_data: int,
        k: float, inflow_speed: float, chord: float, amplitude: float, mean: float, 
        period_res: int=1000,
        # A1: float=0.3, A2: float=0.7, b1: float=0.14, b2: float=0.53
        A1: float=0.165, A2: float=0.335, b1: float=0.0445, b2: float=0.3
        ):
    n_periods = 8
    overall_res = n_periods*period_res
    omega = 2*k*inflow_speed/chord
    T = 2*np.pi/omega
    dt = T/period_res
    t = dt*np.arange(overall_res)

    airfoil = ThreeDOFsAirfoil(dir_profile, t)

    alpha = np.deg2rad(mean+amplitude*np.sin(omega*t))
    alpha_speed = np.deg2rad(amplitude*omega*np.cos(omega*t))
    airfoil.pos = np.c_[np.zeros((overall_res, 2)), -alpha]
    airfoil.vel = np.c_[np.zeros((overall_res, 2)), -alpha_speed]
    airfoil.inflow = inflow_speed*np.c_[np.ones_like(t), np.zeros_like(t)]
    airfoil.inflow_angle = np.zeros_like(t)
    airfoil.dt = np.r_[t[1:]-t[:-1], t[1]]
    airfoil.density = 1

    # airfoil.set_aero_calc(BL_scheme, A1=A1, A2=A2, b1=b1, b2=b2, pitching_around=0.25, alpha_at=0.75)
    airfoil.set_aero_calc(BL_scheme, A1=A1, A2=A2, b1=b1, b2=b2, pitching_around=0.5, alpha_at=0.75)
    return
    airfoil._init_aero_force(airfoil, pitching_around=0.5, A1=A1, A2=A2)
    coeffs = np.zeros((t.size, 3))
    for i in range(overall_res):
        coeffs[i, :] = airfoil._aero_force(airfoil, i, A1=A1, A2=A2, b1=b1, b2=b2)
    airfoil.vel = np.c_[np.zeros((overall_res, 2)), -alpha_speed]  # because it's changed in aero._init_BL()

    dir_res = helper.create_dir(join(dir_profile, "validation", BL_scheme, validation_against,
                                     str(index_unsteady_data)))[0]
    airfoil.save(dir_res)
    f_f_aero = join(dir_res, "f_aero.dat")
    df = pd.read_csv(f_f_aero)
    df["C_d"] = coeffs[:, 0]  # "BL_openFAST"
    df["C_l"] = coeffs[:, 1]
    df["C_m"] = coeffs[:, 2]
    df.to_csv(f_f_aero, index=None)
    

class HHTalphaValidator(DefaultPlot, DefaultStructure):
    def __init__(self, dir_polar: str, dir_root_results: str) -> None:
        DefaultPlot.__init__(self)
        self.dir_polar = dir_polar
        self.dir_root_results = dir_root_results
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
            scheme: str="HHT-alpha"
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
            scheme: str="HHT-alpha"
            ):
        counter = 0
        for n_dt in n_steps_per_oscillation:
            for omega_n in nat_frequency:
                t = np.linspace(0, n_oscillations*2*np.pi/omega_n, n_dt*n_oscillations+1)
                self.set_structure(1, 1, stiffness_edge=omega_n**2)
                results = self._simulate(time=t, f_external=np.zeros((t.size, 3)), init_pos=[x_0, 0, 0], 
                                         init_vel=0, init_accel=[-x_0*omega_n**2, 0 ,0], scheme=scheme)

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
            scheme: str="HHT-alpha"
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
            scheme: str="HHT-alpha"
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
            scheme: str="HHT-alpha"
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
            scheme: str="HHT-alpha"
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
            natural_frequencies: list[np.ndarray],
            damping_coefficients: list[np.ndarray],
            n_steps_per_oscillation: list|np.ndarray,
            n_oscillations: float,
            inertia: np.ndarray=np.ones(3),
            scheme: str="HHT-alpha"
            ):
        counter = 0
        x_0 = [1, 0, -np.pi/2]
        v_0 = [0, 0, 0]
        a_0 = [0, 0, 0]
        for n_dt in n_steps_per_oscillation:
            for omega_n in natural_frequencies:
                for damping_coeff in damping_coefficients:
                    K = omega_n**2*inertia
                    C = damping_coeff*2*np.sqrt(inertia*K)
                    omega_d = omega_n*np.sqrt(1-damping_coeff**2)
                    t = np.linspace(0, n_oscillations*2*np.pi/omega_d.min(), n_dt*n_oscillations+1)
                    self.set_structure(1, mass=inertia[0], mom_inertia=inertia[2], 
                                       stiffness_edge=K[0], stiffness_flap=K[1], stiffness_tors=K[2],
                                       damping_edge=C[0], damping_flap=C[1], damping_tors=C[2])

                    f = np.c_[np.ones_like(t), np.zeros((t.size, 2))]
                    results = self._simulate(time=t, f_external=f, init_pos=x_0, init_vel=v_0, init_accel=a_0,
                                             scheme=scheme)

                    dir_save = helper.create_dir(join(self.dir_root_results, scheme, f"{counter}"))[0]
                    results._save(dir_save)
                    # with open(join(dir_save, "info.json"), "w") as f:
                    #     json.dump({"omega_n": omega_n, "n_dt": n_dt, "damping_coeff": damping_coeff, 
                    #                "step_amplitude": 1}, f)

                    osci = Oscillations(1, omega_n, damping_coeff)
                    force = [1, 0, 0]
                    t_sim, pos, _, _ = osci.rotation(t[-1], 0.01, force, x_0, v_0, a_0)
                    df_analytical = pd.DataFrame({"time": t_sim, "solution_x": pos[:, 0], "solution_y": pos[:, 1],
                                                   "solution_tors": pos[:, 2]})
                    df_analytical.to_csv(join(dir_save, "analytical_sol.dat"), index=None)
                    counter += 1

    def _simulate(
        self,
        time: np.ndarray,
        f_external: np.ndarray,
        init_pos: np.ndarray,
        init_vel: np.ndarray,
        init_accel: np.ndarray,
        scheme: str="HHT-alpha",
        alpha_HHT: float=0.1,
        ) -> ThreeDOFsAirfoil:
        airfoil = ThreeDOFsAirfoil(dir_polar=self.dir_polar, time=time)
        airfoil.pos[0, :] = init_pos
        airfoil.vel[0, :] = init_vel
        airfoil.accel[0, :] = init_accel
        
        airfoil.aero = f_external
        time_integrator = TimeIntegration()
        time_integrator._init_HHT_alpha(airfoil, alpha_HHT, time[1], **self._struc_params)
        self._struct_force._init_linear(airfoil, **self._struc_params)
        integration_func = time_integrator.get_scheme_method(scheme)
        airfoil.damp = np.zeros((time.size, 3))
        airfoil.stiff = np.zeros((time.size, 3))

        for i in range(time.size-1):
            pos, vel, accel = integration_func(airfoil, i, time[1])
            airfoil.pos[i+1, :] = pos
            airfoil.vel[i+1, :] = vel
            airfoil.accel[i+1, :] = accel

            #todo check coordinate sytsem under rotation (also check for force)
            airfoil.damp[i, :], airfoil.stiff[i, :] = self._struct_force._linear(airfoil, i)
            
        i = time.size-1
        airfoil.damp[i, :], airfoil.stiff[i, :] = self._struct_force._linear(airfoil, i)
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
        "HHT_alpha_rotation": False
    }
    dir_airfoil = "data/FFA_WA3_221"
    dir_HHT_alpha_validation = "data/HHT_alpha_validation"
    BL_scheme = "BL_openFAST_Cl_disc"
    # BL_scheme = "BL_chinese"
    HHT_alpha_case = "test"
    sep_point_test_res = 202
    period_res = 3000

    if do["separation_point_calculation"]:
        sep_point_calc(dir_airfoil, BL_scheme, resolution=sep_point_test_res)

    if do["separation_point_plots"]:
        plot_sep_points(join(dir_airfoil, "sep_point_tests", str(sep_point_test_res)), alpha_limits=(-30, 30))

    if do["BL"]:
        validation_against = "measurement"
        cases_defined_in = "cases.dat"
        # validation_against = "polar"
        # cases_defined_in = "cases_polar.dat"
        df_cases = pd.read_csv(join(dir_airfoil, "unsteady", cases_defined_in))
        for i, row in df_cases.iterrows():
            run_BeddoesLeishman(dir_airfoil, BL_scheme, validation_against, index_unsteady_data=i,
                                k=row["k"], inflow_speed=row["inflow_speed"], chord=row["chord"], 
                                amplitude=row["amplitude"], mean=row["mean"], period_res=period_res)

    if do["plot_BL_results_meas"]:
        plotter = BLValidationPlotter()
        plotter.plot_preparation(join(dir_airfoil, "preparation", BL_scheme), join(dir_airfoil, "polars.dat"))
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
        file_polar = join(dir_airfoil, "polars.dat")
        plotter = BLValidationPlotter()
        # plotter.plot_preparation(join(dir_airfoil, "preparation", BL_scheme), join(dir_airfoil, "polars.dat"))
        dir_validations = join(dir_airfoil, "validation", BL_scheme, "polar")
        df_cases = pd.read_csv(join(dir_airfoil, "unsteady", "cases_polar.dat"))
        for case_name in listdir(dir_validations):
            dir_case = join(dir_validations, case_name)
            plotter.plot_over_polar(file_polar, dir_case, period_res, df_cases.iloc[int(case_name)])
  
    if do["calc_HHT_alpha_response"] or do["plot_HHT_alpha_response"]:
        T = 10
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
        n_t_per_oscillation = [8]  # should be a multiple of 4 to get the analytical peaks right
        omega_n = [1, 2, 3, 4]
        root_dir = join(dir_HHT_alpha_validation, "undamped")
        validator = HHTalphaValidator(dir_airfoil, root_dir)
        validator.simulate_undamped(n_t_per_oscillation, omega_n, n_oscillations, scheme="HHT-alpha")
        PostHHT_alpha().amplitude_and_period(root_dir)
        HHTalphaPlotter().sol_and_sim(root_dir, ["ampl", "freq"])

    if do["HHT_alpha_damped"]:
        n_oscillations = 10
        n_t_per_oscillation = [32]  # should be a multiple of 4 to get the analytical peaks right
        omega_n = [1, 2, 3, 4]
        damping_coeffs = [0.1, 0.2, 0.3]
        root_dir = join(dir_HHT_alpha_validation, "damped")
        validator = HHTalphaValidator(dir_airfoil, root_dir)
        validator.simulate_damped(omega_n, damping_coeffs, n_t_per_oscillation, n_oscillations, scheme="HHT-alpha")
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
        validator.simulate_forced(damping_coeffs, n_t_per_oscillation, force_freq, n_oscillations, scheme="HHT-alpha")
        validator.simulate_forced(damping_coeffs, n_t_per_oscillation, force_freq, n_oscillations,
                                   scheme="HHT-alpha-adapted")
        PostHHT_alpha().amplitude_and_period(root_dir)
        HHTalphaPlotter().sol_and_sim(root_dir, ["ampl", "freq"])
    
    if do["HHT_alpha_forced_composite"]:
        n_oscillations = 4
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
                                            n_oscillations, x_0, v_0, scheme="HHT-alpha")
        validator.simulate_forced_composite(damping_coeff, n_t_per_oscillation, force_freq, force_ampl,
                                            n_oscillations, x_0, v_0, scheme="HHT-alpha-adapted")
        PostHHT_alpha().amplitude_and_period(root_dir)
        HHTalphaPlotter().sol_and_sim(root_dir, ["ampl", "freq"])

    if do["HHT_alpha_step"]:

        n_oscillations = 10
        n_t_per_oscillation = [12]  # should be a multiple of 4 to get the analytical peaks right
        omega_n = [1, 2, 3, 4]
        damping_coeffs = [0.1, 0.2, 0.3]
        root_dir = join(dir_HHT_alpha_validation, "step")
        validator = HHTalphaValidator(dir_airfoil, root_dir)
        validator.simulate_step(omega_n, damping_coeffs, n_t_per_oscillation, n_oscillations, scheme="HHT-alpha")
        PostHHT_alpha().amplitude_and_period(root_dir)
        HHTalphaPlotter().sol_and_sim(root_dir, ["ampl", "freq"])

    if do["HHT_alpha_rotation"]:
        n_oscillations = 10
        n_t_per_oscillation = [128]  # should be a multiple of 4 to get the analytical peaks right
        omega_n = [np.asarray([np.sqrt(2), 1, 1])]
        damping_coeffs = [0.2*np.ones(3)]
        root_dir = join(dir_HHT_alpha_validation, "rotation")
        validator = HHTalphaValidator(dir_airfoil, root_dir)
        validator.simulate_rotation(omega_n, damping_coeffs, n_t_per_oscillation, n_oscillations, scheme="HHT-alpha")
        HHTalphaPlotter().rotation(root_dir)