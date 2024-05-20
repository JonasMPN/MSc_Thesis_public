from calculation_utils import get_inflow
from calculations import ThreeDOFsAirfoil
from post_calculations import PostCaluculations
from helper_functions import Helper
import numpy as np
from os.path import join
from os import listdir
import json
from itertools import product
from multiprocessing import Pool
import pandas as pd
helper = Helper()


def prepare_multiprocessing_input(
        n_processes: int,
        always_use: list[float|str],
        *args: list[float],
        add_call_number: bool=False
        ):
    args_passed = max([len(arg) for arg in args])
    split_size = max(1, args_passed//n_processes)
    n_not_accounted = args_passed-split_size*n_processes
    input_args, idx_advanced = [], 0
    for n in range(n_processes if n_not_accounted >= 0 else args_passed):
        idx_start = n*split_size+idx_advanced
        idx_advanced += 1 if n < n_not_accounted else 0
        idx_end = split_size*(n+1)+idx_advanced
        process_input = [arg[idx_start:idx_end] if len(arg) != 0 else [] for arg in args]
        if add_call_number:
            process_input += [[i for i in range(idx_start, idx_end)]]
        process_input += always_use 
        input_args.append(process_input)
    return input_args


def run_forced(
        amplitude: list[float]|float,
        angle_of_attack: list[float]|float,
        case_id: list[int],
        dir_airfoil: str,
        root: str,
        aero_scheme: str,
        t: np.ndarray,
        frequency: float,
        structure_data: dict[str, float],
        helper: Helper,
        ):
    # changes in the order of the arguments must be reflected in prepare_multiprocessing_input() and calls thereof!'
    for ampl, aoa, i in zip(amplitude, angle_of_attack, case_id):
        case_dir = helper.create_dir(join(root, str(i)))[0]
        with open(join(case_dir, "section_data.json"), "w") as f:
            json.dump(structure_data, f, indent=4)
        omega = 2*np.pi*frequency 
        pos = {"x": ampl*np.sin(omega*t)}
        vel = {"x": ampl*omega*np.cos(omega*t)}
        accel = {"x": -ampl*omega**2*np.sin(omega*t)}
        inflow = get_inflow(t, [(0, 3, 1, aoa)], init_velocity=1)

        NACA_643_618 = ThreeDOFsAirfoil(dir_airfoil, t, verbose=False)
        # set the calculation scheme for the aerodynamic forces
        if aero_scheme == "qs":
            NACA_643_618.set_aero_calc()
        elif aero_scheme == "BL_chinese":
            NACA_643_618.set_aero_calc("BL_chinese", A1=0.3, A2=0.7, b1=0.14, b2=0.53, pitching_around=0.25, 
                                       alpha_at=0.75, chord=structure_data["chord"])
        elif aero_scheme == "BL_openFAST_Cl_disc":
            NACA_643_618.set_aero_calc("BL_openFAST_Cl_disc", A1=0.165, A2=0.335, b1=0.0445, b2=0.3, 
                                       pitching_around=0.25, alpha_at=0.75, chord=structure_data["chord"])
        # set the calculation scheme for the structural damping and stiffness forces
        NACA_643_618.set_struct_calc("linear", **structure_data)
        # set the time integration scheme
        NACA_643_618.set_time_integration("HHT-alpha-adapted", alpha=0.1, dt=t[1], **structure_data)
        NACA_643_618.simulate_along_path(inflow, pos, vel, accel)  # perform simulation
        NACA_643_618.save(case_dir)  # save simulation results


def run_forced_parallel(
        dir_airfoil: str,
        root: str,
        n_processes: int,
        aero_scheme: str,
        t: np.ndarray,
        structure_data: dict[str, float],
        frequency: float,
        amplitude: list[float]|float,
        angle_of_attack: list[float]|float,
        ):
    root_dir = helper.create_dir(root)[0]
    amplitude = amplitude if isinstance(amplitude, list) else [amplitude]
    angle_of_attack = angle_of_attack if isinstance(angle_of_attack, list) else [angle_of_attack]
    combinations = [(ampl, aoa) for ampl, aoa in product(amplitude, angle_of_attack)]
    amplitudes = []
    alphas = []
    for combi in combinations:
        amplitudes.append(combi[0])
        alphas.append(combi[1])
    dict_combinations = {"amplitude": amplitudes, "alpha": alphas}
    pd.DataFrame(dict_combinations).to_csv(join(root_dir, "combinations.dat"), index=None)

    always_use = [dir_airfoil, root, aero_scheme, t, frequency, structure_data, helper]
    input_args = prepare_multiprocessing_input(n_processes, always_use, amplitudes, alphas, add_call_number=True)
    with Pool(processes=n_processes) as pool:
        pool.starmap(run_forced, input_args)
    

def post_calc(
        case_dirs: list[str],
        alpha_lift: str,
        ):
    for case_dir in case_dirs:
        post_calc = PostCaluculations(dir_sim_res=case_dir, alpha_lift=alpha_lift)
        post_calc.project_data()  # projects the simulation's x-y-rot_z data into different coordinate systems such as
        # edgewise-flapwise-rot_z or drag-lift_rot_z.
        post_calc.power()  # calculate and save the work done by the aerodynamic and structural damping forces
        post_calc.kinetic_energy()  # calculate the edgewise, flapwise, and rotational kinetic energy
        post_calc.potential_energy()  # calculate the edgewise, flapwise, and rotational potential energy
        post_calc.work_per_cycle()


def post_calculations_parallel(
        root_cases: str,
        alpha_lift: str,
        n_processes: int,
        ):
    case_dirs = [join(root_cases, case_name) for case_name in listdir(root_cases) if case_name not in 
                 ["combinations.dat", "plots"]]
    always_use = [alpha_lift]
    input_args = prepare_multiprocessing_input(n_processes, always_use, case_dirs)
    with Pool(processes=n_processes) as pool:
        pool.starmap(post_calc, input_args)
    
    
    