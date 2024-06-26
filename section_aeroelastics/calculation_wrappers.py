from calculation_utils import get_inflow, compress_oscillation, reconstruct_from_file, zero_oscillations
from calculations import ThreeDOFsAirfoil
from post_calculations import PostCaluculations
from helper_functions import Helper
import numpy as np
from os.path import join, isfile
from os import listdir
import json
from itertools import product
from multiprocessing import Pool
import pandas as pd
helper = Helper()

import matplotlib.pyplot as plt

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
        coordinate_system: str,
        time: np.ndarray,
        base_pos: dict[str, np.ndarray],
        base_vel: dict[str, np.ndarray],
        # base_accel: dict[str, np.ndarray],
        mean_pos: dict[str, float],
        mean_vel: dict[str, float],
        # mean_accel: dict[str, float],
        structure_data: dict[str, float],
        helper: Helper,
        ):
    # changes in the order of the arguments must be reflected in prepare_multiprocessing_input() and calls thereof!'
    for ampl, aoa, i in zip(amplitude, angle_of_attack, case_id):
        case_dir = helper.create_dir(join(root, str(i)))[0]
        with open(join(case_dir, "section_data.json"), "w") as f:
            json.dump(structure_data, f, indent=4)

        pos = {}
        vel = {}
        # accel = {}
        for axes, position in base_pos.items():
            # it is assumed that for a given axes, position, velocity, AND acceleration are given.
            pos[axes] = ampl*position+mean_pos[axes]
            vel[axes] = ampl*base_vel[axes]+mean_vel[axes]
            # accel[axes] = ampl*base_accel[axes]+mean_accel[axes]
            
        inflow = get_inflow(time, [(0, 3, 1, aoa)], init_velocity=1)

        NACA_643_618 = ThreeDOFsAirfoil(dir_airfoil, time, verbose=False)
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
        NACA_643_618.set_time_integration("HHT-alpha-adapted", alpha=0.1, dt=time[1], **structure_data)
        # NACA_643_618.simulate_along_path(inflow, coordinate_system, pos, vel, accel)  # perform simulation
        NACA_643_618.simulate_along_path(inflow, coordinate_system, pos, vel)  # perform simulation
        NACA_643_618.save(case_dir)  # save simulation results


def _run_forced_parallel(
        dir_airfoil: str,
        root: str,
        aero_scheme: str,
        structure_data: dict[str, float],
        n_processes: int,
        coordinate_system: str,
        amplitude: dict[int, list[float]|float],
        angle_of_attack: list[float]|float,
        time: np.ndarray,
        base_pos: dict[str, np.ndarray],
        base_vel: dict[str, np.ndarray],
        # base_accel: dict[str, np.ndarray],
        mean_pos: dict[str, float],
        mean_vel: dict[str, float],
        # mean_accel: dict[str, float],
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

    always_use = [dir_airfoil, root, aero_scheme, coordinate_system,
                  time, base_pos, base_vel, mean_pos, mean_vel, 
                #   time, base_pos, base_vel, base_accel, mean_pos, mean_vel, mean_accel, 
                  structure_data, helper]
    input_args = prepare_multiprocessing_input(n_processes, always_use, amplitudes, alphas, add_call_number=True)
    with Pool(processes=n_processes) as pool:
        pool.starmap(run_forced, input_args)


def run_forced_parallel_from_free_case(
        dir_airfoil: str,
        ffile_free_motion: str,
        motion_cols: list[str],
        coordinate_system: str,
        root: str,
        n_processes: int,
        aero_scheme: str,
        periods: int,
        period_resolution: int,
        structure_data: dict[str, float],
        amplitude: dict[int, list[float]|float],
        angle_of_attack: list[float]|float):
    ffile_free_motion = ffile_free_motion.replace("\\", "/")
    dir_base_motion = ffile_free_motion[:ffile_free_motion.rfind("/")]
    motion_file = ffile_free_motion.split("/")[-1]
    motion_filename = motion_file[:motion_file.rfind(".")]
    ffile_compressed_motion = join(dir_base_motion, f"{motion_filename}_compressed.json")

    # if not isfile(ffile_compressed_motion):
    if True:
        compress_oscillation(ffile_free_motion, ffile_compressed_motion, motion_cols, period_res=period_resolution)
    else:
        with open(ffile_compressed_motion, "r") as f:
            compressed_data = json.load(f)
            old_period_res = compressed_data.values()[0]["N"]
            if old_period_res != period_resolution:
                compress_oscillation(ffile_free_motion, ffile_compressed_motion, motion_cols,
                                     period_res=period_resolution)
    base_motion, mean = reconstruct_from_file(ffile_compressed_motion, separate_mean=True)
    base_time = base_motion[motion_cols[0]]["time"]
    osci_time, base_osci = zero_oscillations(base_time, periods, **base_motion)

    axes_name_to_axes_idx = {
        "edge": 0,
        "x": 0,
        "flap": 1,
        "y": 1,
        "tors": 2
    }
    
    base_pos = {}
    base_vel = {}
    base_accel = {}
    mean_pos = {}
    mean_vel = {}
    mean_accel = {}
    # implemented_motion = ["vel", "pos", "accel"]
    implemented_motion = ["vel", "pos"]
    implemented_axes = ["edge", "flap", "x", "y", "tors"]
    for param, values in base_osci.items():
        motion, axes = param.split("_")
        if motion not in implemented_motion:
            raise ValueError("Keys of 'base_osci' must be named 'motion_axes', where 'motion' must be one of "
                             f"'{implemented_motion}' but was {motion}.")
        if axes not in implemented_axes:
            raise ValueError("Keys of 'base_osci' must be named 'motion_axes', where 'motion' must be one of "
                             f"'{implemented_axes}' but was {axes}.")
        
        if motion == "pos":
            mean_pos[axes_name_to_axes_idx[axes]] = mean[param]
            base_pos[axes_name_to_axes_idx[axes]] = values
        elif motion == "vel":
            mean_vel[axes_name_to_axes_idx[axes]] = mean[param]
            base_vel[axes_name_to_axes_idx[axes]] = values
        # elif motion == "accel":
        #     mean_accel[axes_name_to_axes_idx[axes]] = mean[param]
        #     base_accel[axes_name_to_axes_idx[axes]] = values

    _run_forced_parallel(
        dir_airfoil,
        root,
        aero_scheme,
        structure_data,
        n_processes,
        coordinate_system,
        amplitude,
        angle_of_attack,
        osci_time,
        base_pos,
        base_vel,
        # base_accel,
        mean_pos,
        mean_vel,
        # mean_accel,
    )
    

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
    
    
    