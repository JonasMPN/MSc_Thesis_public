from calculations import ThreeDOFsAirfoil
from post_calculations import PostCaluculations
from calculation_utils import get_inflow
from calculation_wrappers import run_forced_parallel_from_free_case, post_calculations_parallel, run_free_parallel
import numpy as np
import pandas as pd
import json
from plotting import Plotter, Animator, combined_forced, combined_LOC_amplitude
from os.path import join
from os import listdir
from helper_functions import Helper
from itertools import product
from defaults import Staeblein
helper = Helper()

def set_do(
        do: dict,
        sim_post: bool=False, 
        sim_post_res: bool=False,
        sim_post_res_fill: bool=False,
        fill: bool=False,
        animate: bool=False,
        coupled_timeseries: bool=False
):
    if sim_post:
        do["simulate"] = True
        do["post_calc"] = True
    if sim_post_res:
        do["simulate"] = True
        do["post_calc"] = True
        do["plot_results"] = True
    if sim_post_res_fill:
        do["simulate"] = True
        do["post_calc"] = True
        do["plot_results"] = True
        do["plot_results_fill"] = True
    if fill:
        do["plot_results_fill"] = True
    if animate:
        do["animate_results"] = True
    if coupled_timeseries:
        do["plot_coupled_timeseries"] = True
    return do

do = {
    "simulate": False,  # run a simulation
    "post_calc": False,  # peform post calculations
    "plot_results": True,  # plot results,
    "plot_results_fill": False,
    "plot_coupled_timeseries": False,
    "animate_results": False,
    "plot_combined_forced": False,
    "plot_combined_LOC_amplitude": False
}

do = set_do(do=do,
    sim_post = False,
    sim_post_res = False,
    sim_post_res_fill = False,
    fill= False,
    animate = False,
    coupled_timeseries = False,
)

# root = "data/NACA_643_618"  # set which airfoil polars to use. Simulatenously defines to root for the simulation data
root = "data/FFA_WA3_221"  # set which airfoil polars to use. Simulatenously defines to root for the simulation data

sim_type = "free"
# sim_type = "free_parallel"
# sim_type = "forced"
n_processes = 6  # for sim_type="forced" and sim_type="free_parallel".

# aero_scheme = "qs"
aero_scheme = "BL_openFAST_Cl_disc"
# aero_scheme = "BL_openFAST_Cl_disc_f_scaled"
# aero_scheme = "BL_AEROHOR"
# aero_scheme = "BL_first_order_IAG2"
# aero_scheme = "BL_Staeblein"

file_polar = "polars_new.dat"
# file_polar = "polars_staeblein.dat"

alpha_lift = "alpha_qs" if aero_scheme == "qs" else "alpha_eff"

# case_name = "test_BL_openFAST"
# case_name = "test_qs"
# case_name = "initial_y_and_tors_set"
# case_name = "initial_at_steady_state"
case_name = "LCO_25"
# case_name = "test"
# case_name = "LCO_tries_25_polar_staeblein"
# case_name = "flutter"
# case_name = "val_staeblein_main_polar"
# case_name = "val_staeblein_staeblein_polar"

# for forced
# motion_from_axes = ["edge", "flap", "tors"]  # adapt 'cs' for changes here
# coordinate_system = "ef"
motion_from_axes = ["x", "y", "tors"]  # adapt 'cs' for changes here
coordinate_system = "xy"

staeblein = Staeblein()
structure_def = {  # definition of structural parameters
    "chord": staeblein.c,
    "coordinate_system": "xy",
    # "inertia": [1, 1, 1], # [linear, linear, torsion]
    "inertia": staeblein.inertia.tolist(), # [linear, linear, torsion]
    # "damping": [1, 0, 0], # [x_0, x_1, torsion]
    "damping": staeblein.damping.tolist(), 
    # "stiffness": [1, 1, 1],  # [x_0, x_1, torsion]
    "stiffness": staeblein.stiffness.tolist(),
}

# freq = 0.61  # in Hz
# damp_ratio = 0.342
# damping_coeff = freq*damp_ratio
# damped_freq = freq*np.sqrt(1-damping_coeff**2)
# x0 = 0.01
# nat_freq = np.sqrt(structure_def["stiffness"][1]/structure_def["inertia"][1])
# from calculations import Oscillations
# osci = Oscillations(structure_def["inertia"][1], nat_freq, damp_ratio)

def main(simulate, post_calc, plot_results, plot_results_fill, plot_coupled_timeseries, 
         animate_results, plot_combined_forced, plot_combined_LOC_amplitude):
    if simulate:
        case_dir = helper.create_dir(join(root, "simulation", sim_type, aero_scheme, case_name))[0]
        # the following definition of NACA_643_618 is mostly used for the definition of the structure. The time
        # definition has no influence on "run_forced_parallel". It only influences "free" and "free_parallel".
        # Otherwise, the NACA definition is used for "free" and "free_parallel" (not yet, do in future).
        dt = 0.01  # if dt<1e-5, the dt rounding in compress_oscillation() in calculation_utils.py needs to be adapted
        t_end = 400
        save_last = 400  # the last 'save_last' seconds of the simulation will be saved
        t = dt*np.arange(int(t_end/dt))  # set the time array for the simulation
        NACA_643_618 = ThreeDOFsAirfoil(t, verbose=False)
        # set the calculation scheme for the aerodynamic forces
        chord = structure_def["chord"]
        if aero_scheme == "qs":
            NACA_643_618.set_aero_calc(root, file_polar=file_polar, scheme="quasi_steady", chord=chord, 
                                       pitching_around=0.25, alpha_at=0.75)
        elif aero_scheme == "BL_AEROHOR":
            NACA_643_618.set_aero_calc(root, file_polar=file_polar, scheme="BL_AEROHOR", A1=0.3, A2=0.7, b1=0.14,
                                       b2=0.53, pitching_around=0.25, alpha_at=0.75, chord=chord, alpha_critical=14.2)
        elif aero_scheme == "BL_first_order_IAG2":
            NACA_643_618.set_aero_calc(root, file_polar=file_polar, scheme="BL_first_order_IAG2", A1=0.3, A2=0.7, 
                                       b1=0.7, b2=0.53, pitching_around=0.25, alpha_at=0.75, chord=chord, 
                                       alpha_critical=14.2)
        elif aero_scheme == "BL_openFAST_Cl_disc":
            NACA_643_618.set_aero_calc(root, file_polar=file_polar, scheme="BL_openFAST_Cl_disc", A1=0.165, A2=0.335,
                                       b1=0.0445, b2=0.3, pitching_around=0.25, alpha_at=0.75, chord=chord)
        elif aero_scheme == "BL_Staeblein":
            NACA_643_618.set_aero_calc(root, file_polar="polars_staeblein.dat", scheme="BL_Staeblein", A1=0.165, 
                                       A2=0.335, b1=0.0445, b2=0.3, chord=chord, 
                                       pitching_around=(0.25+staeblein.e_ac/chord), alpha_at=0.75)
            structure_def["inertia"] = np.diag(structure_def["inertia"])
            structure_def["inertia"][1, 2] = staeblein.inertia[0]*staeblein.e_cg  # no minus because rot direction
            structure_def["inertia"][2, 1] = staeblein.inertia[0]*staeblein.e_cg  # is opposite to paper!
            structure_def["damping"] = np.diag(structure_def["damping"])
            structure_def["stiffness"] = np.diag(structure_def["stiffness"])
            flap_twist_coupling = -0.
            coupling_stiffness = flap_twist_coupling*np.sqrt(structure_def["stiffness"][1, 1]*
                                                            structure_def["stiffness"][2, 2])
            structure_def["stiffness"][1, 2] = coupling_stiffness
            structure_def["stiffness"][2, 1] = coupling_stiffness
        elif aero_scheme == "BL_openFAST_Cl_disc_f_scaled":
            NACA_643_618.set_aero_calc(root, file_polar=file_polar, scheme="BL_openFAST_Cl_disc_f_scaled", A1=0.165, 
                                       A2=0.335, b1=0.0445, b2=0.3, pitching_around=0.25, alpha_at=0.75, chord=chord)
        else:
            raise NotImplementedError(f"'aero_scheme'={aero_scheme}")
        
        if sim_type == "free":
            with open(join(case_dir, "section_data.json"), "w") as f:
                json.dump(structure_def, f, indent=4)
            alpha = 17.5
            # alpha = None

            inflow = get_inflow(t, [(0, 0, 20, alpha)], init_velocity=0.1)
            approx_steady_x = True
            approx_steady_y = True
            approx_steady_tors = True if aero_scheme != "BL_AEROHOR" else False
            ffile_polar = join(root, file_polar)
    
            init_data = NACA_643_618.approximate_steady_state(ffile_polar, inflow[0, :], structure_def["chord"],  
                                                              structure_def["stiffness"], x=approx_steady_x,  
                                                              y=approx_steady_y, torsion=approx_steady_tors, alpha=alpha)
            init_pos, f_init, inflow_angle = init_data
            NACA_643_618.aero[-1, :] = f_init  # needed if an HHT-alpha algorithm is used
            
            if alpha is not None:
                inflow_vel = np.linalg.norm(inflow[0, :])
                inflow = get_inflow(t, [(0, 0, inflow_vel, inflow_angle)], init_velocity=45)
                
            # initial conditions for position in [x, y, rotation in rad]
            # init_pos = np.zeros(3)
            # init_pos[0] += 5*np.cos(init_pos[2])
            init_pos[0] *= 1.3
            # init_pos[0] += 1.3
            # init_pos[1] += 5*np.sin(init_pos[2])
            # init_pos[1] += 4
            # init_pos[2] -= 0.0001

            # initial conditions for the velocity in [x, y, rotation in rad/s]
            init_vel = np.zeros(3)  
            
            # set the calculation scheme for the structural damping and stiffness forces
            NACA_643_618.set_struct_calc("linear_xy", **structure_def)
            # NACA_643_618.set_time_integration("eE", **structure_def)
            NACA_643_618.set_time_integration("HHT-alpha-xy-adapted", alpha=0.1, dt=t[1]-t[0], **structure_def)
            NACA_643_618.simulate(inflow, init_pos, init_vel)  # perform simulation
            # damped = osci.damped(t)
            # vel = np.r_[0, (damped[1:]-damped[:-1])/t[1]]
            # NACA_643_618.simulate_along_path(inflow, "xy", init_pos, init_vel, 
            #                                  {
            #                                      0: np.zeros_like(t), 
            #                                     #  1: damped, 
            #                                     #  1: np.zeros_like(t), 
            #                                     #  2: np.zeros_like(t)
            #                                  }, 
            #                                  {
            #                                      0: np.zeros_like(t), 
            #                                     #  1: damped, 
            #                                     #  1: np.zeros_like(t), 
            #                                     #  2: np.zeros_like(t)
            #                                  }
            #                                  )  # perform simulation
            NACA_643_618.save(case_dir, save_last=save_last)  # save simulation results
        
        if sim_type == "free_parallel":
            velocities = [5, 7.5, 10, 12.5, 15, 17.5, 20, 22.5, 25, 27.5, 30, 32.5, 35]
            angle_of_attacks = [-20, -17.5, -15, -12.5, -10, -7.5, -5, -2.5, 0, 2.5, 5, 7.5, 10, 12.5, 15, 17.5, 20]
            # angle_of_attacks = [20, 22.5, 25]
            run_free_parallel(root, file_polar, case_dir, n_processes, aero_scheme, t, structure_def, velocities, 
                              angle_of_attacks, save_last)

        if sim_type == "forced":
            root_cases = case_dir
            dir_free = join(root, "simulation", "free", aero_scheme, case_name)
            ff_free_motion = join(dir_free, "general.dat")
            
            df_gen = pd.read_csv(ff_free_motion)
            # motion_types = ["pos", "vel", "accel"]
            motion_types = ["pos", "vel"]
            motion_cols = ["_".join((motion, axes)) for motion, axes in product(motion_types, motion_from_axes)]

            # forced_amplitude = [0.2, 0.4, 0.6, 0.8, 1]
            # forced_amplitude_fac = [0.1, 0.2, 0.5, 0.8, 0.9, 1, 1.1, 1.2, 2, 4]
            forced_amplitude_fac = [1]
            forced_aoa = [20]
            periods = 3
            period_res = 1000
            inflow_velocity = np.linalg.norm(df_gen[["inflow_x", "inflow_y"]].to_numpy()[-1, :])
            run_forced_parallel_from_free_case(root, file_polar, ff_free_motion, motion_cols, coordinate_system, 
                                               root_cases, n_processes, aero_scheme, inflow_velocity, periods, 
                                               period_res, structure_def, forced_amplitude_fac, forced_aoa)

    if post_calc:
        aoa_thresholds = {
            "alpha_qs": 25,
            "alpha_eff": 25
        }
        if sim_type == "free":
            post_calc = PostCaluculations(dir_sim_res=join(root, "simulation", "free", aero_scheme, case_name),
                                          alpha_lift=alpha_lift, coordinate_system_structure=coordinate_system)
            post_calc.project_data()  # projects the simulation's x-y-rot_z data into different coordinate systems such as
            post_calc.check_angle_of_attack(**aoa_thresholds)
            # edgewise-flapwise-rot_z or drag-lift_rot_z.
            post_calc.power()  # calculate and save the work done by the aerodynamic and structural damping forces
            post_calc.kinetic_energy()  # calculate the edgewise, flapwise, and rotational kinetic energy
            post_calc.potential_energy()  # calculate the edgewise, flapwise, and rotational potential energy
            post_calc.work_per_cycle()

        if "forced" in sim_type or sim_type == "free_parallel":
            root_cases = join(root, "simulation", sim_type, aero_scheme, case_name)
            
            # to do post calculations on all, uncomment the next line
            # post_calculations_parallel(root_cases, alpha_lift, n_processes, aoa_thresholds)

            # to do post calculations on only one, use the following lines
            case_id = 170
            post_calc = PostCaluculations(dir_sim_res=join(root, "simulation", sim_type, aero_scheme, case_name, 
                                                           str(case_id)),
                                          alpha_lift=alpha_lift, coordinate_system_structure=coordinate_system)
            post_calc.project_data()  # projects the simulation's x-y-rot_z data into different coordinate systems such as
            post_calc.check_angle_of_attack(**aoa_thresholds)
            # edgewise-flapwise-rot_z or drag-lift_rot_z.
            post_calc.power()  # calculate and save the work done by the aerodynamic and structural damping forces
            post_calc.kinetic_energy()  # calculate the edgewise, flapwise, and rotational kinetic energy
            post_calc.potential_energy()  # calculate the edgewise, flapwise, and rotational potential energy
            post_calc.work_per_cycle()

    if plot_results:
        trailing_every = 5
        time_frame = (0, 400)
        file_profile = join(root, "profile.dat")  # define path to file containing the profile shape data
        if sim_type == "free":
            dir_sim = join(root, "simulation", "free", aero_scheme, case_name)  # define path to the root of the simulation results
        elif "forced" in sim_type or "parallel" in sim_type:
            case_id = "170"
            dir_sim = join(root, "simulation", sim_type, aero_scheme, case_name, case_id)
        dir_plots = join(dir_sim, "plots")  # define path to the directory that the results are plotted into
        plotter = Plotter(file_profile, dir_sim, dir_plots, structure_def["coordinate_system"]) 
        # plotter.force(trailing_every=trailing_every, time_frame=time_frame)  # plot various forces of the simulation
        # plotter.energy(trailing_every=trailing_every, time_frame=time_frame)  # plot various energies and work done by forces of the simulation
        if "BL" in aero_scheme:
            plotter.Beddoes_Leishman(trailing_every=trailing_every, time_frame=time_frame)
        plotter.damping(("alpha_eff", [25, 35]), polar=["data/FFA_WA3_221/polars_new.dat", 5])
    
    if plot_results_fill:
        trailing_every = 5
        peak_distance = 90
        file_profile = join(root, "profile.dat")  # define path to file containing the profile shape data
        if sim_type == "free":
            dir_sim = join(root, "simulation", "free", aero_scheme, case_name)  # define path to the root of the simulation results
            # peak_distance = 400
        elif "forced" in sim_type:
            case_id = "0"
            dir_sim = join(root, "simulation", sim_type, aero_scheme, case_name, case_id)
            # peak_distance = 90
        dir_plots = join(dir_sim, "plots")  # define path to the directory that the results are plotted into
        plotter = Plotter(file_profile, dir_sim, dir_plots, structure_def["coordinate_system"]) 
        alpha = 0.3
        # plot various forces of the simulation
        plotter.force_fill(trailing_every=trailing_every, alpha=alpha, peak_distance=peak_distance) 
        # plot various energies and work done by forces of the simulation
        plotter.energy_fill(trailing_every=trailing_every, alpha=alpha, peak_distance=peak_distance)  
        if "BL" in aero_scheme:
            plotter.Beddoes_Leishman_fill(alpha=alpha, peak_distance=peak_distance)

    if plot_coupled_timeseries:
        file_profile = join(root, "profile.dat")  # define path to file containing the profile shape data
        if sim_type == "free":
            dir_sim = join(root, "simulation", "free", aero_scheme, case_name)  # define path to the root of the simulation results
        elif "forced" in sim_type:
            case_id = "0"
            dir_sim = join(root, "simulation", sim_type, aero_scheme, case_name, case_id)
        dir_plots = join(dir_sim, "plots")  # define path to the directory that the results are plotted into
        plotter = Plotter(file_profile, dir_sim, dir_plots, structure_def["coordinate_system"]) 
        plotter.couple_timeseries()

    if animate_results:
        time_frame = (398.5, 400)
        dt_per_frame = 0.005
        file_profile = join(root, "profile.dat")  # define path to file containing the profile shape data
        dir_sim = join(root, "simulation", sim_type, aero_scheme, case_name)   # define path to the root of the simulation results
        if sim_type == "forced":
            case_id = 0
            dir_sim = join(dir_sim, str(case_id))
        dir_plots = join(dir_sim, "plots")  # define path to the directory that the results are plotted into
        animator = Animator(file_profile, dir_sim, dir_plots) 
        # animate various forces of the simulation
        animator.force(angle_lift=alpha_lift, arrow_scale_forces=0.01, arrow_scale_moment=.5,
                       plot_qc_trailing_every=2, keep_qc_trailing=200, time_frame=time_frame, dt_per_frame=dt_per_frame)
        # plot various energies and work done by forces 
        animator.energy(angle_lift=alpha_lift, arrow_scale_forces=0.01, arrow_scale_moment=.5,
                        plot_qc_trailing_every=2, keep_qc_trailing=200, time_frame=time_frame, 
                        dt_per_frame=dt_per_frame)
        animator.Beddoes_Leishman(angle_lift=alpha_lift, arrow_scale_forces=0.01, arrow_scale_moment=.5,
                                  plot_qc_trailing_every=2, keep_qc_trailing=200, time_frame=time_frame,
                                  dt_per_frame=dt_per_frame)

    if plot_combined_forced:
        combined_forced(join(root, "simulation", sim_type, aero_scheme, case_name))

    if plot_combined_LOC_amplitude:
        combined_LOC_amplitude(join(root, "simulation", sim_type, aero_scheme, case_name))

if __name__ == "__main__":
    main(**do) 