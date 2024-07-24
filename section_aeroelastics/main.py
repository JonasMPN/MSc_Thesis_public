from calculations import ThreeDOFsAirfoil
from post_calculations import PostCaluculations
from calculation_utils import get_inflow
from calculation_wrappers import run_forced_parallel_from_free_case, post_calculations_parallel
import numpy as np
import pandas as pd
import json
from plotting import Plotter, Animator, combined_forced
from os.path import join
from os import listdir
from helper_functions import Helper
from itertools import product
from defaults import Staeblein
helper = Helper()

do = {
    "simulate": False,  # run a simulation
    "forced": False,
    "post_calc": False,  # peform post calculations
    "plot_results": False,  # plot results,
    "plot_results_fill": False,
    "plot_coupled_timeseries": True,
    "animate_results": False,
    "plot_combined_forced": False
}

# root = "data/NACA_643_618"  # set which airfoil polars to use. Simulatenously defines to root for the simulation data
root = "data/FFA_WA3_221"  # set which airfoil polars to use. Simulatenously defines to root for the simulation data

# sim_type = "free"
sim_type = "forced"

# aero_scheme = "qs"
aero_scheme = "BL_openFAST_Cl_disc"
# aero_scheme = "BL_Staeblein"

file_polar = "polars_new.dat"
# file_polar = "polars_staeblein.dat"

alpha_lift = "alpha_qs" if aero_scheme == "qs" else "alpha_eff"

# case_name = "test_BL_openFAST"
# case_name = "test_qs"
# case_name = "initial_y_and_tors_set"
# case_name = "initial_at_steady_state"
case_name = "LCO_tries_25"
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

def main(simulate, forced, post_calc, plot_results, plot_results_fill, plot_coupled_timeseries, 
         animate_results, plot_combined_forced):
    if simulate:
        case_dir = helper.create_dir(join(root, "simulation", "free", aero_scheme, case_name))[0]
        with open(join(case_dir, "section_data.json"), "w") as f:
            json.dump(structure_def, f, indent=4)
        dt = 0.005  # if dt<1e-5, the dt rounding in compress_oscillation() in calculation_utils.py needs to be adapted
        t_end = 400
        t = dt*np.arange(int(t_end/dt))  # set the time array for the simulation
        NACA_643_618 = ThreeDOFsAirfoil(t, verbose=False)

        # get inflow at each time step. The below line creates inflow that has magnitude 1 and changes from 0 deg to 40
        # deg in the first 3s.
        alpha = 9
        # alpha = None

        # inflow = get_inflow(t, [(0, 0, 5, 7)], init_velocity=1)
        inflow = get_inflow(t, [(0, 0, 25, alpha)], init_velocity=0.1)

        # inflow = get_inflow(t, [(0, 0, 184.5, 0)], init_velocity=0.1)  # paper flutter speed
        # inflow = get_inflow(t, [(0, 0, 220.3, 0)], init_velocity=0.1)  # section flutter with paper params
        # inflow = get_inflow(t, [(0, 0, 186.4, 0)], init_velocity=0.1)  # section flutter updated torsion freq
        
        # inflow = get_inflow(t, [(0, 0, 169, 0)], init_velocity=0.1)  # paper flutter -0.4 pitch-flap coupling
        # inflow = get_inflow(t, [(0, 0, 216.3, 0)], init_velocity=0.1)  # section flutter -0.4 pitch-flap coupling

        # inflow = get_inflow(t, [(0, 0, 123.2, 0)], init_velocity=0.1)  # paper divergence 0.1 pitch-flap coupling
        # inflow = get_inflow(t, [(0, 0, 88.9, 0)], init_velocity=0.1)  # paper divergence 0.2 pitch-flap coupling
        # inflow = get_inflow(t, [(0, 0, 59.6, 0)], init_velocity=0.1)  # paper divergence 0.4 pitch-flap coupling
        # inflow = get_inflow(t, [(0, 0, 135, 0)], init_velocity=0.1)  # section divergence 0.4 pitch-flap coupling

        # set the calculation scheme for the aerodynamic forces
        chord = structure_def["chord"]
        if aero_scheme == "qs":
            NACA_643_618.set_aero_calc(root, file_polar=file_polar, scheme="quasi_steady", chord=chord, 
                                       pitching_around=0.25, alpha_at=0.75)
        elif aero_scheme == "BL_chinese":
            NACA_643_618.set_aero_calc(root, file_polar=file_polar, scheme="BL_chinese", A1=0.3, A2=0.7, b1=0.14,
                                       b2=0.53, pitching_around=0.25, alpha_at=0.75, chord=chord)
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
        else:
            raise NotImplementedError(f"'aero_scheme'={aero_scheme} not implemented.")
        
        approx_steady_x = True
        approx_steady_y = True
        approx_steady_tors = True
        ffile_polar = join(root, file_polar)
        init_data = NACA_643_618.approximate_steady_state(ffile_polar, inflow[0, :], structure_def["chord"],  
                                                          structure_def["stiffness"], x=approx_steady_x,  
                                                          y=approx_steady_y, torsion=approx_steady_tors, alpha=alpha)
        init_pos, f_init, inflow_angle = init_data
        NACA_643_618.aero[-1, :] = f_init  # if an HHT-alpha algorithm is used
        
        if alpha is not None:
            inflow_vel = np.linalg.norm(inflow[0, :])
            inflow = get_inflow(t, [(0, 0, inflow_vel, inflow_angle)], init_velocity=45)
            
        # init_pos = np.zeros(3)
        # init_pos[0] += 5*np.cos(init_pos[2])
        init_pos[0] += 0.5
        # init_pos[1] += 5*np.sin(init_pos[2])
        # init_pos[1] += 4
        # init_pos[2] -= 0.0001
        init_vel = np.zeros(3)  # initial conditions for the velocity in [x, y, rotation in rad/s]
        
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
        NACA_643_618.save(case_dir)  # save simulation results

    if forced:
        dir_free = join(root, "simulation", "free", aero_scheme, case_name)
        ff_free_motion = join(dir_free, "general.dat")
        
        n_parallel = 1
        root_cases = join(root, "simulation", sim_type, aero_scheme, case_name)
        df_gen = pd.read_csv(ff_free_motion)
        # motion_types = ["pos", "vel", "accel"]
        motion_types = ["pos", "vel"]
        motion_cols = ["_".join((motion, axes)) for motion, axes in product(motion_types, motion_from_axes)]

        # forced_amplitude = [0.2, 0.4, 0.6, 0.8, 1]
        # forced_amplitude_fac = [0.1, 0.2, 0.5, 0.8, 0.9, 1, 1.1, 1.2, 2, 4]
        forced_amplitude_fac = [1]
        forced_aoa = [9]
        periods = 4
        period_res = 500
        inflow_velocity = np.linalg.norm(df_gen[["inflow_x", "inflow_y"]].to_numpy()[-1, :])
        # forced_amplitude_fac = [0]
        # forced_aoa = [0]
        run_forced_parallel_from_free_case(root, file_polar, ff_free_motion, motion_cols, coordinate_system, 
                                           root_cases, n_parallel, aero_scheme, inflow_velocity, periods, period_res, 
                                           structure_def, forced_amplitude_fac, forced_aoa)

    if post_calc:
        if sim_type == "free":
            post_calc = PostCaluculations(dir_sim_res=join(root, "simulation", "free", aero_scheme, case_name),
                                          alpha_lift=alpha_lift, coordinate_system_structure=coordinate_system)
            post_calc.project_data()  # projects the simulation's x-y-rot_z data into different coordinate systems such as
            # edgewise-flapwise-rot_z or drag-lift_rot_z.
            post_calc.power()  # calculate and save the work done by the aerodynamic and structural damping forces
            post_calc.kinetic_energy()  # calculate the edgewise, flapwise, and rotational kinetic energy
            post_calc.potential_energy()  # calculate the edgewise, flapwise, and rotational potential energy
            post_calc.work_per_cycle()
            
        if "forced" in sim_type:
            n_parallel = 1
            root_cases = join(root, "simulation", sim_type, aero_scheme, case_name)
            post_calculations_parallel(root_cases, alpha_lift, n_parallel)

    if plot_results:
        trailing_every = 5
        file_profile = join(root, "profile.dat")  # define path to file containing the profile shape data
        if sim_type == "free":
            dir_sim = join(root, "simulation", "free", aero_scheme, case_name)  # define path to the root of the simulation results
        elif "forced" in sim_type:
            case_id = "0"
            dir_sim = join(root, "simulation", sim_type, aero_scheme, case_name, case_id)
        dir_plots = join(dir_sim, "plots")  # define path to the directory that the results are plotted into
        plotter = Plotter(file_profile, dir_sim, dir_plots, structure_def["coordinate_system"]) 
        plotter.force(trailing_every=trailing_every)  # plot various forces of the simulation
        plotter.energy(trailing_every=trailing_every)  # plot various energies and work done by forces of the simulation
        if "BL" in aero_scheme:
            plotter.Beddoes_Leishman(trailing_every=trailing_every)
    
    if plot_results_fill:
        trailing_every = 5
        file_profile = join(root, "profile.dat")  # define path to file containing the profile shape data
        if sim_type == "free":
            dir_sim = join(root, "simulation", "free", aero_scheme, case_name)  # define path to the root of the simulation results
            peak_distance = 400
        elif "forced" in sim_type:
            case_id = "0"
            dir_sim = join(root, "simulation", sim_type, aero_scheme, case_name, case_id)
            peak_distance = 90
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
        file_profile = join(root, "profile.dat")  # define path to file containing the profile shape data
        dir_sim = join(root, "simulation", sim_type, aero_scheme, case_name)   # define path to the root of the simulation results
        if sim_type == "forced":
            case_id = 0
            dir_sim = join(dir_sim, str(case_id))
        dir_plots = join(dir_sim, "plots")  # define path to the directory that the results are plotted into
        animator = Animator(file_profile, dir_sim, dir_plots) 
        # animate various forces of the simulation
        # animator.force(angle_lift=alpha_lift, arrow_scale_forces=0.01, arrow_scale_moment=.5,
        #                plot_qc_trailing_every=2, keep_qc_trailing=200, until=3, dt_per_frame=0.01)
        # plot various energies and work done by forces 
        # animator.energy(angle_lift=alpha_lift, arrow_scale_forces=0.5, arrow_scale_moment=.1, 
        #                plot_qc_trailing_every=2, keep_qc_trailing=200, until=30, dt_per_frame=0.1)  
        # of the simulation

    if plot_combined_forced:
        combined_forced(join(root, "simulation", sim_type, aero_scheme, case_name))


if __name__ == "__main__":
    main(**do)