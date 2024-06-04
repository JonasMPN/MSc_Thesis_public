from calculations import ThreeDOFsAirfoil
from post_calculations import PostCaluculations
from calculation_utils import get_inflow
from calculation_wrappers import run_forced_parallel_from_free_case, post_calculations_parallel
import numpy as np
import json
from plotting import Plotter, Animator, combined_forced
from os.path import join
from os import listdir
from helper_functions import Helper
from itertools import product
helper = Helper()

do = {
    "simulate": False,  # run a simulation
    "forced": False,
    "post_calc": False,  # peform post calculations
    "plot_results": False,  # plot results
    "animate_results": True,
    "plot_combined_forced": False
}

# root = "data/NACA_643_618"  # set which airfoil polars to use. Simulatenously defines to root for the simulation data
root = "data/FFA_WA3_221"  # set which airfoil polars to use. Simulatenously defines to root for the simulation data

sim_type = "free"
# sim_type = "forced"

aero_scheme = "qs"

alpha_lift = "alpha_qs"

# case_name = "test_BL_openFAST"
# case_name = "test_qs"
case_name = "aoa_20"

# for forced
motion_from_axes = ["edge", "flap", "tors"]  # adapt 'cs' for changes here
coordinate_system = "ef"
# motion_from_axes = ["x", "y", "tors"]  # adapt 'cs' for changes here
# coordinate_system = "xy"

#todo update to values from paper
struct_def = {  # definition of structural parameters
    "chord": 1,
    "mass": 1,
    "mom_inertia": 1,
    "damping_edge": .568,
    "damping_flap": .2,
    "damping_tors": .5,
    "stiffness_edge": 1,
    "stiffness_flap": 1,
    "stiffness_tors": 1,
}

def main(simulate, forced, post_calc, plot_results, animate_results, plot_combined_forced):
    if simulate:
        case_dir = helper.create_dir(join(root, "simulation", "free", aero_scheme, case_name))[0]
        with open(join(case_dir, "section_data.json"), "w") as f:
            json.dump(struct_def, f, indent=4)
        dt = 0.01  # if dt<1e-5, the dt rounding in compress_oscillation() in calculation_utils.py needs to be adapted
        t_end = 100        
        t = dt*np.arange(int(t_end/dt))  # set the time array for the simulation
        NACA_643_618 = ThreeDOFsAirfoil(root, t, verbose=True)

        # get inflow at each time step. The below line creates inflow that has magnitude 1 and changes from 0 deg to 40
        # deg in the first 3s.
        inflow = get_inflow(t, [(0, 3, 1, 20)], init_velocity=1)
        init_pos = np.zeros(3)  # initial conditions for the position in [x, y, rotation in rad]
        init_vel = np.zeros(3)  # initial conditions for the velocity in [x, y, rotation in rad/s]

        # set the calculation scheme for the aerodynamic forces
        if aero_scheme == "qs":
            NACA_643_618.set_aero_calc()
        elif aero_scheme == "BL_chinese":
            NACA_643_618.set_aero_calc("BL_chinese", A1=0.3, A2=0.7, b1=0.14, b2=0.53, pitching_around=0.25, 
                                       alpha_at=0.75, chord=struct_def["chord"])
        elif aero_scheme == "BL_openFAST_Cl_disc":
            NACA_643_618.set_aero_calc("BL_openFAST_Cl_disc", A1=0.165, A2=0.335, b1=0.0445, b2=0.3, 
                                       pitching_around=0.25, alpha_at=0.75, chord=struct_def["chord"])
        else:
            raise NotImplementedError(f"'aero_scheme'={aero_scheme} not implemented.")
        
        # set the calculation scheme for the structural damping and stiffness forces
        NACA_643_618.set_struct_calc("linear", **struct_def)
        # NACA_643_618.set_time_integration()
        NACA_643_618.set_time_integration("HHT-alpha-adapted", alpha=0.1, dt=t[1], **struct_def)
        NACA_643_618.simulate(inflow, init_pos, init_vel)  # perform simulation
        NACA_643_618.save(case_dir)  # save simulation results

    if forced:
        n_parallel = 8
        root_cases = join(root, "simulation", sim_type, aero_scheme, coordinate_system, case_name)
        ff_free_motion = join(root, "simulation", "free", "qs", "base", "general.dat")
        # motion_types = ["pos", "vel", "accel"]
        motion_types = ["pos", "vel"]
        motion_cols = ["_".join((motion, axes)) for motion, axes in product(motion_types, motion_from_axes)]

        # forced_amplitude = [0.2, 0.4, 0.6, 0.8, 1]
        forced_amplitude_fac = [0.1, 0.2, 0.5, 0.8, 0.9, 1, 1.1, 1.2, 2, 4]
        forced_aoa = [0, 10, 20, 30, 40]
        periods = 4
        period_res = 100
        # forced_amplitude_fac = [0]
        # forced_aoa = [0]
        run_forced_parallel_from_free_case(root, ff_free_motion, motion_cols, coordinate_system, root_cases, 
                                           n_parallel, aero_scheme, periods, period_res, struct_def, 
                                           forced_amplitude_fac, forced_aoa)

    if post_calc:
        if sim_type == "free":
            post_calc = PostCaluculations(dir_sim_res=join(root, "simulation", "free", aero_scheme, case_name),
                                          alpha_lift=alpha_lift)
            post_calc.project_data()  # projects the simulation's x-y-rot_z data into different coordinate systems such as
            # edgewise-flapwise-rot_z or drag-lift_rot_z.
            post_calc.power()  # calculate and save the work done by the aerodynamic and structural damping forces
            post_calc.kinetic_energy()  # calculate the edgewise, flapwise, and rotational kinetic energy
            post_calc.potential_energy()  # calculate the edgewise, flapwise, and rotational potential energy
            post_calc.work_per_cycle()
            
        if "forced" in sim_type:
            n_parallel = 8
            root_cases = join(root, "simulation", sim_type, aero_scheme, coordinate_system, case_name)
            post_calculations_parallel(root_cases, alpha_lift, n_parallel)

    if plot_results:
        file_profile = join(root, "profile.dat")  # define path to file containing the profile shape data
        if sim_type == "free":
            dir_sim = join(root, "simulation", "free", aero_scheme, case_name)  # define path to the root of the simulation results
        elif "forced" in sim_type:
            case_id = "44"
            dir_sim = join(root, "simulation", sim_type, aero_scheme, coordinate_system, case_name, case_id)
        dir_plots = join(dir_sim, "plots")  # define path to the directory that the results are plotted into
        plotter = Plotter(file_profile, dir_sim, dir_plots) 
        plotter.force()  # plot various forces of the simulation
        plotter.energy()  # plot various energies and work done by forces of the simulation
        if "BL" in case_name:
            plotter.Beddoes_Leishman()

    if animate_results:
        file_profile = join(root, "profile.dat")  # define path to file containing the profile shape data
        # dir_sim = join(root, "simulation", sim_type, case_name)  # define path to the root of the simulation results
        # dir_sim = "data/FFA_WA3_221/simulation/forced/qs/ef/e/"
        dir_sim = "test"
        dir_plots = join(dir_sim, "plots")  # define path to the directory that the results are plotted into
        animator = Animator(file_profile, dir_sim, dir_plots) 
        alpha_lift = "alpha_steady"
        animator.force(angle_lift=alpha_lift, arrow_scale_forces=0.5, arrow_scale_moment=.1,
                       plot_qc_trailing_every=2, keep_qc_trailing=200, until=30, dt_per_frame=0.1)  # plot various forces of the simulation
        # animator.energy(angle_lift="alpha_qs", arrow_scale_forces=0.5, arrow_scale_moment=.1, 
        #                plot_qc_trailing_every=2, keep_qc_trailing=200)  # plot various energies and work done by forces of
        # the simulation

    if plot_combined_forced:
        combined_forced(join(root, "simulation", sim_type, aero_scheme, coordinate_system, case_name))


if __name__ == "__main__":
    main(**do)