from calculations import ThreeDOFsAirfoil
from post_calculations import PostCaluculations
import numpy as np
import json
from plotting import Plotter, Animator
from os.path import join

do = {
    "simulate": True,  # run a simulation
    "post_calc": True,  # peform post calculations
    "plot_results": True,  # plot results,
    "animate_results": False
}

# root = "data/NACA_643_618"  # set which airfoil polars to use. Simulatenously defines to root for the simulation data
root = "data/FFA_WA3_221"  # set which airfoil polars to use. Simulatenously defines to root for the simulation data
case_dir = "test"

if do["simulate"]:
    struct_def = {  # definition of structural parameters
        "chord": 1,
        "mass": 1,
        "mom_inertia": 1,
        "damping_edge": .575,
        "damping_flap": .2,
        "damping_tors": .5,
        "stiffness_edge": 1,
        "stiffness_flap": 1,
        "stiffness_tors": 1,
    }
    with open(join(root, "section_data.json"), "w") as f:
        json.dump(struct_def, f, indent=4)
        
    t = np.linspace(0, 40, 1300)  # set the time array for the simulation
    NACA_643_618 = ThreeDOFsAirfoil(root, t, verbose=True)

    # the following lines define the inflow at each time step
    inflow_speed = 1*np.ones((t.size, 1)) 
    t_angle_ramp = 3
    n_steps_angle_change = int(np.ceil(t_angle_ramp/(max(t)/t.size)))
    inflow_angle = 40*np.ones(t.size)  # in degrees
    inflow_angle[:n_steps_angle_change] = np.linspace(0, inflow_angle[0], n_steps_angle_change)
    inflow_angle = np.deg2rad(inflow_angle)
    inflow = inflow_speed*np.c_[np.cos(inflow_angle), np.sin(inflow_angle)]

    init_pos = np.zeros(3)  # initial conditions for the position in [x, y, rotation in rad]
    init_vel = np.zeros(3)  # initial conditions for the velocity in [x, y, rotation in rad/s]

    # set the calculation scheme for the aerodynamic forces
    # NACA_643_618.set_aero_calc()
    NACA_643_618.set_aero_calc("BL", A1=0.3, A2=0.7, b1=0.14, b2=0.53, pitching_around=0.25, alpha_at=0.75, 
                               chord=struct_def["chord"])
    # set the calculation scheme for the structural damping and stiffness forces
    NACA_643_618.set_struct_calc("linear", **struct_def)
    # set the time integration scheme
    # NACA_643_618.set_time_integration()
    NACA_643_618.set_time_integration("HHT-alpha-adapted", alpha=0.1, dt=t[1], **struct_def)
    NACA_643_618.simulate(inflow, init_pos, init_vel)  # perform simulation
    NACA_643_618.save(join(root, "simulation", case_dir))  # save simulation results

if do["post_calc"]:
    post_calc = PostCaluculations(dir_sim_res=join(root, "simulation", case_dir), alpha_lift="alpha_qs")
    post_calc.project_data()  # projects the simulation's x-y-rot_z data into different coordinate systems such as
    # edgewise-flapwise-rot_z or drag-lift_rot_z.
    post_calc.work()  # calculate and save the work done by the aerodynamic and structural damping forces
    post_calc.kinetic_energy()  # calculate the edgewise, flapwise, and rotational kinetic energy
    post_calc.potential_energy()  # calculate the edgewise, flapwise, and rotational potential energy

if do["plot_results"]:
    file_profile = join(root, "profile.dat")  # define path to file containing the profile shape data
    # dir_sim = join(root, "simulation", case_dir)  # define path to the root of the simulation results
    dir_sim = join(root, "simulation", case_dir)  # define path to the root of the simulation results
    dir_plots = join(dir_sim, "plots")  # define path to the directory that the results are plotted into
    plotter = Plotter(file_profile, dir_sim, dir_plots) 
    plotter.force()  # plot various forces of the simulation
    plotter.energy()  # plot various energies and work done by forces of the simulation
    if "BL" in case_dir:
        plotter.Beddoes_Leishman()
    plotter.Beddoes_Leishman()

if do["animate_results"]:
    file_profile = join(root, "profile.dat")  # define path to file containing the profile shape data
    dir_sim = join(root, "simulation", case_dir)  # define path to the root of the simulation results
    dir_plots = join(dir_sim, "plots")  # define path to the directory that the results are plotted into
    animator = Animator(file_profile, dir_sim, dir_plots) 
    animator.force(angle_lift="alpha_qs", arrow_scale_forces=0.5, arrow_scale_moment=.1, 
                   plot_qc_trailing_every=2, keep_qc_trailing=200)  # plot various forces of the simulation
    animator.energy(angle_lift="alpha_qs", arrow_scale_forces=0.5, arrow_scale_moment=.1, 
                   plot_qc_trailing_every=2, keep_qc_trailing=200)  # plot various energies and work done by forces of
    # the simulation