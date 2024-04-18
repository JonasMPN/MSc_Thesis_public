from calculations import ThreeDOFsAirfoil
from post_calculations import PostCaluculations
import numpy as np
from plotting import Plotter
from os.path import join

do = {
    "simulate": True,  # run a simulation
    "post_calc": True,  # peform post calculations
    "plot_results": True  # plot results
}

root = "data/NACA_643_618"

if do["simulate"]:
    file_polar_data = join(root, "polars.dat")
    density = 1
    struct_def = {
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
    t = np.linspace(0, 40, 1300)
    NACA_643_618 = ThreeDOFsAirfoil(file_polar_data, t, **struct_def)

    inflow_speed = 1*np.ones((t.size, 1))
    t_angle_ramp = 3
    n_steps_angle_change = int(np.ceil(t_angle_ramp/(max(t)/t.size)))
    inflow_angle = 50*np.ones(t.size)
    inflow_angle[:n_steps_angle_change] = np.linspace(90, 50, n_steps_angle_change)
    inflow_angle = np.deg2rad(inflow_angle)
    inflow = inflow_speed*np.c_[np.sin(inflow_angle), np.cos(inflow_angle)]

    init_pos = np.zeros(3)
    init_vel = np.zeros(3)

    NACA_643_618.set_aero_calc()
    NACA_643_618.set_struct_calc()
    NACA_643_618.set_time_integration()
    NACA_643_618.simuluate(inflow, density, init_pos, init_vel)
    NACA_643_618.save(join(root, "simulation"))

if do["post_calc"]:
    post_calc = PostCaluculations(dir_sim_res=join(root, "simulation"), alpha_lift="alpha_qs")
    post_calc.project_data()
    post_calc.work()
    post_calc.kinetic_energy()
    post_calc.potential_energy()

if do["plot_results"]:
    file_profile = join(root, "profile.dat")
    dir_sim = join(root, "simulation")
    dir_plots = join(dir_sim, "plots")
    plotter = Plotter(file_profile, dir_sim, dir_plots)
    plotter.force()
    plotter.energy()
