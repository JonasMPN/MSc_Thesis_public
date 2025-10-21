import numpy as np


def quasi_steady_flow_angle(
    velocity: np.ndarray,
    position: np.ndarray,
    flow: np.ndarray,
    chord: float,
    pitching_around: float,  # as factor of chord!
    alpha_at: float,  #  as factor of chord!
):
    pitching_speed = velocity[2] * chord * (alpha_at - pitching_around)
    v_pitching_x = np.sin(-position[2]) * pitching_speed  # x velocity of the point
    v_pitching_y = np.cos(position[2]) * pitching_speed  # y velocity of the point

    v_x = flow[0] - velocity[0] - v_pitching_x
    v_y = flow[1] - velocity[1] - v_pitching_y
    return np.arctan2(v_y, v_x), v_x, v_y
