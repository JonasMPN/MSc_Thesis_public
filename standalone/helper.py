import numpy as np

def get_zero_crossing_and_max_slope(x: np.ndarray, y: np.ndarray, gradient: str = "pos") -> tuple[float, float, float]:
    """Returns the root, the slope at root, and the maximum slope from the root to any other point.
    Could currently only work for functions with positive gradients at the root.
    :param x: _description_
    :type x: np.ndarray
    :param y: _description_
    :type y: np.ndarray
    :param gradient: _description_, defaults to "pos"
    :type gradient: str, optional
    :return: _description_
    :rtype: tuple[float, float]
    """
    idx_greater_0 = np.argmax(y > 0) if gradient == "pos" else np.argmax(y < 0)
    dy_zero = y[idx_greater_0] - y[idx_greater_0 - 1]
    dx_zero = x[idx_greater_0] - x[idx_greater_0 - 1]
    x_0 = x[idx_greater_0] - y[idx_greater_0] * dx_zero / dy_zero
    max_dy_dx = 0
    for i in range(x.size - 1):
        max_dy_dx = max(max_dy_dx, abs(y[i] / (x_0 - x[i])))
    return x_0, dy_zero / dx_zero, max_dy_dx
