import numpy as np

param_names = ["x_s", "y_s", "F_0", "ellip", "pa_deg", "r_eff", "n",
               "b", "q_l", "pa_l", "log_gamma", "pa_gamma"]


# initial guess, lower bound, upper bound
param_table = np.array([
    [0.015,  -0.2,   0.2],  # x_s
    [0.025,  -0.2,   0.2],  # y_s
    [15,    1,    20],  # F_0
    [0.2,    0.1,    0.8],  # ellip
    [100,     0,      180],  # pa_deg
    [0.2,    0.1,   0.25],  # r_eff
    [1.0,    0.5,   4.5],  # n
    [1.50,   1.1,   1.8],   # b
    [0.95,   0.8,    1.0],   # q_l
    [0.5,    0.0,    np.pi], # pa_l
    [-2,   -5,    -1],   # log_gamma
    [0.0,   0.0,    np.pi/2],   # pa_gamma
]).astype(float)
