import numpy as np

param_names = ["x_s", "y_s", "F_0", "ellip", "pa_deg", "r_eff", "n",
               "b", "q_l", "pa_l", "log_gamma", "pa_gamma"]


param_table = np.array([
    [0.0175,  -0.2,   0.2],  # x_s
    [0.0275,  -0.2,   0.2],  # y_s
    [0.01,  0.002, 0.05],  # F_0
    [0.1,    0.1,    0.8],  # ellip
    [100,     0,      180],  # pa_deg
    [0.13,    0.1,   0.25],  # r_eff
    [1.7,    0.5,   4.5],  # n
    [1.30,   1.1,   1.5],   # b
    [0.95,   0.8,    1.0],   # q_l
    [0.01,    0.0,    np.pi], # pa_l
    [-1.2,   -5,    -1],   # log_gamma
    [0.9,   0.0,    np.pi/2],   # pa_gamma
]).astype(float)
