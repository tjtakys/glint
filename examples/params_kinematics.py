import numpy as np

# param_names = ["x_s", "y_s", "F_0", "inc_deg", "pa_deg", "r_scale", "v_c", "r_turn", "gamma_curve", "sigma_0", "r_sigma", "vsys_kms",
# param_names = ["x_s", "y_s", "F_0", "inc_deg", "pa_deg", "r_scale", "v_c", "r_turn", "beta", "gamma_curve", "sigma_0", "r_sigma", "vsys_kms",
# param_names = ["F_0", "pa_deg", "r_scale", "v_c", "r_turn", "beta", "gamma_curve", "sigma_0", "r_sigma", "vsys_kms",
# param_names = ["F_0", "pa_deg", "r_scale", "v_c", "r_turn", "beta", "gamma_curve", "sigma_0", "r_sigma",
param_names = ["F_0", "pa_deg", "r_scale", "r_turn", "beta", "gamma_curve", "sigma_0", "r_sigma",
               "b", "log_gamma", "pa_gamma"]
            #    "b", "q_l", "pa_l", "log_gamma", "pa_gamma"]

# initial guess, lower bound, upper bound
param_table = np.array([
    # [0.015,  -0.1,   0.2],  # x_s
    # [0.025,  -0.1,    0.2],  # y_s
    # [0.034,  0.034-1e-5,   0.034+1e-5],  # x_s fix to the mom-0 best-fit
    # [0.029,  0.029-1e-5,   0.029+1e-5],  # y_s
    [5,    0.1,    20],  # F_0
    # [30,     10,     40],   # inc_deg
    [60,     0,      180],  # pa_deg
    [0.1,    0.05,   0.5],  # r_scale
    # [300,    200,    450],  # v_c
    [0.1,    0.005,   0.5],  # r_turn
    [0.8,    0.5, 1],  # beta_curve
    [3,    1.0,   4.5],  # gamma_curve
    [150,     60,    200],  # sigma_0
    [0.1,    0.05,   1.5],  # r_sigma
    # [0.0,    -50,    50],   # vsys_kms
    [1.30,   1.1,   1.6],   # b
    # [0.95,   0.8,    1.0],   # q_l
    # [0.35,   0.0,    np.pi], # pa_l
    [-1.2,   -5,    -1],   # log_gamma
    [1.0,   0.0,    np.pi/2],   # pa_gamma
]).astype(float)
