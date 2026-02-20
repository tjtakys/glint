import numpy as np

param_names = ["x_s", "y_s", "F_0", "inc_deg", "pa_deg", "r_scale", "v_c", "r_turn", "gamma_curve", "sigma_0", "r_sigma", "vsys_kms",
            #    "x_l", "y_l", "b", "q_l", "pa_l", "log_gamma", "pa_gamma"]
               "b", "q_l", "pa_l", "log_gamma", "pa_gamma"]


param_table = np.array([
    [0.015,  -0.1,   0.1],  # x_s
    [0.025,  -0.1,    0.2],  # y_s
    [0.3,    0.1,    2.0],  # F_0
    [40,     10,     70],   # inc_deg
    [60,     0,      180],  # pa_deg
    [0.1,    0.05,   0.5],  # r_scale
    [300,    200,    500],  # v_c
    [0.1,    0.01,   0.5],  # r_turn
    # [1.0,    0.5,    1.5],  # beta_curve
    [2.0,    1.5,   3.5],  # gamma_curve
    [60,     50,    180],  # sigma_0
    [0.1,    0.05,   0.8],  # r_sigma
    [0.0,    -50,    50],   # vsys_kms
    # [x0_l,   x0_l-dd_l, x0_l+dd_l],  # x_l
    # [y0_l,   y0_l-dd_l, y0_l+dd_l],  # y_l
    [1.30,   1.1,   1.6],   # b
    [0.95,   0.8,    1.0],   # q_l
    [0.0,    0.0,    np.pi], # pa_l
    [-2,   -5,    -1],   # log_gamma
    [0.0,   0.0,    np.pi/2],   # pa_gamma
]).astype(float)

# lb = param_table[:,1]
# ub = param_table[:,2]
# ndim = len(lb)
