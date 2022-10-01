"""
Implement the above 5 regression algorithms for the K-th order polynomial
"""
import matplotlib.pyplot as plt

from utils import *
from regression_algorithm import *


if __name__ == "__main__":
    K = 5
    path_folder = "./data/"
    sample_x, sample_y, poly_x, poly_y = get_polydata(path_folder)
    # sample_y: (50, 1)
    # poly_y: (100, 1)
    # N_sample = 50
    # N_poly = 100                       
    sample_x = poly_feature_trans(sample_x, K)          # (K + 1, 50)
    poly_x = poly_feature_trans(poly_x, K)              # (K + 1, 100)


    ################################################################## 
    # 1_B 
    # LS_estimation
    LS_theta = LS_estimation(sample_x, sample_y)        # (K + 1, 1)
    LS_pre = prediction(poly_x, LS_theta)               # (100, 1)
    LS_mse_error = get_error(LS_pre, poly_y)
    print(f"LS_mse_error: {LS_mse_error}")

    # RLS_estimation
    RLS_theta = RLS_estimation(sample_x, sample_y)      # (K + 1, 1)
    RLS_pre = prediction(poly_x, RLS_theta)             # (100, 1)
    RLS_mse_error = get_error(RLS_pre, poly_y)
    print(f"RLS_mse_error: {RLS_mse_error}")

    # LASSO_estimation
    LASSO_theta = LASSO_estimation(sample_x, sample_y)  # (K + 1, 1)
    LASSO_pre_res = prediction(poly_x, LASSO_theta)     # (100, 1)
    LASSO_mse_error = get_error(LASSO_pre_res, poly_y)
    print(f"LASSO_mse_error: {LASSO_mse_error}")

    # RR_estimation
    RR_theta = RR_estimation(sample_x, sample_y)        # (K + 1, 1)
    RR_pre = prediction(poly_x, RR_theta)               # (100, 1)
    PR_mse_error = get_error(RR_pre, poly_y)
    print(f"PR_mse_error: {PR_mse_error}")

    # BR_estimation
    theta_mean, theta_cov = BR_estimation(sample_x, sample_y)        
    estimate_mean, estimate_variance = BR_prediction(poly_x, theta_mean, theta_cov)  # (100, 1)
    BR_mse_error = get_error(RLS_pre, estimate_mean)
    print(f"BR_mse_error: {BR_mse_error}")