"""
Q1_e.py
Implementation Higher-order Polynomial (10)
"""
from utils import *
from regression_algorithm import *


if __name__ == "__main__":
    path_folder = "./data/"
    sample_x, sample_y, poly_x, poly_y = get_polydata(path_folder)
    # sample_x: (N_train * ratio, 1)
    # sample_y: (N_train * ratio, 1)
    # poly_x: (N_test, 1)
    # poly_y: (N_test, 1)
    # N_sample = N_train * ratio
    # N_poly = N_test                       

    K = 15
    sample_x_tran = poly_feature_trans(sample_x, K)          # (feature_dimension, N_train * ratio)
    poly_x_tran = poly_feature_trans(poly_x, K)              # (feature_dimension, N_test)
    # LS_estimation
    LS_theta = LS_estimation(sample_x_tran, sample_y)        # (feature_dimension, 1)
    LS_parameter_sum = np.sum(LS_theta)
    LS_pre = prediction(poly_x_tran, LS_theta)               # (N_test, 1)
    LS_mse_error = get_mse_error(LS_pre, poly_y)
    print("*" * 20)
    print(f"LS_mse_error: {LS_mse_error}")
    print(f"LS_parameter_sum: {LS_parameter_sum}")

    # RLS_estimation
    RLS_theta = RLS_estimation(sample_x_tran, sample_y)      # (feature_dimension, 1)
    RLS_parameter_sum = np.sum(RLS_theta)
    RLS_pre = prediction(poly_x_tran, RLS_theta)             # (N_test, 1)
    RLS_mse_error = get_mse_error(RLS_pre, poly_y)
    print("*" * 20)
    print(f"RLS_mse_error: {RLS_mse_error}")
    print(f"RLS_parameter_sum: {RLS_parameter_sum}")

    # LASSO_estimation
    LASSO_theta = LASSO_estimation(sample_x_tran, sample_y)  # (feature_dimension, 1)
    LASSO_parameter_sum = np.sum(LASSO_theta)
    LASSO_pre = prediction(poly_x_tran, LASSO_theta)         # (N_test, 1)
    LASSO_mse_error = get_mse_error(LASSO_pre, poly_y)
    print("*" * 20)
    print(f"LASSO_mse_error: {LASSO_mse_error}")
    print(f"LASSO_parameter_sum: {LASSO_parameter_sum}")

    # RR_estimation
    RR_theta = RR_estimation(sample_x_tran, sample_y)        # (feature_dimension, 1)
    RR_parameter_sum = np.sum(RR_theta)
    RR_pre = prediction(poly_x_tran, RR_theta)               # (N_test, 1)
    RR_mse_error = get_mse_error(RR_pre, poly_y)
    print("*" * 20)
    print(f"RR_mse_error: {RR_mse_error}")
    print(f"RR_parameter_sum: {RR_parameter_sum}")

    # BR_estimation
    theta_mean, theta_cov = BR_estimation(sample_x_tran, sample_y)
    BR_parameter_sum = np.sum(theta_mean)
    estimate_mean, estimate_variance = BR_prediction(poly_x_tran, theta_mean, theta_cov)  # (N_test, 1)
    BR_mse_error = get_mse_error(estimate_mean, poly_y)
    print("*" * 20)
    print(f"BR_mse_error: {BR_mse_error}")
    print(f"BR_parameter_sum: {BR_parameter_sum}")

    # plot figure
    title_dict = {
        1: "LS_Estimation",
        2: "RLS_Estimation",
        3: "LASSO_Estimation",
        4: "RR_Estimation"
    }

    esitimation_y_dict = {
        1: LS_pre,
        2: RLS_pre,
        3: LASSO_pre,
        4: RR_pre
    }

    # plot figure for all except BR
    for i in range(1, 5):
        plot_figure(i, title_dict[i], sample_x, sample_y, poly_x, esitimation_y_dict[i])
    # plot figure for BR
    BR_plot_figure(5, "BR_Estimation", sample_x, sample_y, poly_x, estimate_mean, estimate_variance)
