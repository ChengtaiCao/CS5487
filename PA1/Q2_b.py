"""
Q1_b.py
Implementation of 5 Regression Algorithms for The 5-th Order Polynomial
"""
from utils import *
from regression_algorithm import *


if __name__ == "__main__":
    path_folder = "./data/"
    sample_x, sample_y, poly_x, poly_y = get_countdata(path_folder)
    # sample_x: (feature_dimension, N_train)
    # sample_y: (N_train, 1)
    # poly_x: (feature_dimension, N_test)
    # poly_y: (N_test, 1)
    # N_sample = N_train
    # N_poly = N_test          
    sample_x_square = get_square_matrix(sample_x)
    sample_x = np.concatenate((sample_x, sample_x_square), axis=0)
    poly_x_square = get_square_matrix(poly_x)
    poly_x = np.concatenate((poly_x, poly_x_square), axis=0)

    # LS_estimation
    print("*" * 20)
    LS_theta = LS_estimation(sample_x, sample_y)            # (feature_dimension, 1)
    LS_pre = prediction(poly_x, LS_theta)                   # (N_test, 1)
    LS_mse_error = get_mse_error(LS_pre, poly_y)
    print(f"LS_mse_error: {LS_mse_error}")
    LS_mae_error= get_mae_error(LS_pre, poly_y)
    print(f"LS_mae_error: {LS_mae_error}")
    LS_accuracy= get_accuracy(LS_pre, poly_y)
    print(f"LS_accuracy: {LS_accuracy}")

    # RLS_estimation
    print("*" * 20)
    RLS_theta = RLS_estimation(sample_x, sample_y)          # (feature_dimension, 1)
    RLS_pre = prediction(poly_x, RLS_theta)                 # (N_test, 1)
    RLS_mse_error = get_mse_error(RLS_pre, poly_y)
    print(f"RLS_mse_error: {RLS_mse_error}")
    RLS_mae_error= get_mae_error(RLS_pre, poly_y)
    print(f"RLS_mae_error: {RLS_mae_error}")
    RLS_accuracy= get_accuracy(RLS_pre, poly_y)
    print(f"RLS_accuracy: {RLS_accuracy}")

    # LASSO_estimation
    print("*" * 20)
    LASSO_theta = LASSO_estimation(sample_x, sample_y)      # (feature_dimension, 1)
    LASSO_pre = prediction(poly_x, LASSO_theta)             # (N_test, 1)
    LASSO_mse_error = get_mse_error(LASSO_pre, poly_y)
    print(f"LASSO_mse_error: {LASSO_mse_error}")
    LASSO_mae_error= get_mae_error(LASSO_pre, poly_y)
    print(f"LASSO_mae_error: {LASSO_mae_error}")
    LASSO_accuracy= get_accuracy(LASSO_pre, poly_y)
    print(f"LASSO_accuracy: {LASSO_accuracy}")

    # RR_estimation
    print("*" * 20)
    RR_theta = RR_estimation(sample_x, sample_y)            # (feature_dimension, 1)
    RR_pre = prediction(poly_x, RR_theta)                   # (N_test, 1)
    RR_mse_error = get_mse_error(RR_pre, poly_y)
    print(f"RR_mse_error: {RR_mse_error}")
    RR_mae_error= get_mae_error(RR_pre, poly_y)
    print(f"LS_mae_error: {RR_mae_error}")
    RR_accuracy= get_accuracy(RR_pre, poly_y)
    print(f"RR_accuracy: {RR_accuracy}")

    # BR_estimation
    print("*" * 20)
    theta_mean, theta_cov = BR_estimation(sample_x, sample_y)        
    estimate_mean, estimate_variance = BR_prediction(poly_x, theta_mean, theta_cov)  # (N_test, 1)
    BR_mse_error = get_mse_error(estimate_mean, poly_y)
    print(f"BR_mse_error: {BR_mse_error}")
    BR_mae_error= get_mae_error(estimate_mean, poly_y)
    print(f"BR_mae_error: {BR_mae_error}")
    BR_accuracy= get_accuracy(estimate_mean, poly_y)
    print(f"BR_accuracy: {BR_accuracy}")

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
    sample_x = sample_x.transpose()[:,0]
    poly_x = poly_x.transpose()[:,0]
    for i in range(1, 5):
        plot_figure_2(i, title_dict[i], poly_x, poly_y, esitimation_y_dict[i])
    # plot figure for BR
    BR_plot_figure_2(5, "BR_Estimation", poly_x, poly_y, estimate_mean, estimate_variance)
