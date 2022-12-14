"""
Q1_c.py
Implementation of Error Versus Training Size
1. choose N_train%
2. plot figure for [0.1, 0.2, 0.3,..., 1]
"""
import pdb
from utils import *
from regression_algorithm import *


if __name__ == "__main__":
    path_folder = "./data/"
    K = 5
    repeat_num = 100

    #####################################################################
    # 1. choose 0.5 to plot figure
    LS_mse_errors = []
    RLS_mse_errors = []
    LASSO_mse_errors = []
    RR_mse_errors = []
    BR_mse_errors = []
    for i in range(10):
        sample_x, sample_y, poly_x, poly_y = get_polydata(path_folder, ratio=0.5)
        # sample_x: (N_train * ratio, 1)
        # sample_y: (N_train * ratio, 1)
        # poly_x: (N_test, 1)
        # poly_y: (N_test, 1)
        # N_sample = N_train * ratio
        # N_poly = N_test                       
        sample_x_tran = poly_feature_trans(sample_x, K)          # (feature_dimension, N_train * ratio)
        poly_x_tran = poly_feature_trans(poly_x, K)              # (feature_dimension, N_test)
        # LS_estimation
        LS_theta = LS_estimation(sample_x_tran, sample_y)        # (feature_dimension, 1)
        LS_pre = prediction(poly_x_tran, LS_theta)               # (N_test, 1)
        LS_mse_error = get_mse_error(LS_pre, poly_y)
        LS_mse_errors.append(LS_mse_error)

        # RLS_estimation
        RLS_theta = RLS_estimation(sample_x_tran, sample_y)      # (feature_dimension, 1)
        RLS_pre = prediction(poly_x_tran, RLS_theta)             # (N_test, 1)
        RLS_mse_error = get_mse_error(RLS_pre, poly_y)
        RLS_mse_errors.append(RLS_mse_error)

        # LASSO_estimation
        LASSO_theta = LASSO_estimation(sample_x_tran, sample_y)  # (feature_dimension, 1)
        LASSO_pre = prediction(poly_x_tran, LASSO_theta)         # (N_test, 1)
        LASSO_mse_error = get_mse_error(LASSO_pre, poly_y)
        LASSO_mse_errors.append(LASSO_mse_error)

        # RR_estimation
        RR_theta = RR_estimation(sample_x_tran, sample_y)        # (feature_dimension, 1)
        RR_pre = prediction(poly_x_tran, RR_theta)               # (N_test, 1)
        RR_mse_error = get_mse_error(RR_pre, poly_y)
        RR_mse_errors.append(RR_mse_error)

        # BR_estimation
        theta_mean, theta_cov = BR_estimation(sample_x_tran, sample_y)        
        estimate_mean, estimate_variance = BR_prediction(poly_x_tran, theta_mean, theta_cov)  # (N_test, 1)
        BR_mse_error = get_mse_error(RLS_pre, estimate_mean)
        BR_mse_errors.append(BR_mse_error)

        
    print(f"LS_mse_error: {np.mean(LS_mse_error)}")
    print(f"RLS_mse_error: {np.mean(RLS_mse_error)}")
    print(f"LASSO_mse_error: {np.mean(LASSO_mse_error)}")
    print(f"RR_mse_error: {np.mean(RR_mse_error)}")
    print(f"BR_mse_error: {np.mean(BR_mse_error)}")
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

    #####################################################################
    # 2. plot figure for ratio [0.1, 0.2, 0.3, ..., 0.9, 1]
    LS_errors = []
    RLS_errors = []
    LASSO_errors = []
    RR_errors = []
    BR_errors = []
    for i in range(10):
        ratio = 0.1 + i * 0.1
        LS_errors.append([])
        RLS_errors.append([])
        LASSO_errors.append([])
        RR_errors.append([])
        BR_errors.append([])
        for _ in range(repeat_num):
            sample_x, sample_y, poly_x, poly_y = get_polydata(path_folder, ratio=0.5)
            # sample_x: (N_train * ratio, 1)
            # sample_y: (N_train * ratio, 1)
            # poly_x: (N_test, 1)
            # poly_y: (N_test, 1)
            # N_sample = N_train * ratio
            # N_poly = N_test            
            sample_x_tran = poly_feature_trans(sample_x, K)          # (feature_dimension, N_train * ratio)
            poly_x_tran = poly_feature_trans(poly_x, K)              # (feature_dimension, N_test)
            # LS_estimation
            LS_theta = LS_estimation(sample_x_tran, sample_y)        # (feature_dimension, 1)
            LS_pre = prediction(poly_x_tran, LS_theta)               # (N_test, 1)
            LS_mse_error = get_mse_error(LS_pre, poly_y)
            LS_errors[-1].append(LS_mse_error)

            # RLS_estimation
            RLS_theta = RLS_estimation(sample_x_tran, sample_y)      # (feature_dimension, 1)
            RLS_pre = prediction(poly_x_tran, RLS_theta)             # (N_test, 1)
            RLS_mse_error = get_mse_error(RLS_pre, poly_y)
            RLS_errors[-1].append(RLS_mse_error)

            # LASSO_estimation
            LASSO_theta = LASSO_estimation(sample_x_tran, sample_y)  # (feature_dimension, 1)
            LASSO_pre = prediction(poly_x_tran, LASSO_theta)         # (N_test, 1)
            LASSO_mse_error = get_mse_error(LASSO_pre, poly_y)
            LASSO_errors[-1].append(LASSO_mse_error)

            # RR_estimation
            RR_theta = RR_estimation(sample_x_tran, sample_y)        # (feature_dimension, 1)
            RR_pre = prediction(poly_x_tran, RR_theta)               # (N_test, 1)
            PR_mse_error = get_mse_error(RR_pre, poly_y)
            RR_errors[-1].append(PR_mse_error)

            # BR_estimation
            theta_mean, theta_cov = BR_estimation(sample_x_tran, sample_y)        
            estimate_mean, estimate_variance = BR_prediction(poly_x_tran, theta_mean, theta_cov)  # (N_test, 1)
            BR_mse_error = get_mse_error(estimate_mean, poly_y)
            BR_errors[-1].append(BR_mse_error)
    
    LS_errors = np.array(LS_errors)
    RLS_errors = np.array(RLS_errors)
    LASSO_errors = np.array(LASSO_errors)
    RR_errors = np.array(RR_errors)
    BR_errors = np.array(BR_errors)

    plot_figure_for_training_size(LS_errors, RLS_errors, LASSO_errors, RR_errors, BR_errors)
