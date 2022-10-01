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
    sample_x = poly_feature_trans(sample_x, K)              # (K + 1, 50)
    poly_x = poly_feature_trans(poly_x, K)                  # (K + 1, 100)
    # LS_estimation
    LS_theta = LS_estimation(sample_x, sample_y)            # (K + 1, 1)
    LS_pre_res = prediction(poly_x, LS_theta)               # (100, 1)
    # RLS_estimation
    RLS_theta = RLS_estimation(sample_x, sample_y)          # (K + 1, 1)
    RLS_pre_res = prediction(poly_x, RLS_theta)             # (100, 1)
    # LASSO
    LASSO_theta = LASSO_estimation(sample_x, sample_y)      # (K + 1, 1)
    LASSO_pre_res = prediction(poly_x, LASSO_theta)         # (100, 1)
    print(LASSO_theta)