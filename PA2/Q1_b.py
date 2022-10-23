"""
Q1_b.py
Implementation of 5 Regression Algorithms for The 5-th Order Polynomial
"""
import numpy as np

from utils import *
from cluster_algorithm import *

algorithms_dict = {
    0: "K-Means",
    1: "EM-GMM",
    2: "Mean-Shift"
}

if __name__ == "__main__":
    # set random seed
    np.random.seed(1)
    # set number of clusters
    K = 4
    figure_count = 1
    data_names=["A", "B", "C"]
    data_name = "A"
    sample_x, sample_y = get_data(data_name)
    # cur_means, cur_z = k_means(sample_x, K)
    figure_num = figure_count
    plot_title = f"Data {data_name} with {algorithms_dict[2]}"
    # plot1(sample_x, cur_z, figure_num, plot_title)
    # cur_means, cur_covs, cur_pis = em_gmm(sample_x, K)
    # cur_z = EStep(sample_x, cur_means, cur_covs, cur_pis)
    # plot1(sample_x, cur_z, figure_num, plot_title)
    labels, num_peak = mean_shift(sample_x)
    plot2(sample_x, labels, num_peak, figure_num, plot_title)
