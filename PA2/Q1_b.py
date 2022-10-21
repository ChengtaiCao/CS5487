"""
Q1_b.py
Implementation of 5 Regression Algorithms for The 5-th Order Polynomial
"""
import numpy as np

from utils import *
from cluster_algorithm import *

algorithms_dict = {
    0: "K-Means"
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
    cur_means, cur_z = kmeans(sample_x, K)
    figure_num = figure_count
    plot_title = f"Data {data_name} with {algorithms_dict[0]}"
    plot_kmeans(sample_x, cur_z, figure_num, plot_title)
