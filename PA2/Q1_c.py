"""
Q1_c.py
Effect of Bandwidth
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
    data_names = ["A", "B", "C"]
    for data_name in data_names:
        if data_name == "B":
            bandwidths = [0.03, 3, 300]
        else: 
            bandwidths = [0.05, 5, 500]
        for bandwidth in bandwidths:
            sample_x, sample_y = get_data(data_name)
            plot_title = f"Data {data_name} with bandwidth {bandwidth}"
            # Mean-Shift
            labels, num_peak = mean_shift(sample_x, bandwidth)
            print(f"Data: {data_name}, Bandwidth: {bandwidth}, numbef or peaks: {num_peak}")
            plot2(sample_x, labels, num_peak, figure_count, plot_title)
            figure_count += 1
