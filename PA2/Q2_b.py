"""
Q2_b.py
Feature Scaling
"""
import numpy as np
from PIL import Image
from scipy.cluster.vq import whiten

from utils import *
from cluster_algorithm import *
from pa2 import getfeatures


algorithms_dict = {
    0: "K-Means",
    1: "EM-GMM",
    2: "Mean-Shift"
}


if __name__ == "__main__":
    # set random seed
    np.random.seed(1)
    # get image paths
    paths = get_images()
    # choose the first 5 images
    path = paths[0]
    img = Image.open(path)
    X, L = getfeatures(img, 7)
    # hyper-parameters
    K = 2
    bandwidth = 0.3

    # K-Means without feature scaling
    figure_count, j = 1, 0
    sample_x = whiten(X.T)
    figure_title = f"{algorithms_dict[j]} without feature scaling"
    _, cur_z = k_means(sample_x, K)
    plot3(img, L, cur_z, figure_count, figure_title)

    # K-Means with feature scaling
    figure_count, j = 2, 0
    sample_x = whiten(X.T)
    lambda_value = 0.5
    sample_x[:, 2] *= lambda_value
    sample_x[:, 2] *= lambda_value
    figure_title= f"{algorithms_dict[j]} with feature scaling"
    _, cur_z = k_means(sample_x, K)
    plot3(img, L, cur_z, figure_count, figure_title)

    # Mean-shift without feature scaling
    figure_count, j = 3, 2 
    figure_title = f"{algorithms_dict[j]} without feature scaling"
    sample_x = whiten(X.T)
    labels, num_peak = mean_shift(sample_x, bandwidth)
    plot3(img, L, labels, figure_count, figure_title, True)

    # Means-shift with feature scaling
    figure_count, j = 4, 2 
    figure_title = f"{algorithms_dict[j]} with feature scaling"
    sample_x = whiten(X.T)
    h_c = 0.5
    h_p = 0.5
    sample_x[:, 0] *= h_c
    sample_x[:, 1] *= h_c
    sample_x[:, 2] *= h_p
    sample_x[:, 3] *= h_p
    labels, num_peak = mean_shift(sample_x, bandwidth)
    plot3(img, L, labels, figure_count, figure_title, True)