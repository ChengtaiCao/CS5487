"""
Q2_a.py
Three Clustering Algorithms for Image Segmentation
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
    sample_x = whiten(X.T)
    # hyper-parameters
    Ks = {2, 5, 10}
    hs = {0.03, 0.3, 3}
    K = 2
    bandwidth = 3
    figure_count = 1
    for j in range(3):
        figure_title = f"{algorithms_dict[j]} with K = {K} and h = {bandwidth}"
        if j == 0:      # K-Means
            _, cur_z = k_means(sample_x, K)
            plot3(img, L, cur_z, figure_count, figure_title)
        elif j == 1:    # EM-GMM
            cur_means, cur_covs, cur_pis = em_gmm(sample_x, K)
            cur_z = EStep(sample_x, cur_means, cur_covs, cur_pis)
            plot3(img, L, cur_z, figure_count, figure_title)
        else:
            bandwidth = 3
            labels, num_peak = mean_shift(sample_x, bandwidth)
            plot3(img, L, labels, figure_count, figure_title, True)
        figure_count += 1
    # different K
    for K in Ks:
        for j in range(2):
            figure_title= f"{algorithms_dict[j]} with K = {K}"
            if j == 0:      # K-Means
                _, cur_z = k_means(sample_x, K)
                plot3(img, L, cur_z, figure_count, figure_title)
            elif j == 1:    # EM-GMM
                cur_means, cur_covs, cur_pis = em_gmm(sample_x, K)
                cur_z = EStep(sample_x, cur_means, cur_covs, cur_pis)
                plot3(img, L, cur_z, figure_count, figure_title)
    # different h
    for bandwidth in hs:
        j = 2
        figure_title= f"{algorithms_dict[j]} with K = {K} and h = {bandwidth}"
        labels, num_peak = mean_shift(sample_x, bandwidth)
        plot3(img, L, labels, figure_count, figure_title, True)
        figure_count += 1
