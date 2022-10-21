"""
Implement the Above 3 Clustering Algorithms
feature_dimension = d
"""
import numbers
import numpy as np


# K-Means
def euclidean_distance(point1, point2):
    """ 
    Calculating Euclidean Distance betwean point1 and point2
    point1: point1: d-dim vector
    point2: point2: d-dim vector
    return:
        euc_dis: scalar
    """
    euc_dis = np.sqrt(np.sum(np.square(point1 - point2)))
    return euc_dis


def init_means(sample_x, K):
    """ 
    Getting Initialized Means for K-Means
    sample_x: (num_sample, d)
    K: number of cluster
    return:
        means: K initialized d-dim vector for K clusters: (K, d) 
    """
    num_sample = sample_x.shape[0]
    mean_index = np.random.randint(num_sample, size=K)
    means = sample_x[mean_index]
    return means


def update_kmean_z(sample_x, cur_means, K):
    """
    Updata z in K-Means
    sample_x: (num_sample, d)
    cur_means: current z: (K, d)
    K: number of clusters
    return:
        update_z: updated z: (num_sample, K)
    """
    num_sample = sample_x.shape[0]
    update_z = np.zeros((num_sample, K))
    for i in range(num_sample):
        euc_dises = []
        for k in range(K):
            point1 = sample_x[i]
            point2 = cur_means[k]
            euc_dises.append(euclidean_distance(point1, point2))
        min_idx = np.argmin(euc_dises)
        update_z[i][min_idx] = 1
    return update_z


def update_kmeans_means(sample_x, cur_z):
    """
    Updata means in K-Means
    sample_x: (num_sample, d)
    cur_z: (num_sample, K)
    return:
        update_means: updated means: (K, d)
    """
    num_sample, K = cur_z.shape
    feature_dimension = sample_x.shape[1]
    update_means = np.zeros((K, feature_dimension))
    count_cluser = np.zeros(K)
    # sum each cluster
    for i in range(num_sample):
        for k in range(K):
            if cur_z[i][k] == 1:
                update_means[k] += sample_x[i]
                count_cluser[k] += 1
    # get mean
    for k in range(K):
        if count_cluser[k] != 0:
            update_means[k] /= count_cluser[k]
    
    return update_means


def kmeans(sample_x, K, epsilon=10e-3, max_iter=10000):
    """
    Implementation of K-Means
    sample_x: (num_sample, d)
    K: number of cluster
    epsilon: change bound
    max_iter: number of maximum iteration
    return:
        cur_means: current means (K, d)
        cur_z: current z (num_sample, K)
    """
    num_sample = sample_x.shape[0]

    # initialized means
    cur_means = init_means(sample_x, K)
    # initialized z
    cur_z = np.zeros((num_sample, K))
    # initialized error
    mean_change = float("inf")

    iter_count = 0
    while (mean_change > epsilon) and (iter_count < max_iter):
        iter_count += 1
        cur_z = update_kmean_z(sample_x, cur_means, K)
        update_means = update_kmeans_means(sample_x, cur_z)
        # calculate mean change
        mean_change = np.max([euclidean_distance(update_means[k], cur_means[k]) for k in range(K)])
        # reassign cur_means
        cur_means = update_means
    return cur_means, cur_z


# EM-GMM