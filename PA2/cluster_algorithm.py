"""
Implement the Above 3 Clustering Algorithms
feature_dimension = d
"""
import numpy as np
from scipy.stats import multivariate_normal


np.random.seed(1)
EPSILON = 10e-7
MAX_ITER = 10e4


def euclidean_distance(point1, point2):
    """ 
    Calculating Euclidean Distance betwean point1 and point2
    point1: point1: (d,)
    point2: point2: (d,)
    return:
        euc_dis: scalar
    """
    euc_dis = np.sqrt(np.sum(np.square(point1 - point2)))
    return euc_dis


# K-Means
def init_means(sample_x, K):
    """ 
    Getting Initialized Means for K-Means
    sample_x: (num_sample, d)
    K: number of cluster
    return:
        means: K initialized d-dim vector for K clusters: (K, d) 
        z: initialized z: (num_sample, K)
    """
    num_sample = sample_x.shape[0]
    mean_index = np.random.randint(num_sample, size=K)
    means = sample_x[mean_index]
    z = np.zeros((num_sample, K))
    return means, z


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


def k_means(sample_x, K, epsilon=EPSILON, max_iter=MAX_ITER):
    """
    Implementation of K-Means
    sample_x: (num_sample, d)
    K: number of cluster
    epsilon: change bound
    max_iter: number of maximum iteration
    return:
        cur_means: current means: (K, d)
        cur_z: current z: (num_sample, K)
    """
    num_sample = sample_x.shape[0]

    # initialized K-Means
    cur_means, cur_z = init_means(sample_x, K)
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


# EM algorithm for Gaussian mixture models (EM-GMM)
def init_gmm(sample_x, K):
    """ 
    Getting Initialized Statistics fpr GMM
    sample_x: (num_sample, d)
    K: number of cluster
    return:
        init_mean: initialized mean: (K, d)
        init_cov: initialized convarainces: (K, d, d)
        init_pi: initialized pi: (K,)
    """
    num_sample, feature_dimension = sample_x.shape
    init_mean = np.random.rand(K, feature_dimension)
    init_cov = np.array([np.eye(feature_dimension)] * K)
    init_pi = np.array([1 / K] * K)
    return init_mean, init_cov, init_pi


def cal_prob(sample_x, cur_mean, cur_cov):
    """
    Calculate Each Probability
    sample_x: (num_sample, d)
    cur_mean: current mean: (d,)
    cur_cov: current convaraince: (d, d)
    return:
        res: guassian PDF: (num_sample,)
    """
    gua = multivariate_normal(mean=cur_mean, cov=cur_cov)
    res = gua.pdf(sample_x)
    return res


def EStep(sample_x, cur_means, cur_covs, cur_pis):
    """
    E Step for GMM
    sample_x: (num_sample, d)
    cur_means: current means: (K, d)
    cur_covs: current convarainces: (K, d, d)
    cur_pis: current pi: (K,)
    return:
        z: (num_sample, K)
    """
    num_sample = sample_x.shape[0]
    K = cur_means.shape[0]
    z = np.mat(np.zeros((num_sample, K)))
    prob = np.zeros((num_sample, K))
    for k in range(K):
        prob[:, k] = cal_prob(sample_x, cur_means[k], cur_covs[k])
    prob = np.matrix(prob)
    for k in range(K):
        z[:, k] = cur_pis[k] * prob[:, k]
    for i in range(num_sample):
        z[i] /= np.sum(z[i])
    return z


def MStep(sample_x, cur_z):
    """
    M Step for GMM
    sample_x: (num_sample, d)
    cur_z: current z: (num_sample, K)
    return:
        update_means: current means: (K, d)
        update_covs: current convarainces: (K, d, d)
        update_pis: current pis: (K,)
    """
    (num_sample, feature_dimension), K = sample_x.shape, cur_z.shape[1]

    update_means = np.zeros((K, feature_dimension))
    update_covs = []
    update_pis = np.zeros(K)

    for k in range(K):
        allk = np.sum(cur_z[:, k])
        update_means[k] = np.sum(np.multiply(sample_x, cur_z[:, k]), axis=0) / allk
        update_cov = (sample_x - update_means[k]).transpose() * np.multiply((sample_x - update_means[k]), cur_z[:, k]) / allk
        update_covs.append(update_cov)
        update_pis[k] = allk / num_sample
    update_covs = np.array(update_covs)
    return update_means, update_covs, update_pis


def em_gmm(sample_x, K, epsilon=EPSILON, max_iter=MAX_ITER):
    """
    Implementation of K-Means
    sample_x: (num_sample, d)
    K: number of cluster
    epsilon: change bound
    max_iter: number of maximum iteration
    return:
        cur_means: current means: (K, d)
        cur_covs: current convarainces: (K, d, d)
        cur_pis: current pi: (K,)
    """
    # initialized GMM
    cur_means, cur_covs, cur_pis = init_gmm(sample_x, K)
    # initialized error
    mean_change = float("inf")
    iter_count = 0
    while (mean_change > epsilon) and (iter_count < max_iter):
        iter_count += 1
        cur_z = EStep(sample_x, cur_means, cur_covs, cur_pis)
        update_means, update_covs, update_pis = MStep(sample_x, cur_z)
        # calculate mean change
        mean_change = np.max([euclidean_distance(update_means[k], cur_means[k]) for k in range(K)])
        # reassign cur_means, cur_covs, cur_pis
        cur_means, cur_covs, cur_pis = update_means, update_covs, update_pis
    print(iter_count)
    return cur_means, cur_covs, cur_pis


# Mean-shift algorithm