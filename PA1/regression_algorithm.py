"""
Implement the above 5 regression algorithms for the K-th order polynomial
"""
import numpy as np
import scipy
import cvxopt
import matplotlib.pyplot as plt


def LS_estimation(sample_x, sample_y):
    """ 
    Least-squares (LS) Estimation
    sample_x: batch of transformed input: [K + 1, 50]
    sample_y: batch of output: [N_train, 1]
    return: estimation parameters
        theta: [K + 1, 1]
    """
    sample_x_tran = sample_x.transpose()
    inver_matrix = np.linalg.inv(sample_x.dot(sample_x_tran))
    theta = inver_matrix.dot(sample_x).dot(sample_y)
    return theta


def RLS_estimation(sample_x, sample_y, RLS_lambda=1):
    """ 
    Regularized least-squares (RLS) Estimation
    sample_x: batch of transformed input: [K + 1, 50]
    sample_y: batch of output: [N_train, 1]
    RLS_lambda: hyper-parameter for regularization term
    return: estimation parameters
        theta: [K + 1, 1]
    """
    feature_dimension = sample_x.shape[0]
    sample_x_tran = sample_x.transpose()
    inver_matrix = np.linalg.inv(sample_x.dot(sample_x_tran) + RLS_lambda * np.identity(feature_dimension))
    theta = inver_matrix.dot(sample_x).dot(sample_y)
    return theta


def LASSO_estimation(sample_x, sample_y, LASSO_lambda=1):
    """ 
    LASSO Estimation
    min 1/2 theta^T * H * theta + f^T * theta, g^T * theta >= 0
    sample_x: batch of transformed input: [K + 1, 50]
    sample_y: batch of output: [50, 1]
    LASSO_lambda: hyper-parameter for regularization term
    return: estimation parameters
        theta: [K + 1, 1]
    """
    feature_dimension = sample_x.shape[0]
    sample_x_tran = sample_x.transpose()
    H = np.zeros((2 * feature_dimension, 2 * feature_dimension))
    H[:feature_dimension, :feature_dimension] = sample_x.dot(sample_x_tran)
    H[:feature_dimension, feature_dimension:] = -1 * sample_x.dot(sample_x_tran)
    H[feature_dimension:, :feature_dimension] = -1 * sample_x.dot(sample_x_tran)
    H[feature_dimension:, feature_dimension:] = sample_x.dot(sample_x_tran)
    # H: (2 * (K + 1), 2 * (K + 1))
    x_y = sample_x.dot(sample_y)    # x_y: [K + 1, 1]
    f = LASSO_lambda * np.ones((2 * feature_dimension, 1)) - np.concatenate((x_y, -1 * x_y), axis=0)
    # f: (2 * (K + 1, 1)
    g = -1 * np.identity(2 * feature_dimension)
    # g: (2 * (K + 1), 2 * (K + 1))
    values = np.zeros((2 * feature_dimension, 1))
    # value: (2 * (K + 1, 1)
    qp_result = cvxopt.solvers.qp(cvxopt.matrix(H), cvxopt.matrix(f), cvxopt.matrix(g), cvxopt.matrix(values))['x']
    # qp_result: (2 * (K + 1, 1)
    theta = [qp_result[i] - qp_result[i + feature_dimension] for i in range(feature_dimension)]
    theta = np.array(theta).transpose()
    # theta: [K + 1, 1]
    return theta


def RR_estimation(sample_x, sample_y):
    """ 
    Robust Regression (RR) estimation
    sample_x: batch of transformed input: [K + 1, 50]
    sample_y: batch of output: [50, 1]
    return: estimation parameters
        theta: [K + 1, 1]
    """
    feature_dimension = sample_x.shape[0]
    trainging_num = sample_x.shape[1]
    sample_x_tran = sample_x.transpose()
    f = np.concatenate((np.zeros((feature_dimension, 1)), np.ones((trainging_num, 1))), axis=0)
    # f: (56, 1)
    H = np.zeros((2 * trainging_num, feature_dimension + trainging_num))
    H[:trainging_num, :feature_dimension] = -1 * sample_x_tran
    H[:trainging_num, feature_dimension:] = -1 * np.identity(trainging_num)
    H[trainging_num:, :feature_dimension] = sample_x_tran
    H[trainging_num:, feature_dimension:] = -1 * np.identity(trainging_num)
    # H: (100, 56)
    G = np.concatenate((-1 * sample_y, sample_y), axis=0)
    # G: (50, 1)
    lp_result = cvxopt.solvers.lp(cvxopt.matrix(f), cvxopt.matrix(H), cvxopt.matrix(G))['x']
    theta = lp_result[:feature_dimension]
    # theta: [K + 1, 1]
    return theta


def BR_estimation(sample_x, sample_y, BR_alpha=1, BR_variance = 5):
    """ 
    Bayesian Regression (BR) Estimation
    sample_x: batch of transformed input: [K + 1, 50]
    sample_y: batch of output: [50, 1]
    BR_alpha: variance of prior
    return: estimation parameters
        theta_mean: [K + 1, 1]
        theta_cov: [K + 1, K + 1]
    """
    feature_dimension = sample_x.shape[0]
    sample_x_tran = sample_x.transpose()
    mat_1 = np.matrix((1 / BR_alpha) * np.identity(feature_dimension) + (1/BR_variance) * sample_x.dot(sample_x_tran))
    theta_cov = np.linalg.inv(mat_1)
    theta_mean = (1 / BR_variance) * theta_cov.dot(sample_x).dot(sample_y)
    return theta_mean, theta_cov


def prediction(poly_x, theta):
    """ 
    Prediction 
    poly_x: batch of transformed input: [K + 1, 100]
    theta: estimated parameter: [K + 1, 1]
    return: estimation poly_y
        estimate_y: [100, 1]
    """
    poly_x_tran = poly_x.transpose()
    estimate_y = poly_x_tran.dot(theta)
    return estimate_y


def BR_prediction(poly_x, theta_mean, theta_cov):
    """ 
    Bayesion Prediction 
    poly_x: batch of transformed input: [K + 1, 100]
    theta_mean: estimated parameter mean: [K + 1, 1]
    theta_cov: estimated parameter convaraince: [K + 1, 1 + 1]
    return: estimation polt_y
        estimate_mean: [100, 1]
        estimate_variance: [100, 100]
    """
    poly_x_tran = poly_x.transpose()
    estimate_mean = poly_x_tran.dot(theta_mean)
    # (100, 1)
    estimate_variance = poly_x_tran.dot(theta_cov).dot(poly_x)
    # (100, 100)
    return estimate_mean, estimate_variance
