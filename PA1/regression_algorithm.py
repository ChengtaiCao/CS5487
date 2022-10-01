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
    sample_x: batch of transformed input [K + 1, 50]
    sample_y: batch of output [N_train, 1]
    return: theta [K + 1, 1]
    """
    sample_x_tran = sample_x.transpose()
    inver_matrix = np.linalg.inv(sample_x.dot(sample_x_tran))
    theta = inver_matrix.dot(sample_x).dot(sample_y)
    return theta


def RLS_estimation(sample_x, sample_y, RLS_lambda=1):
    """ 
    Regularized least-squares (RLS) Estimation
    sample_x: batch of transformed input [K + 1, 50]
    sample_y: batch of output [N_train, 1]
    RLS_lambda: hyper-parameter for regularization term
    return: theta [K + 1, 1]
    """
    feature_dimension = sample_x.shape[0]
    sample_x_tran = sample_x.transpose()
    inver_matrix = np.linalg.inv(sample_x.dot(sample_x_tran) + RLS_lambda * np.identity(feature_dimension))
    theta = inver_matrix.dot(sample_x).dot(sample_y)
    return theta


def LASSO_estimation(sample_x, sample_y, LASSO_lambda=1):
    """ 
    LASSO estimation
    min 1/2 theta^T * H * theta + f^T * theta, g^T * theta >= 0
    sample_x: batch of transformed input [K + 1, 50]
    sample_y: batch of output [50, 1]
    LASSO_lambda: hyper-parameter for regularization term
    return: theta [K + 1, 1]
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


def prediction(poly_x, theta):
    """ prediction """
    estimate_y = poly_x.transpose().dot(theta)
    return estimate_y
