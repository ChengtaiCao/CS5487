"""
Implement the above 5 regression algorithms for the K-th order polynomial
"""
import pdb
import numpy as np
from utils import *


def LS_estimation(sample_x, sample_y):
    """ least-squares (LS) estimation """
    inver_matrix = np.linalg.inv(sample_x.dot(sample_x.transpose()))
    theta = inver_matrix.dot(sample_x).dot(sample_y)
    return theta


def RLS_estimation(sample_x, sample_y, RLS_lambda=1):
    """ regularized  least-squares (RLS) estimation """
    inver_matrix = np.linalg.inv(sample_x.dot(sample_x.transpose()) + RLS_lambda * np.identity(len(sample_x)))
    theta = inver_matrix.dot(sample_x).dot(sample_y)
    return theta


def prediction(poly_x, theta):
    """ prediction """
    pre_res = poly_x.transpose().dot(theta)
    return pre_res


if __name__ == "__main__":
    K = 5
    path_folder = "./data/"
    sample_x, sample_y, poly_x, poly_y = get_data(path_folder)
    sample_x = feature_trans(sample_x, K)
    poly_x = feature_trans(poly_x, K)
    theta = RLS_estimation(sample_x, sample_y)
    pre_res = prediction(poly_x, theta)
    print(theta)
    print(theta.shape)