"""
Tools for other main files
"""
import numpy as np


def get_polydata(path_folder):
    """ 
    Get Polydata from Folder
    path_folder: folder path
    return: training data & testing data
        sample_x: training x: (50, 1)
        sample_y: training y: (50, 1)
        poly_x: testing x: (100, 1)
        poly_y: testing y: (100, 1)
    """
    sample_x = np.loadtxt(f"{path_folder}polydata_data_sampx.txt")
    sample_y = np.loadtxt(f"{path_folder}polydata_data_sampy.txt")
    poly_x = np.loadtxt(f"{path_folder}polydata_data_polyx.txt")
    poly_y = np.loadtxt(f"{path_folder}polydata_data_polyy.txt")

    sample_x = sample_x.reshape(-1, 1)
    sample_y = sample_y.reshape(-1, 1)
    poly_x = poly_x.reshape(-1, 1)
    poly_y = poly_y.reshape(-1, 1)

    return sample_x, sample_y, poly_x, poly_y


def poly_feature_trans(features, K):
    """ 
    Poly Feature Transformation
    features: input feature: [N, 1]
    return: fearture_matrix after transformed
        fearture_matrix: [K, N]
    """
    def helper(feature, K):
        """ helper function """
        tranformed_feature = [feature ** k for k in range(K + 1)]
        tranformed_feature = np.array(tranformed_feature)
        return tranformed_feature
    
    tranformed_features = np.array([helper(feature, K) for feature in features])
    fearture_matrix = np.matrix(tranformed_features)
    fearture_matrix = fearture_matrix.transpose()
    return fearture_matrix


def get_error(estimation_y, poly_y):
    """ 
    Mean-Squared Error for estimation_y & poly_y
    estimation_y: estimation y: [100, 1]
    poly_y: ground truth y: [100, 1]
    return: mse_error: scalar
    """
    estimation_y = np.array(estimation_y)
    poly_y = np.array(poly_y)
    total_error = ((estimation_y - poly_y) ** 2).sum()
    print(estimation_y.shape)
    mse_error = total_error/(len(estimation_y))
    return mse_error
