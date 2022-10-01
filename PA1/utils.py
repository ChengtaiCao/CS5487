"""
Tools for other main files
"""
import numpy as np


def get_polydata(path_folder):
    """ get polydata from folder """
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
    """ poly feature transformation """
    def helper(feature, K):
        """ helper function """
        tranformed_feature = [feature ** k for k in range(K + 1)]
        tranformed_feature = np.array(tranformed_feature)
        return tranformed_feature
    
    tranformed_features = np.array([helper(feature, K) for feature in features])
    fearture_matrix = np.matrix(tranformed_features)
    fearture_matrix = fearture_matrix.transpose()
    return fearture_matrix


if __name__ == "__main__":
    path_folder = "./data/"
    sample_x, sample_y, poly_x, poly_y = get_polydata(path_folder)
    fearture_matrix = poly_feature_trans(sample_x, 5)
    print(fearture_matrix.shape == (50, 6))
