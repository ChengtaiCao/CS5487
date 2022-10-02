"""
Tools for other main files
"""
import numpy as np
import matplotlib.pyplot as plt


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
    mse_error = total_error/(len(estimation_y))
    return mse_error


def plot_figure(figure_num, title, sample_x, sample_y, poly_x, esitimation_y):
    """ 
    Plot Figure
    figure_num: number of figure
    title: title of figure
    sample_x: training x
    sample_y: training y
    poly_x: testing x
    esitimation_y: esitimation_y
    """
    plt.figure(figure_num)
    plt.title(title)
    plt.xlabel('x')
    plt.ylabel('y')
    plt.plot(sample_x, sample_y, '^', color="blue", label="sample")
    plt.plot(poly_x, esitimation_y, '.', color="red", label="poly")
    plt.legend(loc=1, ncol=1)
    plt.show()


def BR_plot_figure(figure_num, title, sample_x, sample_y, poly_x, esitimation_y, estimate_variance):
    """ 
    Plot Figure
    figure_num: number of figure
    title: title of figure
    sample_x: training x
    sample_y: training y
    poly_x: testing x
    esitimation_y: esitimation_y
    esitimation_var: esitimation_y_variance
    """
    plt.figure(figure_num)
    plt.title(title)
    plt.xlabel('x')
    plt.ylabel('y')
    plt.plot(sample_x, sample_y, '^', color="blue", label="sample")
    plt.plot(poly_x, esitimation_y, '.', color="red", label="poly")

    poly_x = list(np.squeeze(poly_x))
    esitimation_y = np.array(esitimation_y)
    esitimation_y = list(np.squeeze(esitimation_y))
    variance_list = []
    estimate_variance = np.array(estimate_variance)
    for i in range(len(estimate_variance)):
        variance_list.append(estimate_variance[i][i])

    r1 = list(map(lambda x: x[0]+abs(x[1]), zip(esitimation_y, variance_list)))
    r2 = list(map(lambda x: x[0]-abs(x[1]), zip(esitimation_y, variance_list)))
    plt.fill_between(poly_x, r1, r2, color="#FF3399", alpha=0.2)
    plt.legend()
    plt.show()