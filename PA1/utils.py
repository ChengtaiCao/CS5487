"""
Tools for other main files
"""
import pdb
import numpy as np
import matplotlib.pyplot as plt


def get_polydata(path_folder, ratio=1):
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
    # random shuffile
    state = np.random.get_state()
    np.random.shuffle(sample_x)
    np.random.set_state(state)
    np.random.shuffle(sample_y)
    # get subset of data
    ratio_length = int(sample_x.shape[0] * ratio)
    sample_x = sample_x[:ratio_length]
    sample_y = sample_y[:ratio_length]
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
    plt.legend(loc=1)
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
    plt.legend(loc=1)
    plt.show()


def plot_figure_for_training_size(LS_errors, RLS_errors, LASSO_errors, RR_errors, BR_errors):
    """ 
    Plot Figure for Training Size
    split_num = 10
    repeat_num = 10
    LS_errors: error array [split_num, repeat_num] 
    RLS_errors: error array [split_num, repeat_num] 
    LASSO_errors: error array [split_num, repeat_num] 
    RR_errors: error array [split_num, repeat_num] 
    BR_errors: error array [split_num, repeat_num] 
    """
    LS_errors_mean = np.mean(LS_errors, axis=1)
    RLS_errors_mean = np.mean(RLS_errors, axis=1)
    LASOO_errors_mean = np.mean(LASSO_errors, axis=1)
    RR_errors_mean = np.mean(RR_errors, axis=1)
    BR_errors_mean = np.mean(BR_errors, axis=1)

    plt.title("Error Versus Training Size")
    plt.xlabel('Ratio')
    plt.ylabel('Error')
    x = [0.1 + i * 0.1 for i in range(10)]
    plt.plot(x, LS_errors_mean, 'o-', color="#ff0080", label="LS")
    plt.plot(x, RLS_errors_mean, 'o-', color="#3300FF", label="RLS")
    plt.plot(x, LASOO_errors_mean, 'o-', color="#33FF00", label="LASSO")
    plt.plot(x, RR_errors_mean, 'o-', color="#9966FF", label="RR")
    plt.plot(x, BR_errors_mean, 'o-', color="#FF9900", label="BR")
    plt.legend(loc=1)
    plt.show()
