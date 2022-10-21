"""
Tools for Other Main Files
feature_dimension = d
"""
import random
import numpy as np
import matplotlib.pyplot as plt


POINT_COLOR = {
    0: 'red',
    1: 'green',
    2: 'blue',
    3: 'purple'
}


def get_data(data_name):
    """ 
    Get Data from Folder
    data_name: data name {A, B, C}
    return: data
        sample_x: (num_sample, d)
        sample_y: (num_sample, 1)
    """
    x_path = f"./cluster_data_text/cluster_data_data{data_name}_X.txt"
    y_path = f"./cluster_data_text/cluster_data_data{data_name}_Y.txt"
    sample_x = np.loadtxt(x_path)
    sample_y = np.loadtxt(y_path)
    sample_y = np.expand_dims(sample_y, -1)
    sample_x = sample_x.transpose()
    return sample_x, sample_y


def plot_kmeans(sample_x, cur_z, figure_num, plot_title):
    """
    Plot for K-Means
    sample_x: (num_sample, d)
    cur_z: (num_sample, K)
    figure_num: figure number: int
    plot_name: plot_title: string
    """
    num_sample, K = cur_z.shape
    plt.figure(figure_num)
    plt.title(plot_title)
    plt.xlabel('x1')
    plt.ylabel('x2')
    for i in range(num_sample):
        cluster_index = np.argmax(cur_z[i])
        plt.plot(sample_x[i][0], sample_x[i][1], '.', color=POINT_COLOR[cluster_index])
    plt.show()