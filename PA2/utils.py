"""
Tools for Other Main Files
feature_dimension = d
"""
from enum import Flag
import pdb
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import os
import pylab as pl

from pa2 import labels2seg, colorsegms


np.random.seed(1)

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


def plot1(sample_x, cur_z, figure_num, plot_title):
    """
    Plot for K-Means
    sample_x: (num_sample, d)
    cur_z: (num_sample, K)
    figure_num: figure number: int
    plot_name: plot_title: string
    """
    num_sample = cur_z.shape[0]
    plt.figure(figure_num)
    plt.title(plot_title)
    plt.xlabel('x1')
    plt.ylabel('x2')
    for i in range(num_sample):
        cluster_index = np.argmax(cur_z[i])
        plt.plot(sample_x[i][0], sample_x[i][1], '.', color=POINT_COLOR[cluster_index])
    plt.show()


def plot2(sample_x, labels, num_peak, figure_num, plot_title):
    """
    Plot for Mean-Shift
    sample_x: (num_sample, d)
    labels: (num_sample,)
    num_peak: scalar
    figure_num: figure number: int
    plot_name: plot_title: string
    """
    num_sample = len(labels)
    colors = cm.rainbow(np.linspace(0, 1, num_peak))
    plt.figure(figure_num)
    plt.title(plot_title)
    plt.xlabel('x1')
    plt.ylabel('x2')
    for i in range(num_sample):
        cluster_index = labels[i]
        plt.plot(sample_x[i][0], sample_x[i][1], '.', color=colors[cluster_index])
    plt.show()


def get_images(input_path = "./images"):
    """
    Get Images for Problem 2
    input_path: image path
    return:
        res: path list
    """
    res = []
    for filename in os.listdir(input_path):
        res.append(input_path + "/" + filename)
    return res


def plot3(img, L, cur_z, figure_num, plot_title, flag=False):
    """
    Plot for Problem 2
    img: img
    L: scalar
    cur_z: (num_sample, K)
    figure_num: figure number: int
    plot_title: plot title
    flag: if Mean-Shift, True
    """
    pl.figure(figure_num)
    pl.subplot(1, 3, 1)
    pl.title('Original Image')
    pl.imshow(img)
    
    if not flag:
        num_sample = cur_z.shape[0]
        cur_z = np.array([np.argmax(cur_z[i]) for i in range(num_sample)])
    else:
        cur_z = np.array(cur_z)
    
    seg = labels2seg(cur_z, L)
    pl.subplot(1, 3, 2)
    pl.title('Segmentation')
    pl.imshow(seg)

    color_seg = colorsegms(seg, img)
    pl.subplot(1, 3, 3)
    pl.title(plot_title)
    pl.imshow(color_seg)
    pl.show()
