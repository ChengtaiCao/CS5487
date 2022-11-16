"""
Implementation of Utils Function
"""
import numpy as np

from tensorflow.keras.utils import to_categorical
from sklearn.utils import shuffle


SEED = 42


def get_data(file_path, seed=SEED):
    """
    Get Data
    Input:
        file_path: file path
    Return:
        data_dict:
            train_x_1: trial 1 train input  (num_sampels, 784)
            train_y_1: trial 1 train ouput  (num_sampels,)
            test_x_1: trial 1 test input    (num_sampels, 784)
            test_y_1: trial 1 test ouput    (num_sampels,)
            train_x_2: trial 2 train input  (num_sampels, 784)
            train_y_2: trial 2 train ouput  (num_sampels,)
            test_x_2: trial 2 test input    (num_sampels, 784)
            test_y_2: trial 2 test ouput    (num_sampels,)
    """
    NUM_CLASS = 10
    X_path = f"{file_path}/digits4000_digits_vec.txt"
    Y_path = f"{file_path}/digits4000_digits_labels.txt"

    X = np.loadtxt(X_path).astype(np.float32)
    X /= 255
    Y = np.loadtxt(Y_path)

    train_x_1, train_y_1 = X[:2000], Y[:2000]
    test_x_1, test_y_1 = X[2000:], Y[2000:]
    train_x_1, train_y_1 = shuffle(train_x_1, train_y_1, random_state=seed)
    test_x_1, test_y_1 = shuffle(test_x_1, test_y_1, random_state=seed)

    train_x_2, train_y_2 = X[2000:], Y[2000:]
    test_x_2, test_y_2 = X[:2000], Y[:2000]
    train_x_2, train_y_2 = shuffle(train_x_2, train_y_2, random_state=seed)
    test_x_2, test_y_2 = shuffle(test_x_2, test_y_2, random_state=seed)
    
    data_dict = {
        "train_x_1": train_x_1,
        "train_y_1": train_y_1,
        "test_x_1": test_x_1,
        "test_y_1": test_y_1,
        "train_x_2": train_x_2,
        "train_y_2": train_y_2,
        "test_x_2": test_x_2,
        "test_y_2": test_y_2,
    }

    return data_dict


def pre_process_MLP(data_dict, NUM_CLASS=10):
    """
    Data Process For MLP
    Input:
        data_dict: data_dict
    Return:
        trail_1: data for trail 1
            train_x: trial 1 train input  (num_sampels, 784)
            train_y: trial 1 train ouput  (num_sampels, 10)
            validation_x: trial 1 validation input  (num_sampels, 784)
            validation_y: trial 1 validation ouput  (num_sampels, 10)
            test_x: trial 1 test input    (num_sampels, 784)
            test_y: trial 1 test ouput    (num_sampels, 10)
        trail_2: data for trail 2
    """
    train_x_1_origin = data_dict["train_x_1"]
    train_y_1 = data_dict["train_y_1"]
    test_x_1 = data_dict["test_x_1"]
    test_y_1 = data_dict["test_y_1"]
    train_x_2_origin = data_dict["train_x_2"]
    train_y_2 = data_dict["train_y_2"]
    test_x_2 = data_dict["test_x_2"]
    test_y_2 = data_dict["test_y_2"]

    # one-hot encoding label
    train_y_1_origin = to_categorical(train_y_1, NUM_CLASS)
    test_y_1 = to_categorical(test_y_1, NUM_CLASS)
    train_y_2_origin = to_categorical(train_y_2, NUM_CLASS)
    test_y_2 = to_categorical(test_y_2, NUM_CLASS)

    # validation split
    length = train_x_1_origin.shape[0]
    split_index = int(length * 0.8)

    train_x_1 = train_x_1_origin[:split_index]
    validation_x_1 = train_x_1_origin[split_index:]
    train_y_1 = train_y_1_origin[:split_index]
    validation_y_1 = train_y_1_origin[split_index:]

    train_x_2 = train_x_2_origin[:split_index]
    validation_x_2 = train_x_2_origin[split_index:]
    train_y_2 = train_y_2_origin[:split_index]
    validation_y_2 = train_y_2_origin[split_index:]

    trail_1 = {
        "train_x": train_x_1,
        "train_y": train_y_1,
        "validation_x": validation_x_1,
        "validation_y": validation_y_1,
        "test_x": test_x_1,
        "test_y": test_y_1,
    }

    trail_2 = {
        "train_x": train_x_2,
        "train_y": train_y_2,
        "validation_x": validation_x_2,
        "validation_y": validation_y_2,
        "test_x": test_x_2,
        "test_y": test_y_2,
    }

    return trail_1, trail_2


def pre_process_CNN(data_dict, NUM_CLASS=10):
    """
    Data Process For CNN
    Input:
        data_dict: data_dict
    Return:
        trail_1: data for trail 1
            train_x: trial 1 train input  (num_sampels, 28, 28, 1)
            train_y: trial 1 train ouput  (num_sampels, 10)
            validation_x: trial 1 validation input  (num_sampels, 28, 28, 1)
            validation_y: trial 1 validation ouput  (num_sampels, 10)
            test_x: trial 1 test input    (num_sampels, 28, 28, 1)
            test_y: trial 1 test ouput    (num_sampels, 10)
        trail_2: data for trail 2
    """
    train_x_1 = data_dict["train_x_1"]
    train_y_1 = data_dict["train_y_1"]
    test_x_1 = data_dict["test_x_1"]
    test_y_1 = data_dict["test_y_1"]
    train_x_2 = data_dict["train_x_2"]
    train_y_2 = data_dict["train_y_2"]
    test_x_2 = data_dict["test_x_2"]
    test_y_2 = data_dict["test_y_2"]

    # reshape to 2-D
    train_x_1_origin = np.reshape(train_x_1, [-1, 28, 28, 1])
    test_x_1 = np.reshape(test_x_1, [-1, 28, 28, 1])
    train_x_2_origin = np.reshape(train_x_2, [-1, 28, 28, 1])
    test_x_2 = np.reshape(test_x_2, [-1, 28, 28, 1])

    # one-hot encoding label
    train_y_1_origin = to_categorical(train_y_1, NUM_CLASS)
    test_y_1 = to_categorical(test_y_1, NUM_CLASS)
    train_y_2_origin = to_categorical(train_y_2, NUM_CLASS)
    test_y_2 = to_categorical(test_y_2, NUM_CLASS)

    # validation split
    length = train_x_1_origin.shape[0]
    split_index = int(length * 0.8)

    train_x_1 = train_x_1_origin[:split_index]
    validation_x_1 = train_x_1_origin[split_index:]
    train_y_1 = train_y_1_origin[:split_index]
    validation_y_1 = train_y_1_origin[split_index:]

    train_x_2 = train_x_2_origin[:split_index]
    validation_x_2 = train_x_2_origin[split_index:]
    train_y_2 = train_y_2_origin[:split_index]
    validation_y_2 = train_y_2_origin[split_index:]

    trail_1 = {
        "train_x": train_x_1,
        "train_y": train_y_1,
        "validation_x": validation_x_1,
        "validation_y": validation_y_1,
        "test_x": test_x_1,
        "test_y": test_y_1,
    }

    trail_2 = {
        "train_x": train_x_2,
        "train_y": train_y_2,
        "validation_x": validation_x_2,
        "validation_y": validation_y_2,
        "test_x": test_x_2,
        "test_y": test_y_2,
    }

    return trail_1, trail_2


def pre_process_ML(data_dict, encoding_label=False, NUM_CLASS=10):
    """
    Data Process For Machine Learning
    Input:
        data_dict: data_dict
    Return:
        trail_1: data for trail 1
            train_x: trial 1 train input  (num_sampels, 784)
            train_y: trial 1 train ouput  (num_sampels, 10)
            test_x: trial 1 test input    (num_sampels, 784)
            test_y: trial 1 test ouput    (num_sampels, 10)
        trail_2: data for trail 2
    """
    train_x_1 = data_dict["train_x_1"]
    train_y_1 = data_dict["train_y_1"]
    test_x_1 = data_dict["test_x_1"]
    test_y_1 = data_dict["test_y_1"]
    train_x_2 = data_dict["train_x_2"]
    train_y_2 = data_dict["train_y_2"]
    test_x_2 = data_dict["test_x_2"]
    test_y_2 = data_dict["test_y_2"]

    if encoding_label:
        # one-hot encoding label
        train_y_1 = to_categorical(train_y_1, NUM_CLASS)
        test_y_1 = to_categorical(test_y_1, NUM_CLASS)
        train_y_2 = to_categorical(train_y_2, NUM_CLASS)
        test_y_2 = to_categorical(test_y_2, NUM_CLASS)
    

    trail_1 = {
        "train_x": train_x_1,
        "train_y": train_y_1,
        "test_x": test_x_1,
        "test_y": test_y_1,
    }

    trail_2 = {
        "train_x": train_x_2,
        "train_y": train_y_2,
        "test_x": test_x_2,
        "test_y": test_y_2,
    }

    return trail_1, trail_2
