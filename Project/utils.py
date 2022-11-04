"""
Utils
"""
import numpy as np
import pdb
import tensorflow as tf
import keras
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten
from tensorflow.keras.layers import Dropout, Dense, BatchNormalization


AUTO = tf.data.AUTOTUNE


def get_data(file_path):
    """
    Get Data
    Input:
        file_path: file path
    Return:
        train_x_1: trial 1 train input  (num_sampels, 28, 28, 1)
        train_y_1: trial 1 train ouput  (num_sampels, num_classes)
        test_x_1: trial 1 test input    (num_sampels, 28, 28, 1)
        test_y_1: trial 1 test ouput    (num_sampels, num_classes)
        train_x_2: trial 2 train input  (num_sampels, 28, 28, 1)
        train_y_2: trial 2 train ouput  (num_sampels, num_classes)
        test_x_2: trial 2 test input    (num_sampels, 28, 28, 1)
        test_y_2: trial 2 test ouput    (num_sampels, num_classes)
        
    """
    NUM_CLASS = 10
    X_path = f"{file_path}/digits4000_digits_vec.txt"
    Y_path = f"{file_path}/digits4000_digits_labels.txt"

    X = np.loadtxt(X_path).astype(np.float32)
    X /= 255
    # reshape as (-1, 28, 28, 1)
    X = tf.reshape(X, [-1, 28, 28, 1])
    Y = np.loadtxt(Y_path)
    # one-hot encoding
    Y = keras.utils.to_categorical(Y, NUM_CLASS)

    train_x_1, train_y_1 = X[:2000], Y[:2000]
    test_x_1, test_y_1 = X[2000:], Y[2000:]
    train_x_2, train_y_2 = X[2000:], Y[2000:]
    test_x_2, test_y_2 = X[:2000], Y[:2000]

    return train_x_1, train_y_1, test_x_1, test_y_1, train_x_2, train_y_2, test_x_2, test_y_2


def get_mix_data(file_path):
    """
    Get Mixed Data
    Input:
        file_path: file path
    Return:
        train_x_1: trial 1 train input  (num_sampels, 28, 28, 1)
        train_y_1: trial 1 train ouput  (num_sampels, num_classes)
        test_x_1: trial 1 test input    (num_sampels, 28, 28, 1)
        test_y_1: trial 1 test ouput    (num_sampels, num_classes)
        train_x_2: trial 2 train input  (num_sampels, 28, 28, 1)
        train_y_2: trial 2 train ouput  (num_sampels, num_classes)
        test_x_2: trial 2 test input    (num_sampels, 28, 28, 1)
        test_y_2: trial 2 test ouput    (num_sampels, num_classes)
        
    """
    NUM_CLASS = 10
    X_path = f"{file_path}/digits4000_digits_vec.txt"
    Y_path = f"{file_path}/digits4000_digits_labels.txt"

    X = np.loadtxt(X_path).astype(np.float32)
    X /= 255
    # reshape as (-1, 28, 28, 1)
    X = tf.reshape(X, [-1, 28, 28, 1])
    Y = np.loadtxt(Y_path)
    # one-hot encoding
    Y = keras.utils.to_categorical(Y, NUM_CLASS)

    train_x_1, train_y_1 = X[:2000], Y[:2000]
    test_x_1, test_y_1 = X[2000:], Y[2000:]
    new_x_train, new_y_train = train_x_1, train_y_1
    BATCH_SIZE = 128
    train_ds_one = (
    tf.data.Dataset.from_tensor_slices((new_x_train, new_y_train))
    .shuffle(BATCH_SIZE * 100)
    .batch(BATCH_SIZE)
    )
    train_ds_two = (
        tf.data.Dataset.from_tensor_slices((new_x_train, new_y_train))
        .shuffle(BATCH_SIZE * 100)
        .batch(BATCH_SIZE)
    )
    train_ds = tf.data.Dataset.zip((train_ds_one, train_ds_two))
    train_ds_mu = train_ds.map(
        lambda ds_one, ds_two: mix_up(ds_one, ds_two, alpha=0.2), num_parallel_calls=AUTO
    )
    train_x_2, train_y_2 = X[2000:], Y[2000:]
    test_x_2, test_y_2 = X[:2000], Y[:2000]

    return train_ds_mu, test_x_1, test_y_1


def get_model(input_shape, num_classes, nonlinear="relu"):
    """
    Get Model
    Input:
        input_shape: input shape
        num_classes: num of classes
        activate: activate function
    Return:
        model
    """
    model = keras.Sequential([
        keras.Input(shape=input_shape),
        Conv2D(32, kernel_size=(3, 3), activation=nonlinear),
        MaxPooling2D(pool_size=(2, 2)),
        Conv2D(64, kernel_size=(3, 3), activation=nonlinear),
        MaxPooling2D(pool_size=(2, 2)),
        Flatten(),
        Dropout(0.5),
        Dense(num_classes, activation="softmax"),
    ])
    return model


def get_model2(input_shape, num_classes, nonlinear="relu"):
    """
    Get Model
    Input:
        input_shape: input shape
        num_classes: num of classes
        activate: activate function
    Return:
        model
    """
    model = keras.Sequential([
        keras.Input(shape=input_shape),
        Conv2D(filters=64, kernel_size = (3,3), activation=nonlinear),
        Conv2D(filters=64, kernel_size = (3,3), activation=nonlinear),
        MaxPooling2D(pool_size=(2,2)),
        BatchNormalization(),
        Conv2D(filters=128, kernel_size = (3,3), activation=nonlinear),
        Conv2D(filters=128, kernel_size = (3,3), activation=nonlinear),
        MaxPooling2D(pool_size=(2,2)),
        BatchNormalization(),
        Conv2D(filters=256, kernel_size = (3,3), activation=nonlinear),
        MaxPooling2D(pool_size=(2,2)),
        BatchNormalization(),
        Flatten(),
        Dense(512,activation=nonlinear),
        Dense(num_classes,activation="softmax")
    ])  
    return model


def sample_beta_distribution(size, concentration_0=0.2, concentration_1=0.2):
    """
    Sample Beta Distribution
    input:
        size: size of mix ratio
        concentration_0: 0-coefficient
        concentration_1: 1-coefficient
    return:
        mixup ratio: (size)
    """
    gamma_1_sample = tf.random.gamma(shape=[size], alpha=concentration_0)
    gamma_2_sample = tf.random.gamma(shape=[size], alpha=concentration_1)
    return gamma_1_sample / (gamma_1_sample + gamma_2_sample)


def mix_up(ds_one, ds_two, alpha=0.2):
    """
    Sample Beta Distribution
    input:
        ds_one: batch one
        ds_two: batch two
        alpha: parameter for beta distribution
    return:
        (images, labels)
    """
    images_one, labels_one = ds_one
    images_two, labels_two = ds_two
    batch_size = tf.shape(images_one)[0]
    l = sample_beta_distribution(batch_size, alpha, alpha)
    x_l = tf.reshape(l, (batch_size, 1, 1, 1))
    y_l = tf.reshape(l, (batch_size, 1))
    images = images_one * x_l + images_two * (1 - x_l)
    labels = labels_one * y_l + labels_two * (1 - y_l)
    return (images, labels)
