"""
Implementation of Model
"""
import tensorflow as tf
import keras
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Activation
from tensorflow.keras.layers import Dropout, Dense, BatchNormalization
from tensorflow.keras import activations


def get_shallow_cnn(input_shape, num_classes=10, dropout=0.25, reg=0.01):
    """
    Get Shallow CNN Model
    Input:
        input_shape: input shape
        num_classes: num of classes
        activate: activate function
    Return:
        model
    """
    model = keras.Sequential([
        keras.Input(shape=input_shape),

        Conv2D(32, kernel_size=(3, 3)),
        BatchNormalization(),
        Activation(activations.relu),
        MaxPooling2D(pool_size=(2, 2)),

        Conv2D(64, kernel_size=(3, 3)),
        BatchNormalization(),
        Activation(activations.relu),
        MaxPooling2D(pool_size=(2, 2)),

        Flatten(),
        Dropout(dropout),
        Dense(num_classes, activation="softmax", kernel_regularizer=tf.keras.regularizers.l2(reg)),
    ])

    return model


def get_deep_cnn(input_shape, num_classes=10, dropout=0.25, reg=0.01):
    """
    Get Shallow CNN Model
    Input:
        input_shape: input shape
        num_classes: num of classes
        activate: activate function
    Return:
        model
    """
    model = keras.Sequential([
        keras.Input(shape=input_shape),

        Conv2D(32, kernel_size=(3, 3)),
        BatchNormalization(),
        Activation(activations.relu),
        MaxPooling2D(pool_size=(2, 2)),

        Conv2D(64, kernel_size=(3, 3)),
        BatchNormalization(),
        Activation(activations.relu),
        MaxPooling2D(pool_size=(2, 2)),

        Conv2D(128, kernel_size=(3, 3)),
        BatchNormalization(),
        Activation(activations.relu),
        MaxPooling2D(pool_size=(2, 2)),

        Flatten(),
        Dropout(dropout),
        Dense(256, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(reg)),
        Dropout(dropout),
        Dense(num_classes, activation="softmax", kernel_regularizer=tf.keras.regularizers.l2(reg)),
    ])

    return model
