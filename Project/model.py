"""
Implementation of Deep Learning Model
"""
import tensorflow as tf
import keras_tuner as kt
import keras
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Activation
from tensorflow.keras.layers import Dropout, Dense, BatchNormalization
from tensorflow.keras import activations


def get_shallow_cnn(hp):
    """
    Get Shallow CNN Model
    Input:
        hp: keras_tuner.HyperParameters
    Return:
        model
    """
    filter_1 = hp.Choice(name="filter_1", values= [16, 32, 64, 128])
    filter_2 = hp.Choice(name="filter_2", values= [32, 64, 128, 256])
    lrs = hp.Choice(name="lrs", values=[0.01, 0.1, 0.2])

    model = keras.Sequential([
        keras.Input(shape=(28, 28, 1)),

        Conv2D(filter_1, kernel_size=(3, 3)),
        BatchNormalization(),
        Activation(activations.relu),
        MaxPooling2D(pool_size=(2, 2)),

        Conv2D(filter_2, kernel_size=(3, 3)),
        BatchNormalization(),
        Activation(activations.relu),
        MaxPooling2D(pool_size=(2, 2)),

        Flatten(),
        Dropout(0.2),
        Dense(10, activation="softmax"),
    ])

    model.compile(loss=tf.keras.losses.categorical_crossentropy,
                        optimizer=tf.keras.optimizers.Adam(lrs),
                        metrics=['accuracy'])

    return model


def get_deep_cnn(hp):
    """
    Get Deep CNN Model
    Input:
        hp: keras_tuner.HyperParameters
    Return:
        model
    """
    filter_1 = hp.Choice(name="filter_1", values= [32, 64, 128])
    filter_2 = hp.Choice(name="filter_2", values= [64, 128, 256])
    filter_3 = hp.Choice(name="filter_3", values= [128, 256, 512])
    hidden = hp.Choice(name="hidden", values= [512, 256, 1024])
    lrs = hp.Choice(name="lrs", values=[0.01, 0.1, 0.2])

    model = keras.Sequential([
        keras.Input(shape=(28, 28, 1)),

        Conv2D(filter_1, kernel_size=(3, 3)),
        BatchNormalization(),
        Activation(activations.relu),
        MaxPooling2D(pool_size=(2, 2)),

        Conv2D(filter_2, kernel_size=(3, 3)),
        BatchNormalization(),
        Activation(activations.relu),
        MaxPooling2D(pool_size=(2, 2)),

        Conv2D(filter_3, kernel_size=(3, 3)),
        BatchNormalization(),
        Activation(activations.relu),
        MaxPooling2D(pool_size=(2, 2)),

        Flatten(),
        Dropout(0.2),
        Dense(hidden, activation='relu'),
        Dropout(0.2),
        Dense(10, activation="softmax"),
    ])

    model.compile(loss=tf.keras.losses.categorical_crossentropy,
                        optimizer=tf.keras.optimizers.Adam(lrs),
                        metrics=['accuracy'])
    return model


def get_MLP(hp):
    """
    Get MLP Model
    Input:
        hp: keras_tuner.HyperParameters
    Return:
        model
    """
    hidden_1 = hp.Choice(name="hidden_1", values= [512, 1024])
    hidden_2 = hp.Choice(name="hidden_2", values= [256, 128])
    hidden_3 = hp.Choice(name="hidden_3", values= [32, 64])

    lrs = hp.Choice(name="lrs", values=[0.0001, 0.001, 0.01, 0.1])

    model = keras.Sequential([
        keras.Input(shape=(784)),

        Dense(hidden_1),
        BatchNormalization(),
        Activation(activations.relu),

        Dense(hidden_2),
        BatchNormalization(),
        Activation(activations.relu),

        Dense(hidden_3),
        BatchNormalization(),
        Activation(activations.relu),
 
        Dropout(0.2),
        Dense(10, activation="softmax"),
    ])

    model.compile(loss=tf.keras.losses.categorical_crossentropy,
                        optimizer=tf.keras.optimizers.Adam(lrs),
                        metrics=['accuracy'])

    return model