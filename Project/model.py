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
    filter_num_1 = hp.Choice(name="filter_num_1", values= [32, 64])
    filter_num_2 = hp.Choice(name="filter_num_2", values= [32, 64])

    dropouts = hp.Float(name="dropouts", min_value=0.1, max_value=0.2, step=0.1)
    regs = hp.Float(name="regs", min_value=0.01, max_value=0.02, step=0.01)
    lrs = hp.Float(name="lrs", min_value=0.001, max_value=0.002, step=0.001)

    model = keras.Sequential([
        keras.Input(shape=(28, 28, 1)),

        Conv2D(filter_num_1, kernel_size=(3, 3)),
        BatchNormalization(),
        Activation(activations.relu),
        MaxPooling2D(pool_size=(2, 2)),

        Conv2D(filter_num_2, kernel_size=(3, 3)),
        BatchNormalization(),
        Activation(activations.relu),
        MaxPooling2D(pool_size=(2, 2)),

        Flatten(),
        Dropout(dropouts),
        Dense(10, activation="softmax", kernel_regularizer=tf.keras.regularizers.l2(regs)),
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
    filter_num_1 = hp.Choice(name="filter_num_1", values= [32, 64])
    filter_num_2 = hp.Choice(name="filter_num_2", values= [32, 64])
    filter_num_3 = hp.Choice(name="filter_num_3", values= [32, 64])

    hidden = hp.Choice(name="hidden", values= [128, 256])

    dropouts = hp.Float(name="dropouts", min_value=0.1, max_value=0.2, step=0.1)
    regs = hp.Float(name="regs", min_value=0.01, max_value=0.02, step=0.01)
    lrs = hp.Float(name="lrs", min_value=0.001, max_value=0.002, step=0.001)

    model = keras.Sequential([
        keras.Input(shape=(28, 28, 1)),

        Conv2D(filter_num_1, kernel_size=(3, 3)),
        BatchNormalization(),
        Activation(activations.relu),
        MaxPooling2D(pool_size=(2, 2)),

        Conv2D(filter_num_2, kernel_size=(3, 3)),
        BatchNormalization(),
        Activation(activations.relu),
        MaxPooling2D(pool_size=(2, 2)),

        Conv2D(filter_num_3, kernel_size=(3, 3)),
        BatchNormalization(),
        Activation(activations.relu),
        MaxPooling2D(pool_size=(2, 2)),

        Flatten(),
        Dropout(dropouts),
        Dense(hidden, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(regs)),
        Dropout(dropouts),
        Dense(10, activation="softmax", kernel_regularizer=tf.keras.regularizers.l2(regs)),
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
    hidden_1 = hp.Choice(name="hidden_1", values= [256, 512])
    hidden_2 = hp.Choice(name="hidden_2", values= [256, 512])
    hidden_3 = hp.Choice(name="hidden_3", values= [256, 512])

    dropouts = hp.Float(name="dropouts", min_value=0.1, max_value=0.2, step=0.1)
    regs = hp.Float(name="regs", min_value=0.01, max_value=0.02, step=0.01)
    lrs = hp.Float(name="lrs", min_value=0.001, max_value=0.002, step=0.001)

    model = keras.Sequential([
        keras.Input(shape=(784)),

        Dense(hidden_1, kernel_regularizer=tf.keras.regularizers.l2(regs)),
        BatchNormalization(),
        Activation(activations.relu),

        Dense(hidden_2, kernel_regularizer=tf.keras.regularizers.l2(regs)),
        BatchNormalization(),
        Activation(activations.relu),

        Dense(hidden_3, kernel_regularizer=tf.keras.regularizers.l2(regs)),
        BatchNormalization(),
        Activation(activations.relu),
 
        Dropout(dropouts),
        Dense(10, activation="softmax", kernel_regularizer=tf.keras.regularizers.l2(regs)),
    ])

    model.compile(loss=tf.keras.losses.categorical_crossentropy,
                        optimizer=tf.keras.optimizers.Adam(lrs),
                        metrics=['accuracy'])

    return model