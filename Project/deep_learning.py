"""
Implementation of Main
"""
import argparse
import pdb

from utils import *
from model import *

import keras_tuner as kt
from tensorflow.keras.callbacks import ReduceLROnPlateau
from augmentation import *


reduceLROnPlat = ReduceLROnPlateau(
    monitor='val_loss', 
    factor=0.9,
    patience=3,
    verbose=1,
    mode='min',
    min_delta=0.0001,
    cooldown=2,
    min_lr=0.00001
)


def shallow_function(data_dict, str_txt):
    """
    Shallow CNN
    Input:
        data_dict: data dict
        str_txt: print text
    """
    # pre-process data
    trail_1, trail_2 = pre_process_CNN(data_dict)
    strs = []
    scores = []

    trails = [trail_1, trail_2]
    for i in range(len(trails)):
        data = trails[i]
        # train config
        BATCH_SIZE = 32
        # search best hyper-parameter
        tuner = kt.Hyperband(get_shallow_cnn,
                    objective="val_accuracy",
                    max_epochs=200,
                    factor=3,
                    directory="HyperSearch",
                    project_name=f"Shallow_HyperSearch_{i}")

        train_ds = (
                tf.data.Dataset.from_tensor_slices((data["train_x"], data["train_y"]))
            )
        
        validation_ds = (
                tf.data.Dataset.from_tensor_slices((data["validation_x"], data["validation_y"]))
                .batch(BATCH_SIZE)
            )
        
        if str_txt == "FRZ" or str_txt == "Both":
            train_ds = FRZ_aug(train_ds, BATCH_SIZE, str_txt)
        if str_txt == "Mixup" or str_txt == "Both":
            train_ds = Mixup_aug(train_ds, BATCH_SIZE)
        
        tuner.search(train_ds, epochs=2, validation_data=validation_ds, callbacks=[reduceLROnPlat])
        best_hps=tuner.get_best_hyperparameters(num_trials=1)[0]
        best_dropout = best_hps.get("dropouts")
        best_reg = best_hps.get("regs")
        best_lr = best_hps.get("lrs")

        # retrain_model
        model = tuner.hypermodel.build(best_hps)
        history = model.fit(
            train_ds,
            validation_data=validation_ds,
            epochs=200,
            verbose=1,
            callbacks=[reduceLROnPlat])
    
        score = model.evaluate(data["test_x"], data["test_y"], verbose=0)
        str = f"-- Trail {i + 1} --- \n"
        str += f"the best_dropout is {best_dropout} \n"
        str += f"the best_reg is {best_reg} \n"
        str += f"the best_lr is {best_lr}"
        strs.append(str)
        scores.append(score[1])
    for i in range(2):
        print(strs[i])
        print(scores[i])
    print(f"Shallow CNN with {str_txt} augmentation(s) is {np.mean(scores): .4f}")


def deep_function(data_dict, str_txt):
    """
    Deep CNN
    Input:
        data_dict: data dict
        str_txt: print text
    """
    # pre-process data
    trail_1, trail_2 = pre_process_CNN(data_dict)
    strs = []
    scores = []

    trails = [trail_1, trail_2]
    for i in range(len(trails)):
        data = trails[i]
        # train config
        BATCH_SIZE = 32
        # search best hyper-parameter
        tuner = kt.Hyperband(get_deep_cnn,
                    objective="val_accuracy",
                    max_epochs=120,
                    factor=3,
                    directory="HyperSearch",
                    project_name=f"Deep_HyperSearch_{i}")

        train_ds = (
                tf.data.Dataset.from_tensor_slices((data["train_x"], data["train_y"]))
                .batch(BATCH_SIZE)
            )
        
        validation_ds = (
                tf.data.Dataset.from_tensor_slices((data["validation_x"], data["validation_y"]))
                .batch(BATCH_SIZE)
            )
        
        tuner.search(train_ds, epochs=2, validation_data=validation_ds, callbacks=[reduceLROnPlat])
        best_hps=tuner.get_best_hyperparameters(num_trials=1)[0]
        best_dropout = best_hps.get("dropouts")
        best_reg = best_hps.get("regs")
        best_lr = best_hps.get("lrs")

        # retrain_model
        model = tuner.hypermodel.build(best_hps)
        history = model.fit(
            train_ds,
            validation_data=validation_ds,
            epochs=100,
            verbose=1,
            callbacks=[reduceLROnPlat])
    
        score = model.evaluate(data["test_x"], data["test_y"], verbose=0)
        str = f"-- Trail {i + 1} --- \n"
        str += f"the best_dropout is {best_dropout} \n"
        str += f"the best_reg is {best_reg} \n"
        str += f"the best_lr is {best_lr}"
        strs.append(str)
        scores.append(score[1])
    for i in range(2):
        print(strs[i])
        print(scores[i])
    print(f"Deep CNN with {str_txt} augmentation(s) is {np.mean(scores): .4f}")


def mlp_function(data_dict, str_txt):
    """
    MLP
    Input:
        data_dict: data dict
        str_txt: print text
    """
    # pre-process data
    trail_1, trail_2 = pre_process_MLP(data_dict)
    strs = []
    scores = []

    trails = [trail_1, trail_2]
    for i in range(len(trails)):
        data = trails[i]
        # train config
        BATCH_SIZE = 32
        # search best hyper-parameter
        tuner = kt.Hyperband(get_MLP,
                    objective="val_accuracy",
                    max_epochs=120,
                    factor=3,
                    directory="HyperSearch",
                    project_name=f"MLP_HyperSearch_{str_txt}_{i}")

        train_ds = (
                tf.data.Dataset.from_tensor_slices((data["train_x"], data["train_y"]))
                .batch(BATCH_SIZE)
            )
        
        validation_ds = (
                tf.data.Dataset.from_tensor_slices((data["validation_x"], data["validation_y"]))
                .batch(BATCH_SIZE)
            )
        
        tuner.search(train_ds, epochs=2, validation_data=validation_ds, callbacks=[reduceLROnPlat])
        best_hps=tuner.get_best_hyperparameters(num_trials=1)[0]
        hidden_1 = best_hps.get("hidden_1")
        hidden_2 = best_hps.get("hidden_2")
        hidden_3 = best_hps.get("hidden_3")
        best_lr = best_hps.get("lrs")

        # retrain_model
        model = tuner.hypermodel.build(best_hps)
        history = model.fit(
            train_ds,
            validation_data=validation_ds,
            epochs=100,
            verbose=1,
            callbacks=[reduceLROnPlat])
    
        score = model.evaluate(data["test_x"], data["test_y"], verbose=0)
        str = f"-- Trail {i + 1} --- \n"
        str += f"the hidden_1 is {hidden_1} \n"
        str += f"the hidden_2 is {hidden_2} \n"
        str += f"the hidden_3 is {hidden_3} \n"
        str += f"the best_lr is {best_lr}"
        strs.append(str)
        scores.append(score[1])
    for i in range(2):
        print(strs[i])
        print(scores[i])
    print(f"MAP with {str_txt} augmentation(s) is {np.mean(scores): .4f}")


if __name__ == "__main__":
    FILE_PATH = "./digits4000_txt"
    # parse config
    parser = argparse.ArgumentParser()
    parser.add_argument('--aug', default="no", 
                        choices=["no", "Mixup", "FRZ", "Both"],
                        help='What kind of data augmentation.')

    parser.add_argument('--model', default="Shallow", 
                        choices=["Shallow", "Deep", "MLP"],
                        help='Which model.')           
    args = parser.parse_args()

    model_str = args.model
    aug_str = args.aug
    # read data
    data_dict = get_data(FILE_PATH)

    if model_str == "Shallow":
        # Shallow CNN
        shallow_function(data_dict, aug_str)
    elif model_str == "Deep":
        # Deep CNN
        deep_function(data_dict, aug_str)
    elif model_str == "MLP":
        # MLP
        mlp_function(data_dict, aug_str)
    