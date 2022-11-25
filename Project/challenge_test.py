"""
Challenge MNIST
"""
import joblib
import os 
import numpy as np

from tensorflow.keras.utils import to_categorical
from keras.models import load_model


def get_challenge_data(file_path):
    """
    Get Challenge Data
    Input:
        file_path: file path
    Return:
        data_dict_ML:
            test_x: test input  (num_sampels, 784)
            test_y: test ouput  (num_sampels,)
        data_dict_MLP:
            test_x: test input    (num_sampels, 784)
            test_y: test ouput    (num_sampels, 10)
        data_dict_CNN:
            test_x: test input    (num_sampels, 28, 28, 1)
            test_y: test ouput    (num_sampels, 10)
    """
    NUM_CLASS = 10
    X_path = f"{file_path}/cdigits_digits_vec.txt"
    Y_path = f"{file_path}/cdigits_digits_labels.txt"
    X = np.loadtxt(X_path).astype(np.float32)
    X /= 255
    Y = np.loadtxt(Y_path)
    Y_encoding = to_categorical(Y, NUM_CLASS)
    X_CNN = np.reshape(X, [-1, 28, 28, 1])
    data_dict_ML = {
        "test_x": X,
        "test_y": Y
    }
    data_dict_MLP = {
        "test_x": X,
        "test_y": Y_encoding
    }
    data_dict_CNN = {
        "test_x": X_CNN,
        "test_y": Y_encoding
    }

    data_dict = {
        "data_dict_ML": data_dict_ML,
        "data_dict_MLP": data_dict_MLP,
        "data_dict_CNN": data_dict_CNN
    }

    return data_dict


if __name__ == "__main__":
    DATA_PATH = "./challenge"
    data_dict = get_challenge_data(DATA_PATH)
    print(f"Finish loading data")
    MODEL_PATH = "./models"
    g = os.walk(MODEL_PATH)
    res_dict = {}
    for path, _, file_list in g:  
        for file_name in file_list:
            path_name = os.path.join(path, file_name)
            if file_name.endswith(".pkl"):
                # ML model
                test_x = data_dict["data_dict_ML"]["test_x"]
                test_y = data_dict["data_dict_ML"]["test_y"]
                model = joblib.load(path_name)
                score = model.score(test_x, test_y)
                
                split_index = file_name.index(".")
                model_name = file_name[:split_index]
                score = float(score)
                res_dict[model_name] = score

            elif file_name.startswith("MLP"):
                test_x = data_dict["data_dict_MLP"]["test_x"]
                test_y = data_dict["data_dict_MLP"]["test_y"]
                model = load_model(path_name)
                score = model.evaluate(test_x, test_y, verbose=0)[1]

                split_index = file_name.index(".")
                model_name = file_name[:split_index]
                score = float(score)
                res_dict[model_name] = score
            elif file_name.startswith("CNN"):
                test_x = data_dict["data_dict_CNN"]["test_x"]
                test_y = data_dict["data_dict_CNN"]["test_y"]
                model = load_model(path_name)
                score = model.evaluate(test_x, test_y, verbose=0)[1]
                
                split_index = file_name.index(".")
                model_name = file_name[:split_index]
                score = float(score)
                res_dict[model_name] = score
        
        sorted = sorted(res_dict.items(), key=lambda x:x[1], reverse=False)
        for item in sorted:
            print(f"Model: {item[0]} & {item[1]:.4f} \\\\")