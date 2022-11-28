from sklearn.metrics import confusion_matrix
import pickle
from tensorflow import keras
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sn
import numpy as np
from utils import *


# SAVE_PATH = "./models/MLP_no_trail1.h5"
SAVE_PATH = "./models/MLP_Mixup_trail1.h5"
model = keras.models.load_model(SAVE_PATH)

FILE_PATH = "./digits4000_txt"
data_dict = get_data(FILE_PATH)
trail_1, trail_2 = pre_process_MLP(data_dict)

test_x = trail_2["test_x"]
test_y = trail_2["test_y"]

GENRES_MAP = {
    0: '0',
    1: '1',
    2: '2',
    3: '3',
    4: '4',
    5: '5',
    6: '6',
    7: '7',
    8: '8',
    9: '9'
    }

y_prediction = model.predict(test_x)
y_prediction = np.argmax(y_prediction, axis=1)
test_y = np.argmax(test_y, axis=1)
result = confusion_matrix(test_y, y_prediction , normalize='pred')

df_cm = pd.DataFrame(result, index = [GENRES_MAP[i] for i in range(10)],
                  columns = [GENRES_MAP[i]  for i in range(10)])
plt.figure(figsize = (10,7))
sn.heatmap(df_cm, annot=True)
plt.show()