"""
Implementation of Main
"""
import argparse

from utils import *
from model import *

from tensorflow.keras.callbacks import ReduceLROnPlateau


if __name__ == "__main__":
    FILE_PATH = "./digits4000_txt"
    # parse config
    parser = argparse.ArgumentParser()
    parser.add_argument('--aug', default=None, 
                        choices=[None, "Mixup", "RZS", "both"],
                        help='What kind of data augmentation.')

    parser.add_argument('--model', default="Shallow", 
                        choices=["Shallow", "Deep", "MLP"],
                        help='Which model.')           
    args = parser.parse_args()

    # read data
    data_dict = get_data(FILE_PATH)
    # pre-process data
    trial_1, trial_2 = pre_process_CNN(data_dict)
    # get model
    input_shape = trial_1["train_x"][0].shape
    if args.model == "Shallow":
        model = get_shallow_cnn(input_shape)
    elif args.model == "Deep":
        model = get_deep_cnn(input_shape)
    # model config
    model.compile(loss=tf.keras.losses.categorical_crossentropy,
                optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
                metrics=['accuracy'])
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
    # train config
    BATCH_SIZE = 32
    train_ds = (
            tf.data.Dataset.from_tensor_slices((trial_1["train_x"], trial_1["train_y"]))
            .batch(BATCH_SIZE)
        )
    
    validation_ds = (
            tf.data.Dataset.from_tensor_slices((trial_1["validation_x"], trial_1["validation_y"]))
            .batch(BATCH_SIZE)
        )
    
    history = model.fit(
        train_ds,
        validation_data=validation_ds,
        epochs=100,
        verbose=1,
        callbacks=[reduceLROnPlat])
    
    score = model.evaluate(trial_1["test_x"], trial_1["test_y"], verbose=0)
    print(f"CNN mean accuracy: {score[1]:.4f}")
