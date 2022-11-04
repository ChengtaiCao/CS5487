import pdb

from utils import *


FILE_PATH = "./digits4000_txt"


if __name__ == "__main__":
    train_ds_mu, test_x_1, test_y_1 = get_mix_data(FILE_PATH)
    input_shape = test_x_1.shape[1:]
    num_classes = test_y_1.shape[-1]
    model = get_model2(input_shape, num_classes)
    model.summary()
    batch_size = 128
    epochs = 200
    model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])
    model.fit(train_ds_mu, epochs=epochs)
    score = model.evaluate(test_x_1, test_y_1, verbose=0)
    print("Test loss:", score[0])
    print("Test accuracy:", score[1])
    print("Well Done!")