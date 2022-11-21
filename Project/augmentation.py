"""
Implementation of Augmentation
"""
import tensorflow as tf


AUTOTUNE = tf.data.AUTOTUNE


def FRZ_aug(train_ds, BATCH_SIZE, str_txt):
    """
    flip, rotation, zoom augmentation
    Parameters:
        train_ds: training data
        BATCH_SIZE: batch size
        str_txt: augmentation choice
    return:
        aug_ds: augmented data
    """
    augmentation_layer = tf.keras.Sequential([
        tf.keras.layers.experimental.preprocessing.RandomFlip(),
        tf.keras.layers.experimental.preprocessing.RandomRotation(0.3),
        tf.keras.layers.RandomZoom(height_factor=(-0.5, 0.5), width_factor=(-0.5, 0.5))
    ])
    aug_ds = train_ds.map(
        lambda x, y: (augmentation_layer(x, training=True), y), 
        num_parallel_calls=AUTOTUNE
    )
    if str_txt != "Both":
        aug_ds = aug_ds.batch(BATCH_SIZE)
    return aug_ds



def Mixup_aug(train_ds, BATCH_SIZE):
    """
    Mixup augmentation
    Parameters:
        train_ds: training data
        BATCH_SIZE: batch_size
    return:
        aug_ds: augmented data
    """
    train_ds_one = train_ds.shuffle(BATCH_SIZE * 100).batch(BATCH_SIZE)
    train_ds_two = train_ds.shuffle(BATCH_SIZE * 100).batch(BATCH_SIZE)
    train_ds = tf.data.Dataset.zip((train_ds_one, train_ds_two))
    aug_ds = train_ds.map(
            lambda ds_one, ds_two: mixup(ds_one, ds_two, alpha=0.2), num_parallel_calls=AUTOTUNE
        )
    return aug_ds


def sample_beta_distribution(size, concentration_0=0.2, concentration_1=0.2):
    """
    sample beta distribution
    Parameter:
        size: data_size
    return:
        gamma_sample: gamma with same size
    """
    gamma_1_sample = tf.random.gamma(shape=[size], alpha=concentration_1)
    gamma_2_sample = tf.random.gamma(shape=[size], alpha=concentration_0)
    gamma_sample = gamma_1_sample / (gamma_1_sample + gamma_2_sample)
    return gamma_sample


def mixup(ds_one, ds_two, alpha=0.2):
    """
    mixup
    Parameter:
        ds_one: data 1
        ds_two: data 2
        alpha: hyper-parameter
    return:
        images: mixup_ed images
        labels: mixup_ed labels
    """
    # Unpack two datasets
    images_one, labels_one = ds_one
    images_two, labels_two = ds_two
    batch_size = tf.shape(images_one)[0]

    # Sample lambda and reshape it to do the mixup
    l = sample_beta_distribution(batch_size, alpha, alpha)
    x_l = tf.reshape(l, (batch_size, 1, 1, 1))
    y_l = tf.reshape(l, (batch_size, 1))

    # Perform mixup on both images and labels by combining a pair of images/labels
    # (one from each dataset) into one image/label
    images = images_one * x_l + images_two * (1 - x_l)
    labels = labels_one * y_l + labels_two * (1 - y_l)
    return (images, labels)