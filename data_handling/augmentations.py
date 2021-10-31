import random

import numpy as np
import tensorflow as tf
from numpy import random
from sklearn.utils import shuffle
from tensorflow.keras.preprocessing.image import ImageDataGenerator

from data_handling.datasets import load_plankton_split


def naive_mix_up(x, y, alpha):
    '''
    Applies MixUp randomly to samples.
    Parameters:
        x: Samples
        y: Labels
        alpha: How much the original is weighted.
    '''
    x2, y2 = shuffle(x, y)
    set_x = x * alpha + x2 * (1. - alpha)
    set_y = y * alpha + y2 * (1. - alpha)
    return set_x, set_y

def mixup_gen(train_x, train_y, batch_size, datagen, alpha):
    '''
    Applies MixUp before a datagenerator to yield
    an infinite datagenerator.
    Parameters:
        train_x: Array like of training samples
        train_y: Array like of training sample labels
        batch_size: Batch size for datagenerator to yield.
        datagen: Datagenerator applied befor after MixUp.
        alpha: How much the original samples are weighted.
    Returns:
        Infinite sequence of augmented samples.
    '''
    i = 0
    while True:
        if i % batch_size == 0:
            set_x, set_y = naive_mix_up(train_x, train_y, alpha)
        i += 1
        gen = datagen.flow(set_x, set_y, batch_size)
        (x, y) = gen.next()
        yield (x, y)
    return

def add_noise(img):
    '''
    Adds normalized noise to an image.
    Parameters:
        img: The image in question normalized to [0,1].
    '''
    INTENSITY = 0.1 # 0.2 
    randomize = INTENSITY * random.random()
    noise = np.random.normal(0, randomize, img.shape)
    img += noise
    np.clip(img, 0., 1.)
    return img

def load_datagen():
    '''
    Setups and returns the datagenerator for augmentations.
    '''
    datagen = ImageDataGenerator(
        rotation_range=10,
        width_shift_range=0.1,
        height_shift_range=0.1,
        shear_range=0.1,
        zoom_range=0.1,
        horizontal_flip=True,
        preprocessing_function=add_noise)
    return datagen

def augment(image):
    '''
    Applies augmentations to a single sample image.
    '''
    #image = tf.image.random_brightness(image, max_delta=0.1)
    #image = tf.image.random_contrast(image, lower=0.1, upper=0.2)
    #image = tf.image.random_flip_left_right(image)
    #image = tf.image.random_saturation(image, lower=0.1, upper=0.2)
    image = tf.keras.preprocessing.image.random_rotation(image, 20)
    image = tf.keras.preprocessing.image.random_shear(image, 10)
    image = tf.keras.preprocessing.image.random_shift(image, 0.1, 0.1)
    image = tf.keras.preprocessing.image.random_zoom(image, (0.1, 0.1))
    return image

def pseudo_label_gen(train_x, train_y, unlabeled_x, model, pseudo_count, datagen, batch_size):
    '''
    Applies pseudo labeling, before applying a datagenerator.
    Parameters:
        train_x: Array like of labeled samples
        train_y: Array like of labeled sample labels
        unlabeled_x: Array like of unlabeled samples
        model: The model being trained.
        pseudo_count: Number of pseudo labeled samples mixed into the set.
        datagen: Datagenerator applied after pseudo labeling.
        batch_size: Size of the batch for the datagenerator.
    Returns:
        Sequential training set feeder of batch size
    '''
    pseudo_x = shuffle(unlabeled_x.copy())
    pseudo_x = pseudo_x[:pseudo_count]
    pseudo_y_tensor = model(pseudo_x)
    pseudo_y = []
    for elem in pseudo_y_tensor:
        elem = elem - np.amin(elem)
        pseudo_y.append(elem / np.sum(elem))
    pseudo_y = np.array(pseudo_y)
    train_x, train_y = shuffle(np.concatenate([train_x, pseudo_x]), np.concatenate([train_y, pseudo_y]))
    return datagen.flow(train_x, train_y, batch_size)


