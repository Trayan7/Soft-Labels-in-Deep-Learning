import numpy as np
from numpy import random
from sklearn.utils import shuffle
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator

def load_dataset():
    (_, _), (set_x, _) = tf.keras.datasets.cifar10.load_data()
    set_y = np.load("cifar10h-probs.npy")
    set_x, set_y = shuffle(set_x, set_y)
    set_x.astype(np.float32)
    set_x = set_x / 255.0
    return set_x, set_y

def load_split_dataset(val):
    (_, _), (set_x, _) = tf.keras.datasets.cifar10.load_data()
    set_y = np.load("cifar10h-probs.npy")
    set_x, set_y = shuffle(set_x, set_y)
    set_x.astype(np.float32)
    set_x = set_x / 255.0
    i_test = round(len(set_x) * val)
    test_x = set_x[:i_test]
    train_x = set_x[i_test:]
    test_y = set_y[:i_test]
    train_y = set_y[i_test:]
    return train_x, train_y, test_x, test_y

def naive_mix_up(x, y, rounds, alpha):
    set_x = x
    set_y = y
    print(x.dtype)
    print(y.dtype)
    for n in range(rounds):
        print(n)
        x2, y2 = shuffle(x, y)
        set_x = np.concatenate([set_x, x * alpha + x2 * (1. - alpha)])
        set_y = np.concatenate([set_y, y * alpha + y2 * (1. - alpha)])
    return set_x, set_y

def naive_mix_up2(x, y, alpha):
    x2, y2 = shuffle(x, y)
    set_x = x * alpha + x2 * (1. - alpha)
    set_y = y * alpha + y2 * (1. - alpha)
    return set_x, set_y

def mixup_gen(train_x, train_y, batch_size, datagen, alpha):
    i = 0
    while True:
        if i%32==0:
            set_x, set_y = naive_mix_up2(train_x, train_y, alpha)
        i += 1
        gen = datagen.flow(set_x, set_y, batch_size)
        (x, y) = gen.next()
        yield (x, y)
    return

def add_noise(img):
    INTENSITY = 0.2
    randomize = INTENSITY * random.random()
    noise = np.random.normal(0, randomize, img.shape)
    img += noise
    np.clip(img, 0., 1.)
    return img

def load_datagen():
    datagen = ImageDataGenerator(
        rotation_range=10,
        width_shift_range=0.1,
        height_shift_range=0.1,
        shear_range=0.1,
        zoom_range=0.1,
        horizontal_flip=True)
        #preprocessing_function=add_noise)
    return datagen
'''
# aug 3
def load_datagen():
    datagen = ImageDataGenerator(
        rotation_range=10,
        width_shift_range=0.1,
        height_shift_range=0.1,
        shear_range=0.1,
        zoom_range=0.1,
        horizontal_flip=True)
    return datagen
'''
'''
# aug 1
def load_datagen():
    datagen = ImageDataGenerator(
        rotation_range=40,
        width_shift_range=0.1,
        height_shift_range=0.1,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True)
    return datagen
'''
'''
# "weaker" setting
def load_datagen():
    datagen = ImageDataGenerator(
        rotation_range=20,
        width_shift_range=0.1,
        height_shift_range=0.1,
        shear_range=0.1,
        zoom_range=0.1,
        horizontal_flip=True)
    return datagen
'''
'''
# First parameters
def load_datagen():
    datagen = ImageDataGenerator(
        rotation_range=40,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True)
    return datagen
'''

def load_datagen_val():
    datagen = ImageDataGenerator()
    return datagen

def augment(image):
    #image = tf.image.random_brightness(image, max_delta=0.1)
    #image = tf.image.random_contrast(image, lower=0.1, upper=0.2)
    #image = tf.image.random_flip_left_right(image)
    #image = tf.image.random_saturation(image, lower=0.1, upper=0.2)
    image = tf.keras.preprocessing.image.random_rotation(image, 20)
    image = tf.keras.preprocessing.image.random_shear(image, 10)
    image = tf.keras.preprocessing.image.random_shift(image, 0.1, 0.1)
    image = tf.keras.preprocessing.image.random_zoom(image, (0.1, 0.1))
    return image

def get_data_aug(images, labels, count):
    gen = load_datagen()
    augmented = []
    for x in images:
        aug_iter = gen.flow(x)
        aug_images = [next(aug_iter)[0] for i in range(count)]
        augmented += aug_images
    np.repeat(labels, count)
    return images, labels

def load_dataset_semi_supervised():
    (_, _), (set_x, _) = tf.keras.datasets.cifar10.load_data()
    set_y = np.load("cifar10h-probs.npy")
    set_x, set_y = shuffle(set_x, set_y)
    i_test = round(len(set_x) * 0.25)
    i_train = i_test + round(len(set_x) * 0.01)
    i_unlabeled = i_train + round(len(set_x) * 0.74)

    test_x = set_x[:i_test]
    train_x = set_x[i_test:i_train]
    test_y = set_y[:i_test]
    train_y = set_y[i_test:i_train]
    unlabeled_x = set_x[i_train:i_unlabeled]

    return train_x, train_y, test_x, test_y, unlabeled_x

def load_dataset_semi_supervised_augmented():
    (_, _), (set_x, _) = tf.keras.datasets.cifar10.load_data()
    set_y = np.load("cifar10h-probs.npy")
    set_x, set_y = shuffle(set_x, set_y)
    i_test = round(len(set_x) * 0.25)
    i_train = i_test + round(len(set_x) * 0.01)
    i_unlabeled = i_train + round(len(set_x) * 0.74)

    test_x = set_x[:i_test]
    train_x = set_x[i_test:i_train]
    test_y = set_y[:i_test]
    train_y = set_y[i_test:i_train]
    unlabeled_x = set_x[i_train:i_unlabeled]

    #datagen = ImageDataGenerator(horizontal_flip=True)
    #it = datagen.flow(samples, batch_size=1)    

    return train_x, train_y, test_x, test_y, unlabeled_x