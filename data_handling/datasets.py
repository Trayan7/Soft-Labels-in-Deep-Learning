import cv2
import numpy as np
import tensorflow as tf
from keras.utils import to_categorical
from sklearn.utils import shuffle

from data_handling.json_creator import AnnotationJson


def sample_split(x, y, train_set_size):
    '''
    Splits the array likes x and y at train_set_size.
    Returns:
        Training x, y followed by testing x, y
    '''
    test_x = x[train_set_size:]
    train_x = x[:train_set_size]
    test_y = y[train_set_size:]
    train_y = y[:train_set_size]
    return train_x, train_y, test_x, test_y

def sample_split_semi(x, y, train_set_size, test_set_size):
    '''
    Splits the array likes x and y at train_set_size and test_set_size.
    Returns:
        Training x, y followed by testing x, y and unlabeled x
    '''
    train_set_index = test_set_size + train_set_size
    test_x = x[:test_set_size]
    train_x = x[test_set_size:train_set_index]
    test_y = y[:test_set_size]
    train_y = y[test_set_size:train_set_index]
    unlabeled_x = x[train_set_index:]
    return train_x, train_y, test_x, test_y, unlabeled_x

def load_cifar10h_dataset():
    '''
    Loads the CIFAR-10 test set and replaces the labels with the soft labels.
    Returns:
        Array of images normalized to [0,1],
        followed by the array of soft labels.
        Each label is an array of class probabilities. 
    '''
    (_, _), (set_x, _) = tf.keras.datasets.cifar10.load_data()
    set_y = np.load('tables/cifar10h-probs.npy')
    #set_x, set_y = shuffle(set_x, set_y)
    set_x.astype(np.float32)
    set_x = set_x / 255.0
    return set_x, set_y

def load_cifar10h_split_dataset(train_set_size):
    '''
    Loads the CIFAR-10 test set and replaces the labels with the soft labels.
    Then splits the dataset.
    Returns:
        Array of images normalized to [0,1],
        followed by the array of soft labels.
        for the training set. Then the test set
        in the same way.
        Each label is an array of class probabilities. 
    '''
    (_, _), (set_x, _) = tf.keras.datasets.cifar10.load_data()
    set_y = np.load('tables/cifar10h-probs.npy')
    #set_x, set_y = shuffle(set_x, set_y)
    set_x.astype(np.float32)
    set_x = set_x / 255.0
    return sample_split(set_x, set_y, train_set_size)

def load_cifar10h_split_dataset_hard(train_set_size):
    '''
    Loads the CIFAR-10 test set and splits it.
    Returns:
        Array of images normalized to [0,1],
        followed by the array of hard labels.
        for the training set. Then the test set
        in the same way. The hard labels are
        one-hot encoded.
    '''
    (_, _), (set_x, set_y) = tf.keras.datasets.cifar10.load_data()
    set_y = to_categorical(set_y)
    #set_x, set_y = shuffle(set_x, set_y)
    set_x.astype(np.float32)
    set_x = set_x / 255.0
    return sample_split(set_x, set_y, train_set_size)

def load_cifar10h_semi(test_set_size, train_set_size):
    '''
    Loads the CIFAR-10 test set and replaces the labels with the soft labels.
    Then splits them in training, testing, and unlabeled sets.
    Returns:
        Array of images normalized to [0,1],
        followed by the array of soft labels.
        Then the test set in the same way.
        Lastly an array of the unlabeled images.
        Each label is an array of class probabilities. 
    '''
    (_, _), (set_x, _) = tf.keras.datasets.cifar10.load_data()
    set_y = np.load('tables/cifar10h-probs.npy')
    set_x, set_y = shuffle(set_x, set_y)
    train_set_size = test_set_size + train_set_size

    return sample_split_semi(set_x, set_y, train_set_size, test_set_size)

def load_cifar10h_k_fold(k):
    '''
    Loads the CIFAR-10 test set and replaces the labels with the soft labels.
    Then creates k equally sized folds.
    Returns:
        Two arrays with the first holding all
        k image sets. The second array holds
        the corresponding label sets.
        Each image is normalized to [0,1].
        Each label is an array of class probabilities. 
    '''
    (_, _), (set_x, _) = tf.keras.datasets.cifar10.load_data()
    set_y = np.load('tables/cifar10h-probs.npy')
    set_x, set_y = shuffle(set_x, set_y)
    set_x.astype(np.float32)
    set_x = set_x / 255.0
    val = 1.0 / k
    n = round(len(set_x) * val)
    sets_x = []
    sets_y = []
    for i in range(k-1):
        cur_n = n * (i + 1)
        prev_n = n * i
        cur_set_x = set_x[prev_n:cur_n]
        cur_set_y = set_y[prev_n:cur_n]
        sets_x.append(cur_set_x)
        sets_y.append(cur_set_y)
    cur_n = n*(k - 1)
    cur_set_x = set_x[cur_n:]
    cur_set_y = set_y[cur_n:]
    sets_x.append(cur_set_x)
    sets_y.append(cur_set_y)
    return sets_x, sets_y

def load_plankton_split(train_set_size):
    '''
    Loads the plankton set after the duplicate check and
    splits them into train and test sets.
    Returns:
        Array of training images normalized to [0,1],
        followed by the array of soft labels.
        This is then repeated for the test set.
        Each label is an array of class probabilities. 
    '''
    images = np.load('./tables/no_duplicates_plankton_x2.npy')
    labels = np.load('./tables/no_duplicates_plankton_y2.npy')
    return sample_split(images, labels, train_set_size)

def load_plankton_direct():
    '''
    Loads the plankton set directly using json_creator.py
    and rescales them to (64,64)
    Returns:
        Array of images followed by the array of soft labels.
        Each label is an array of class probabilities. 
    '''
    json = AnnotationJson.from_file('../Plankton/plankton_no_fit-s01@default/annotations.json')
    imgs, _, data = AnnotationJson.get_probability_data(json)
    images = list()
    for path in imgs:
        cur_img = cv2.imread('../Plankton/' + path, flags=cv2.IMREAD_GRAYSCALE)
        resize_img = cv2.resize(cur_img, dsize=(64, 64))
        images.append(resize_img)
    return images, data

def load_plankton_split_hard(train_set_size):
    '''
    Loads the plankton set after the duplicate check and
    splits them into train and test sets.
    Returns:
        Array of training images normalized to [0,1],
        followed by the array of hard labels.
        This is then repeated for the test set.
        Each label is one-hot encoded.
    '''
    images = np.load('tables/no_duplicates_plankton_x2.npy')
    labels = np.load('tables/no_duplicates_plankton_y2.npy')
    labels = np.argmax(labels, axis=1)
    labels = np.array(tf.one_hot(labels, 10))
    return sample_split(images, labels, train_set_size)

def load_plankton_semi(test_set_size, train_set_size):
    '''
    Loads the plankton set after the duplicate check and
    splits them into train, test and unlabeled sets.
    Returns:
        Array of training images normalized to [0,1],
        followed by the array of soft labels.
        This is then repeated for the test set.
        Lastly an array of the unlabeled images.
        Each label is an array of class probabilities. 
    '''
    set_x = np.load('tables/no_duplicates_plankton_x2.npy')
    set_y = np.load('tables/no_duplicates_plankton_y2.npy')
    train_set_size = test_set_size + train_set_size

    return sample_split_semi(set_x, set_y, train_set_size, test_set_size)
