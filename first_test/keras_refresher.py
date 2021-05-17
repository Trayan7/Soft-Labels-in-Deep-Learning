import sys

import numpy as np
import tensorflow as tf
import wandb
from keras.preprocessing.image import ImageDataGenerator
from matplotlib import pyplot
from sklearn.utils import shuffle
from tensorflow import keras
from tensorflow.keras import Input, Model, callbacks
from tensorflow.keras.applications.resnet_v2 import ResNet50V2
from tensorflow.keras.applications.nasnet import NASNetLarge
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.layers import (Conv2D, Dense, Dropout, Flatten,
                                     GlobalAveragePooling2D, MaxPooling2D)
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.regularizers import l2
from tensorflow.keras.utils import to_categorical
from wandb.keras import WandbCallback


def load_dataset():
    (_, _), (set_x, _) = tf.keras.datasets.cifar10.load_data()
    set_y = np.load("first_test\cifar10h-probs.npy")
    set_x, set_y = shuffle(set_x, set_y)
    for x in set_x:
        x = x / 255.0
    return set_x, set_y

def load_dataset_semi_supervised():
    (_, _), (set_x, _) = tf.keras.datasets.cifar10.load_data()
    set_y = np.load("first_test\cifar10h-probs.npy")
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
    set_y = np.load("first_test\cifar10h-probs.npy")
    set_x, set_y = shuffle(set_x, set_y)
    i_test = round(len(set_x) * 0.25)
    i_train = i_test + round(len(set_x) * 0.01)
    i_unlabeled = i_train + round(len(set_x) * 0.74)

    test_x = set_x[:i_test]
    train_x = set_x[i_test:i_train]
    test_y = set_y[:i_test]
    train_y = set_y[i_test:i_train]
    unlabeled_x = set_x[i_train:i_unlabeled]

    datagen = ImageDataGenerator(horizontal_flip=True)
    it = datagen.flow(samples, batch_size=1)    

    return train_x, train_y, test_x, test_y, unlabeled_x

def define_model():
    model = Sequential()
    model.add(Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same', kernel_regularizer=l2(0.001), input_shape=(32, 32, 3)))
    model.add(Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same', kernel_regularizer=l2(0.001)))
    model.add(MaxPooling2D((2, 2)))
    model.add(Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same', kernel_regularizer=l2(0.001)))
    model.add(Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same', kernel_regularizer=l2(0.001)))
    model.add(MaxPooling2D((2, 2)))
    model.add(Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same', kernel_regularizer=l2(0.001)))
    model.add(Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same', kernel_regularizer=l2(0.001)))
    model.add(MaxPooling2D((2, 2)))
    model.add(Flatten())
    model.add(Dense(128, activation='relu', kernel_initializer='he_uniform', kernel_regularizer=l2(0.001)))
    model.add(Dense(10, activation='softmax'))
    opt = keras.optimizers.Adam(learning_rate=0.001)
    model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy'])
    return model
 
def run_test():
    wandb.init(project='supervised')
    config = wandb.config
    config.learning_rate = 0.001
    config.batch_size = 256
    config.model = "ResNet50V2"
    config.dataset = "CIFAR10H"
    wandb.run.name = "ResNet50V2 MSLE 256 0.5 sig"
    wandb.run.save()
    set_x, set_y = load_dataset()
    #train_x, train_y, test_x, test_y, unlabeled_x = load_dataset_semi_supervised()
    #model = define_model()
    input_tensor = Input(shape=(32, 32, 3))
    base_model = ResNet50V2(include_top=False, weights=None, input_tensor=input_tensor, classes=10)
    #base_model = InceptionResNetV2(include_top=False, weights=None, input_tensor=input_tensor, classes=10)
    #base_model = NASNetLarge(include_top=False, weights=None, input_tensor=input_tensor, classes=10)
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dropout(0.5)(x)
    predictions = Dense(10, activation= 'sigmoid')(x)
    model = Model(inputs = base_model.input, outputs = predictions)
    model.compile(keras.optimizers.Adam(lr=config.learning_rate, beta_1=0.9, beta_2=0.999, epsilon=10e-8, decay=0.0, clipvalue=0.5), loss=tf.keras.losses.MeanSquaredLogarithmicError(), metrics=['accuracy', 'mae'])
    model.summary()
    reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=5, min_lr=1e-6, verbose=1)

    #pred_x = unlabeled_x
    #pred_y = []
    #model.fit(train_x, train_y, epochs=100, batch_size=64, validation_data=(test_x, test_y), verbose=1, callbacks=[WandbCallback(), reduce_lr])
    model.fit(set_x, set_y, epochs=100, batch_size=config.batch_size, validation_split=0.2, verbose=1, callbacks=[WandbCallback()])
    #for x in range(5):
    #    pred_y = model.predict(unlabeled_x)
    #    model.fit(np.concatenate((train_x, pred_x), axis=0), np.concatenate((train_y, pred_y), axis=0), epochs=50, batch_size=32, validation_data=(test_x, test_y), verbose=1, callbacks=[WandbCallback(), reduce_lr])
    #pred = model.predict(test_x)
    #for i in range(100):
    #    print("target: ")
    #    print(test_y[i])
    #    print("pred:   ")
    #    print(pred[i])

sess = tf.compat.v1.Session(config=tf.compat.v1.ConfigProto(log_device_placement=True))
run_test()
