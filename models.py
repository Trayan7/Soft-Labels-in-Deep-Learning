from tensorflow.keras.layers import (Conv2D, Dense, Dropout, Flatten,
                                     GlobalAveragePooling2D, MaxPooling2D)
from tensorflow.keras.models import Sequential
from tensorflow import keras
from tensorflow.keras.applications.resnet_v2 import ResNet50V2
from tensorflow.keras.applications.nasnet import NASNetLarge
from tensorflow.keras import Input, Model
from tensorflow.keras.regularizers import l2
import tensorflow as tf

def CNN3_model():
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

def get_model(config):
    input_tensor = Input(shape=(32, 32, 3))
    base_model = ResNet50V2(include_top=False, weights="imagenet", input_tensor=input_tensor, classes=10)
    #base_model = InceptionResNetV2(include_top=False, weights=None, input_tensor=input_tensor, classes=10)
    #base_model = NASNetLarge(include_top=False, weights=None, input_tensor=input_tensor, classes=10)
    '''
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dropout(config.Dropout)(x)
    predictions = Dense(10, activation= 'sigmoid')(x)
    '''
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dropout(config.Dropout)(x)
    x = Dense(200, activation= 'sigmoid')(x)
    x = Dropout(config.Dropout)(x)
    x = Dense(50, activation= 'sigmoid')(x)
    x = Dropout(config.Dropout)(x)
    predictions = Dense(10, activation= 'softmax')(x)
    model = Model(inputs = base_model.input, outputs = predictions)
    model.compile(keras.optimizers.Adam(lr=config.learning_rate, beta_1=0.9, beta_2=0.999, epsilon=10e-8, decay=0.01, clipvalue=0.5), loss=tf.keras.losses.CategoricalCrossentropy(), metrics=['accuracy', 'mae', "kullback_leibler_divergence"])
    return model