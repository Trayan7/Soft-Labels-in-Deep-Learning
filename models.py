import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
from tensorflow import keras
from tensorflow.keras import Input, Model
from tensorflow.keras.applications.resnet_v2 import ResNet50V2
from tensorflow.keras.layers import (Conv2D, Dense, Dropout, Flatten,
                                     GlobalAveragePooling2D, MaxPooling2D)
from tensorflow.keras.models import Sequential
from tensorflow.keras.regularizers import l2


def CNN3_model():
    '''
    Simple convolutional neural network with 3 convolutional layers.
    '''
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

def neg_log_likelihood_bayesian(y_true, y_pred):
    '''
    Negative log likelihood loss in Tensorflow.
    '''
    labels_distribution = tfp.distributions.Categorical(logits=y_pred)
    log_likelihood = labels_distribution.log_prob(tf.argmax(input=y_true, axis=1))
    loss = -tf.reduce_mean(input_tensor=log_likelihood)
    return loss


def get_model(config):
    '''
    Neural network based on ResNet50V2 and adapted to new output.
    Parameters:
        config: Holds all important parameters for the model.
    Returns:
        The neural network prepared for training.
    '''
    # Load base model and setup input shape.
    input_tensor = Input(shape=config.shape)
    base_model = ResNet50V2(include_top=False, weights=None, input_tensor=input_tensor, classes=config.classes)

    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dropout(config.Dropout)(x)
    x = Dense(200, activation='sigmoid')(x)
    x = Dropout(config.Dropout)(x)
    x = Dense(50, activation= 'sigmoid')(x)
    x = Dropout(config.Dropout)(x)
    predictions = Dense(10, activation='softmax')(x)

    # Define new layers before output.
    model = Model(inputs = base_model.input, outputs = predictions)
    model.compile(
        keras.optimizers.Adam(
            lr=config.learning_rate,
            beta_1=0.9,
            beta_2=0.999,
            epsilon=10e-8,
            decay=0.01,
            clipvalue=0.5),
        metrics=['accuracy', 'categorical_crossentropy'],
        loss=tf.keras.losses.mean_squared_logarithmic_error,
        run_eagerly=False)
    return model

def get_model_gmm(config):
    '''
    Neural network based in ResNet50V2 and adapted to new output with GMM output layer.
    Parameters:
        config: Holds all important parameters for the model.
    Returns:
        The neural network prepared for training.
    '''
    # Load base model and setup shapes.
    input_tensor = Input(shape=config.shape)
    base_model = ResNet50V2(include_top=False, weights=None, input_tensor=input_tensor, classes=config.classes)

    event_shape = [10]
    num_components = 1
    params_size = tfp.layers.MixtureNormal.params_size(num_components, event_shape)

    # Define new layers before output.
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dropout(config.Dropout)(x)
    x = Dense(200, activation='sigmoid')(x)
    x = Dropout(config.Dropout)(x)
    x = Dense(50, activation= 'sigmoid')(x)
    x = Dropout(config.Dropout)(x)
    x = Dense(params_size, activation=None)(x)
    predictions = tfp.layers.MixtureNormal(num_components, event_shape)(x)

    # Setup model
    model = Model(inputs = base_model.input, outputs = predictions)
    model.compile(
        keras.optimizers.Adam(
            lr=config.learning_rate,
            beta_1=0.9,
            beta_2=0.999,
            epsilon=10e-8,
            decay=0.01,
            clipvalue=0.5),
        metrics=['accuracy', 'categorical_crossentropy'],
        loss=neg_log_likelihood_bayesian,
        run_eagerly=False)
    return model

def get_model_visual(config):
    '''
    Neural network based in ResNet50V2 and adapted to new output with GMM output layer.
    With intermediary output before the GMM layer.
    Parameters:
        config: Holds all important parameters for the model.
    Returns:
        The neural network prepared for training.
    '''
    # Load base model and setup shapes.
    input_tensor = Input(shape=config.shape)
    base_model = ResNet50V2(include_top=False, weights='imagenet', input_tensor=input_tensor, classes=config.classes)
    event_shape = [2]
    num_components = 1
    params_size = tfp.layers.MixtureNormal.params_size(num_components, event_shape)

    # Define new layers before output.
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dropout(config.Dropout)(x)
    x = Dense(200, activation='sigmoid')(x)
    x = Dropout(config.Dropout)(x)
    x = Dense(50, activation= 'sigmoid')(x)
    x = Dropout(config.Dropout)(x)
    x = Dense(params_size, activation=None)(x)
    predictions = tfp.layers.MixtureNormal(num_components, event_shape)(x)
    
    # Setup models
    intermed_model = Model(inputs = base_model.input, outputs = x)
    model = Model(inputs = base_model.input, outputs = predictions)
    model.compile(
        keras.optimizers.Adam(
            lr=config.learning_rate,
            beta_1=0.9,
            beta_2=0.999,
            epsilon=10e-8,
            decay=0.01,
            clipvalue=0.5),
        metrics=['accuracy', 'categorical_crossentropy'],
        run_eagerly=False,
        loss=neg_log_likelihood_bayesian)

    # Setup model for output before GMM layer
    intermed_model.compile(
        keras.optimizers.Adam(
            lr=config.learning_rate,
            beta_1=0.9,
            beta_2=0.999,
            epsilon=10e-8,
            decay=0.01,
            clipvalue=0.5),
        metrics=['accuracy', 'categorical_crossentropy'],
        run_eagerly=False,
        loss=neg_log_likelihood_bayesian)
    return model, intermed_model
