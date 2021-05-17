from custom_callbacks import CustomCallback
import itertools
import sys
import wandb
import tensorflow as tf
import numpy as np
import tempfile
from process_data import load_dataset, load_datagen, load_split_dataset, get_data_aug, load_datagen_val, augment, mixup_gen, naive_mix_up, naive_mix_up2
from tensorflow.keras import Input, Model, callbacks
from models import get_model
from wandb.keras import WandbCallback

def run_test():
    tempfile.tempdir = 'E:/Uni/Masterarbeit/wandb temp'
    wandb.init(project='supervised')
    config = wandb.config
    config.learning_rate = 0.01
    config.batch_size = 256
    config.model = "ResNet50V2"
    config.dataset = "CIFAR10H"
    config.loss = "categorical crossentropy"
    config.optimizer = "Adam"
    config.Dropout = 0.2
    config.lrdecay = 0.01
    wandb.run.name = "80,20 layers 200,50,10 aug 3"
    wandb.run.save()
    #set_x, set_y = load_dataset()
    train_x, train_y, test_x, test_y = load_split_dataset(0.8)
    #train_x, train_y, test_x, test_y, unlabeled_x = load_dataset_semi_supervised()
    model = get_model(config)
    model.summary()
    #reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=5, min_lr=1e-6, verbose=1)

    #model.fit(set_x, set_y, epochs=500, batch_size=config.batch_size, validation_split=0.2, verbose=1, callbacks=[WandbCallback()])

    datagen = load_datagen()

    #model.fit(mixup_gen(train_x, train_y, config.batch_size, datagen, 0.1), epochs=200, steps_per_epoch=32, batch_size=config.batch_size, validation_data=(test_x, test_y), verbose=1, callbacks=[WandbCallback()])
    model.fit(datagen.flow(train_x, train_y, config.batch_size), epochs=400, batch_size=config.batch_size, validation_data=(test_x, test_y), verbose=1, callbacks=[WandbCallback()])
    #model.fit(datagen.flow(train_x, train_y, config.batch_size), epochs=10, validation_data=(test_x, test_y), verbose=1, callbacks=[WandbCallback()])

#sess = tf.compat.v1.Session(config=tf.compat.v1.ConfigProto(log_device_placement=True))
run_test()