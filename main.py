import os
from custom_callbacks import CustomCallback
import itertools
import sys
import wandb
import tensorflow as tf
import tensorflow_probability as tfp
import numpy as np
import tempfile
from process_data import load_dataset, load_datagen, load_k_fold, load_split_dataset, get_data_aug, load_datagen_val, augment, load_split_dataset_normal, mixup_gen, naive_mix_up, naive_mix_up2, samples_per_class
from tensorflow.keras import Input, Model, callbacks
from models import get_model
from wandb.keras import WandbCallback
#from fastai.callback.wandb import SaveModelCallback

def test_callback(self, epoch):
    y_true = self.y_true
    y_pred = self.y_pred

def run_test():
    tempfile.tempdir = 'E:/Uni/Masterarbeit/wandb'
    wandb.init(project='supervised', reinit=True)
    config = wandb.config
    config.learning_rate = 0.01
    config.batch_size = 256
    config.model = "ResNet50V2"
    config.dataset = "CIFAR10H"
    config.loss = "MSLE"
    config.optimizer = "Adam"
    config.Dropout = 0
    config.lrdecay = 0.01
    wandb.run.name = "1000, test metrics 2"
    wandb.run.save()
    #set_x, set_y = load_dataset()
    train_x, train_y, test_x, test_y = load_split_dataset(0.9)
    #train_x, train_y, test_x, test_y, unlabeled_x = load_dataset_semi_supervised()
    model = get_model(config)
    model.summary()
    
    #reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=5, min_lr=1e-6, verbose=1)

    #model.fit(set_x, set_y, epochs=500, batch_size=config.batch_size, validation_split=0.2, verbose=1, callbacks=[WandbCallback()])

    datagen = load_datagen()

    #model.fit(mixup_gen(train_x, train_y, config.batch_size, datagen, 0.1), epochs=200, steps_per_epoch=32, batch_size=config.batch_size, validation_data=(test_x, test_y), verbose=1, callbacks=[WandbCallback()])
    model.fit(datagen.flow(train_x, train_y, config.batch_size), epochs=500, batch_size=config.batch_size, validation_data=(test_x, test_y), verbose=1, callbacks=[WandbCallback(save_model=False)])
    #wandb.save("E:/Uni/Masterarbeit/wandb/model.h5", base_path="E:/Uni/Masterarbeit/wandb")
    #model.fit(datagen.flow(train_x, train_y, config.batch_size), epochs=10, validation_data=(test_x, test_y), verbose=1, callbacks=[WandbCallback()])
    pred = model(test_x)
    for i in range(100):
        print("Sample ", i)
        print("truth ", test_y[i])
        print("prediction ", pred[i])

def run_test_k(k):

    #wandb.run.save()

    sets_x, sets_y = load_k_fold(k)
    print("Shape Xs: ", np.shape(sets_x))
    print("Shape Ys: ", np.shape(sets_y))
    for i in range(len(sets_x)):
        print("Index: ", i)
        print("Shape X: ", np.shape(sets_x[i]))
        print("Shape Y: ", np.shape(sets_y[i]))

    datagen = load_datagen()
    for i in range(k):
        idx = []
        for j in range(k):
            if i != j:
                idx.append(j)

        train_x = [sets_x[i] for i in idx]
        train_y = [sets_y[i] for i in idx]
        train_x = [e for sub in train_x for e in sub]
        train_y = [e for sub in train_y for e in sub]
        train_x = np.array(train_x)
        train_y = np.array(train_y)
        
        test_x = sets_x[i]
        test_y = sets_y[i]
        print("Shape X: ", np.shape(train_x))
        print("Shape Y: ", np.shape(train_y))
        print("Shape X: ", np.shape(test_x))
        print("Shape Y: ", np.shape(test_y))

        wandb.init(project='supervised', reinit=True)
        config = wandb.config
        config.learning_rate = 0.01
        config.batch_size = 256
        config.model = "ResNet50V2"
        config.dataset = "CIFAR10H"
        config.loss = "MSLE"
        config.optimizer = "Adam"
        config.Dropout = 0.2
        config.lrdecay = 0.01
        wandb.run.name = "best config 10 fold test metrics"
        model = get_model(config)
        #model.summary()
        model.fit(datagen.flow(train_x, train_y, config.batch_size), epochs=500, batch_size=config.batch_size, validation_data=(test_x, test_y), verbose=1, callbacks=[WandbCallback(save_model=False)])

def run_soft_labeling(vote_count, repetitions):
    train_x, train_y, test_x, test_y, unlabeled_x = samples_per_class(100, 100, 10)
    train_x_cur = train_x
    train_y_cur = train_y
    cur_epochs = 500
    datagen = load_datagen()

    for i in range(repetitions):
        unlabeled_pred = list()
        if i == 0:
            cur_epochs = 250
        else:
            cur_epochs = 100
        for j in range(vote_count):
            tempfile.tempdir = 'E:/Uni/Masterarbeit/wandb'
            wandb.init(project='supervised', reinit=True)
            config = wandb.config
            config.learning_rate = 0.005
            config.batch_size = 256
            config.model = "ResNet50V2"
            config.dataset = "CIFAR10H"
            config.loss = "neg_log_likelihood_bayesian"
            config.optimizer = "Adam"
            config.Dropout = 0
            config.lrdecay = 0.01
            wandb.run.name = "100 labeling v: " + str(vote_count) + " r: " + str(i)
            wandb.run.save()

            model = get_model(config)
            #model.summary()
            model.fit(datagen.flow(train_x_cur, train_y_cur, config.batch_size), epochs=cur_epochs, batch_size=config.batch_size, validation_data=(test_x, test_y), verbose=1, callbacks=[WandbCallback(save_model=False)])

            unlabeled_pred.append(model.predict(unlabeled_x))
        
        avg_pred = [(sum(k) / vote_count) for k in zip(*unlabeled_pred)]
        train_x_cur = np.concatenate([train_x, unlabeled_x])
        train_y_cur = np.concatenate([train_y, avg_pred])


#sess = tf.compat.v1.Session(config=tf.compat.v1.ConfigProto(log_device_placement=True))
run_test()
    