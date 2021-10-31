from enum import Enum

import numpy as np
from wandb.keras import WandbCallback

import wandb
from data_handling.augmentations import load_datagen, pseudo_label_gen
from data_handling.datasets import (load_cifar10h_k_fold,
                                    load_cifar10h_split_dataset,
                                    load_cifar10h_split_dataset_hard,
                                    load_plankton_semi, load_plankton_split,
                                    load_plankton_split_hard)
from evaluation.metric_callback import (IntervalEvaluation,
                                        IntervalEvaluationHard)
from models import get_model


class Dataset(Enum):
    CIFAR10H = 1
    PLANKTON = 2

def run_test(name, dataset, train_sample_num, epochs, lr, use_soft_labels=True):
    '''
    Trains the model in question under defined parameters.
    Arguments:
        name: The name the test run will have in wandb.
        dataset: Enum for the selection of the dataset.
        train_sample_num: Number of labeled samples used during training.
        epochs: Number of epochs the model is trained for.
        lr: The learning rate for training.
        use_soft_labels: Specifies whether or not the dataset has soft labels.
    '''
    # Setup parameters and wandb
    wandb.init(project='Tests', reinit=True)
    config = wandb.config
    config.learning_rate = lr
    config.batch_size = 256
    config.model = 'ResNet50V2'
    config.dataset = dataset
    config.loss = 'MSLE' # only to keep track, check model
    config.optimizer = 'Adam'
    config.Dropout = 0.2
    config.lrdecay = 0.01
    config.classes = 10
    # config.pre_train_epochs = 200
    # config.num_pseudo = 1000
    # config.k_folds = 5
    wandb.run.name = name
    wandb.run.save()

    # Load dataset
    train_x, train_y, test_x, test_y = [], [], [], []
    if dataset == Dataset.PLANKTON:
        config.shape = (64, 64, 3)
        if use_soft_labels:
            train_x, train_y, test_x, test_y = load_plankton_split(train_sample_num)
        else:
            train_x, train_y, test_x, test_y = load_plankton_split_hard(train_sample_num)
        # Convert to RGB by color channel duplication for model
        train_x = np.repeat(train_x[..., np.newaxis], 3, -1)
        test_x = np.repeat(test_x[..., np.newaxis], 3, -1)
    elif dataset == Dataset.CIFAR10H:
        config.shape = (32, 32, 3)
        if use_soft_labels:
            train_x, train_y, test_x, test_y = load_cifar10h_split_dataset(train_sample_num)
        else:
            train_x, train_y, test_x, test_y = load_cifar10h_split_dataset_hard(train_sample_num)
    # Setup augmentation generator 
    datagen = load_datagen()

    # Load model
    model = get_model(config)
    model.summary()

    # Add custom evaluation
    if use_soft_labels:
        evaluation = IntervalEvaluation(validation_data=(test_x, test_y), interval=10)
    else:
        evaluation = IntervalEvaluationHard(validation_data=(test_x, test_y), interval=10)
        

    # Train model
    model.fit(datagen.flow(train_x, train_y, config.batch_size),
                epochs=epochs,
                batch_size=config.batch_size,
                validation_data=(test_x, test_y),
                verbose=1,
                steps_per_epoch=4,
                callbacks=[WandbCallback(save_model=False), evaluation])
    
    '''
    # Pseudo labeling
    train_x, train_y, test_x, test_y, unlabeled_x = load_plankton_semi(train_sample_num, (10000 - train_sample_num))
    # Pretraining 
    model.fit(datagen.flow(train_x, train_y, config.batch_size),
                epochs=config.pre_train_epochs,
                batch_size=config.batch_size,
                validation_data=(test_x, test_y),
                verbose=1,
                callbacks=[WandbCallback(save_model=False), evaluation])
    # Training with pseudo labels
    model.fit(pseudo_label_gen(train_x, train_y, unlabeled_x, model, config.num_pseudo, datagen, config.batch_size),
                epochs=epochs,
                batch_size=config.batch_size,
                validation_data=(test_x, test_y),
                verbose=1,
                callbacks=[WandbCallback(save_model=False), evaluation])
    '''
    
    '''
    # k-fold cross validation
    sets_x, sets_y = load_cifar10_k_fold(config.k_folds)
    for i in range(config.k_folds):
        idx = []
        for j in range(config.k_folds):
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
    '''
    
'''
train_epochs = 400
for i in range(5):
    run_test('BaselineCifarSoft8k', Dataset.CIFAR10H, 8000, train_epochs * 2, 0.01)
for i in range(5):
    run_test('BaselineCifarHard1k', Dataset.CIFAR10H, 1000, train_epochs, 0.01, False)
for i in range(5):
    run_test('BaselineCifarHard8k', Dataset.CIFAR10H, 8000, train_epochs * 2, 0.01, False)
for i in range(5):
    run_test('BaselineCifarSoft1k', Dataset.CIFAR10H, 1000, train_epochs, 0.01)
for i in range(5):
    run_test('BaselinePlanktonHard1k', Dataset.PLANKTON, 1000, train_epochs, 0.01, False)
for i in range(5):
    run_test('BaselinePlanktonHard10k', Dataset.PLANKTON, 10000, train_epochs * 2, 0.01, False)
for i in range(5):
    run_test('BaselinePlanktonSoft1k', Dataset.PLANKTON, 1000, train_epochs, 0.01)
for i in range(5):
    run_test('BaselinePlanktonSoft10k', Dataset.PLANKTON, 10000, train_epochs * 2, 0.01)
'''
