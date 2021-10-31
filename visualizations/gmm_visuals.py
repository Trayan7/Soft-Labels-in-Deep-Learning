import math

import matplotlib.pyplot as plt
import numpy as np
import wandb
from data_handling.augmentations import load_datagen
from data_handling.datasets import load_plankton_split_hard
from models import get_model_visual
from mpl_toolkits.mplot3d import Axes3D
from sklearn.mixture import GaussianMixture
from wandb.integration.keras.keras import WandbCallback


def rotate(points, origin, angle):
    '''
    Rotates a 2D point set and returns them in new list.
    :param points: Set of 2D points
    :param origin: 2D point to turn the points around.
    :param angle: Rotation angle in radians.
    '''
    rot_points = []
    ox, oy = origin
    for point in points:
        px, py = point
        rx = ox + math.cos(angle) * (px - ox) - math.sin(angle) * (py - oy)
        ry = oy + math.sin(angle) * (px - ox) + math.cos(angle) * (py - oy)
        rot_points.append([rx, ry])
    return rot_points

def gmm_model_visual(name, train_sample_num, epochs, lr):
    '''
    Reduces the plankton set to 2 classes and trains a GMM model on it.
    Produces point clouds of all pre GMM layer 2 dimensional combinations
    with GMM output as height.
    '''
    # Setup model and training parameters
    wandb.init(project='GMM', reinit=True)
    config = wandb.config
    config.learning_rate = lr
    config.batch_size = 256
    config.model = 'ResNet50V2'
    config.dataset = 'Plankton'
    config.loss = 'neg_log_likelihood'
    config.optimizer = 'Adam'
    config.Dropout = 0.0
    config.lrdecay = 0.01
    config.classes = 2
    wandb.run.name = name
    wandb.run.save()
    train_x, train_y, test_x, test_y = load_plankton_split_hard(train_sample_num)

    train_x2 = []
    train_y2 = []
    test_x2 = []
    test_y2 = []

    # Filter samples by selected classes
    selected_classes = (0, 1)
    for i in range(len(train_x)):
        label = train_y[i]
        if np.argmax(label) == selected_classes[0]:
            train_x2.append(train_x[i])
            train_y2.append([1,0])
        if np.argmax(label) == selected_classes[1]:
            train_x2.append(train_x[i])
            train_y2.append([0,1])
    
    for i in range(len(test_x)):
        label = test_y[i]
        if np.argmax(label) == selected_classes[0]:
            test_x2.append(test_x[i])
            test_y2.append([1,0])
        if np.argmax(label) == selected_classes[1]:
            test_x2.append(test_x[i])
            test_y2.append([0,1])

    train_x = np.array(train_x2)
    train_y = np.array(train_y2)
    test_x = np.array(test_x2)
    test_y = np.array(test_y2)

    # Setup model
    model, intermed = get_model_visual(config)
    model.summary()

    datagen = load_datagen()

    # convert to RGB for model
    train_x = np.repeat(train_x[..., np.newaxis], 3, -1)
    test_x = np.repeat(test_x[..., np.newaxis], 3, -1)

    # Train in segments and save visualizations of output.
    repetitions = 20
    epochs_per_rep = 10
    for x in range(repetitions):
        model.fit(datagen.flow(train_x, train_y, config.batch_size), epochs=epochs_per_rep, batch_size=config.batch_size, validation_data=(test_x, test_y), verbose=1, callbacks=[WandbCallback(save_model=False)])
        pred = model.predict(test_x, verbose=0)
        pre_pred = intermed.predict(test_x, verbose=0)
        for i in range(5):
            for j in range(5):
                fig = plt.figure()
                ax = Axes3D(fig)
                ax.scatter(pre_pred[:,i], pre_pred[:,j], pred[:,0], c=(1,0,0))
                ax.scatter(pre_pred[:,i], pre_pred[:,j], pred[:,1], c=(0,0,1))
                fig.savefig('./preview/GMMPlankton' + str(i) + str(j) + 'e' + str(x*10+10) + '.pdf')

def gmm_visual():
    # Generate multiple graphs to choose applicable example.
    for i in range(10):
        # Generate a combination of 2 normal distributions.
        points1 = np.random.normal([0.0, 0.0], [0.8, 1.5], (100, 2))
        points2 = np.random.normal([0.0, 0.0], [1.6, 1], (100, 2))
        points = np.concatenate([rotate(points1, [1.0, 0.0], 40), points2])

        # Fit GMM to points.
        gmm = GaussianMixture(n_components=2)
        gmm.fit(points)

        # Setup meshgrid to visualize contour lines.
        X, Y = np.meshgrid(np.linspace(-10, 10, 500), np.linspace(-10, 10, 500))
        XX = np.array([X.ravel(), Y.ravel()]).T
        Z = gmm.score_samples(XX)
        Z = Z.reshape((500, 500))

        # Generate and save plots.
        fig = plt.figure()
        plt.scatter(points[:, 0], points[:, 1])
        plt.contour(X, Y, Z)
        fig.savefig('./preview/GMMTest' + str(i) + '.pdf')
