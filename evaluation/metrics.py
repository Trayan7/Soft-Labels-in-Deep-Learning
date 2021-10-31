from __future__ import absolute_import, division, print_function

import functools

import numpy as np
import tensorflow as tf
from keras import backend as K
from tensorflow.python.ops.math_ops import argmax


def center_diff_loss_func(features, labels, num_classes):
    '''
    Calculates the average distance between label and prediciton centers.
    '''
    centroids_labels = np.zeros((num_classes, num_classes))
    centroids_features = np.zeros((num_classes, num_classes))
    for i in range(num_classes):
        weighted_features = features * K.flatten(features[:][i])
        centroids_features[i] = K.sum(weighted_features) / K.sum(features)
        weighted_labels = labels * K.flatten(labels[:][i])
        centroids_labels[i] = K.sum(weighted_labels) / K.sum(labels)
    
    loss = K.sqrt(K.sum(K.square(centroids_features - centroids_labels)))

    return loss

def get_center_diff_loss(num_classes):
    '''
    Returns a wrapped version of the center difference function.
    '''
    @functools.wraps(center_diff_loss_func)
    def center_diff_loss(y_true, y_pred):
        return center_diff_loss_func(y_pred, y_true, num_classes)
    return center_diff_loss
    

def mean_center_loss_func(features, labels, num_classes):
    '''
    Calculates the average distance of the predictions to the label center.
    '''
    centroids_labels = np.zeros((num_classes, num_classes))
    for i in range(num_classes):
        weighted_labels = labels * K.flatten(labels[:][i])
        centroids_labels[i] = K.sum(weighted_labels) / K.sum(labels)
    
    label_classes = K.argmax(labels)
    label_classes = np.reshape(label_classes, (tf.size(label_classes), 1))
    weighted_labels = tf.cast(features, tf.float32) * tf.cast(tf.gather_nd(labels, label_classes), tf.float32)
    diffs = K.sqrt(K.sum(K.square(tf.cast(tf.gather_nd(centroids_labels, label_classes), tf.float32) - weighted_labels)))
    loss = K.sum(diffs) / tf.cast(tf.size(diffs), tf.float32)
    loss = loss / num_classes

    return loss

def get_mean_center_loss(num_classes):
    '''
    Returns a wrapped version of the mean center loss function.
    '''
    @functools.wraps(mean_center_loss_func)
    def mean_center_loss(y_true, y_pred):
        return mean_center_loss_func(y_pred, y_true, num_classes)
    return mean_center_loss


def second_best_accuracy_func(features, labels):
    '''
    Calculates the accuracy of the second most probable prediciton,
    if the labels have at least one label if a second class membership.
    '''
    scnd_pred = features.copy()
    scnd_label = labels.copy()
    scnd_pred_pruned = []
    scnd_label_pruned = []
    for elem in scnd_pred:
        elem[np.argmax(elem)] = 0
    for elem in scnd_label:
        elem[np.argmax(elem)] = 0
    # Only take labels with more than one class
    for i in range(len(scnd_label)):
        if np.argmax(scnd_label[i]) != 0:
            scnd_pred_pruned.append(scnd_pred[i])
            scnd_label_pruned.append(scnd_label[i])
    if len(scnd_label_pruned) > 0:
        scnd_acc = (np.argmax(scnd_pred_pruned, axis=1) == np.argmax(scnd_label_pruned, axis=1)).mean() * 100
    else:
        # Default failure case
        scnd_acc = 0.0
    return scnd_acc

def get_second_best_accuracy():
    '''
    Returns a wrapped version of the second best accuracy function.
    '''
    @functools.wraps(second_best_accuracy_func)
    def second_best_accuracy(y_true, y_pred):
        return second_best_accuracy_func(y_pred, y_true)
    return second_best_accuracy
