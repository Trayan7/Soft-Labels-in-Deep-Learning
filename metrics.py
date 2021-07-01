from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import functools
from keras import backend as K
import tensorflow as tf
import numpy as np
from tensorflow.python.ops.math_ops import argmax


def center_diff_loss_func(features, labels, num_classes):
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
    @functools.wraps(center_diff_loss_func)
    def center_diff_loss(y_true, y_pred):
        return center_diff_loss_func(y_pred, y_true, num_classes)
    return center_diff_loss

def mean_center_loss_func(features, labels, num_classes):
    centroids_labels = np.zeros((num_classes, num_classes))
    for i in range(num_classes):
        weighted_labels = labels * K.flatten(labels[:][i])
        centroids_labels[i] = K.sum(weighted_labels) / K.sum(labels)
    
    label_classes = K.argmax(labels)
    # Put each element in a list
    #label_classes = [label_classes[i:i + 1] for i in range(len(label_classes))]
    label_classes = np.reshape(label_classes, (tf.size(label_classes), 1))
    weighted_labels = tf.cast(features, tf.float32) * tf.cast(tf.gather_nd(labels, label_classes), tf.float32)
    diffs = K.sqrt(K.sum(K.square(tf.cast(tf.gather_nd(centroids_labels, label_classes), tf.float32) - weighted_labels)))
    loss = K.sum(diffs) / tf.cast(tf.size(diffs), tf.float32)
    loss = loss / num_classes

    return loss

def get_mean_center_loss(num_classes):
    @functools.wraps(mean_center_loss_func)
    def mean_center_loss(y_true, y_pred):
        return mean_center_loss_func(y_pred, y_true, num_classes)
    return mean_center_loss

