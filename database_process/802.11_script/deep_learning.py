# coding: utf-8
# tensorflow/titantic

import numpy as np
from __future__ import absolute_import, division, print_function

import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='3'

import tflearn

import tensorflow as tf
gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.9)
sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))

def build_neral_network_model(in_shape=[None, 6]):
    net = tflearn.input_data(shape=in_shape)
    net = tflearn.fully_connected(net, 32)
    net = tflearn.fully_connected(net, 32)
    net = tflearn.fully_connected(net, 2, activation="softmax")
    net = tflearn.regression(net)

    return tflearn.DNN(net)

def nn_train(model, data, label, epoch=50, rate=0.02, batch_size=32, show_metric=True):
    model.fit(data, label, n_epoch=epoch, batch_size=batch_size, show_metric=show_metric)

