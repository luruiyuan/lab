# -*- coding: utf-8 -*-
import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='3'
import tensorflow as tf
import numpy as np

gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.8)
sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))

features = [tf.contrib.layers.real_valued_column("x", dimension=1)]

estimator = tf.contrib.learn.LinearRegressor(feature_columns=features)

x = np.array([1.,2.,3.,4.])
y = np.array([0.,-1.,-2.,-3.])

input_fn = tf.contrib.learn.io.numpy_input_fn({"x": x}, y, batch_size=4, num_epochs=1000)

estimator.fit(input_fn=input_fn, steps=1000)

print(estimator.evaluate(input_fn=input_fn))