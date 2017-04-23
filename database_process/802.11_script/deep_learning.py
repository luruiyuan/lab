# coding: utf-8
# tensorflow/titantic

from __future__ import absolute_import, division, print_function

from classification import get_data_label as get_data

import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='3'

import tflearn

import tensorflow as tf
gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.9)
sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))
sess.close()

def build_neral_network_model(in_shape=[None, 6]):
    net = tflearn.input_data(shape=in_shape)
    net = tflearn.fully_connected(net, 32, activation="relu")
    net = tflearn.fully_connected(net, 64, activation="relu")
    net = tflearn.fully_connected(net, 128, activation="relu")
    net = tflearn.fully_connected(net, 64, activation="relu")
    net = tflearn.fully_connected(net, 32, activation="relu")
    net = tflearn.fully_connected(net, 31, activation="relu")
    net = tflearn.regression(net)

    return tflearn.DNN(net)

def assign_id_for_label(labels):
    """
    assign id for labels
    labels must be a 1-d tensor
    """
    print("assging id for label...")
    table = {j: i for (i, j) in enumerate(set(labels))}
    print("assging id finished!")
    return table

def get_label_list(look_up_table):
    """
    generate a list order by the look_up_table
    for example:
        look_up_table:{"shenzhen":0, "huawei":1}
        the list will be ["shenzhen", "huawei"]
    
    Return:
        ordered list
    """
    print("generating label list by look up table..")
    ordered_labels = [0 for i in look_up_table]
    for key, value in look_up_table.items():
        ordered_labels[value] = key
    print("generating finished!")    
    return ordered_labels

def transform_label_to_vector(data, look_up_table):
    """
    assign id for non numeric data
    
    Param:
    data: must be a 1-d tensor
    look_up_table: a value to index dict. the key of it is value, and the value is the index in the list

    Return:
        return a 2-d tensor
        For example: if we have 3 catogries, ["Tony","Sam","Anna", "Anna"].
        The look_up_table might look like this:{"Tony":2, "Sam":0, "Anna":1}.
        We will return a list:[[0,0,1], [1,0,0], [0,1,0], [0,1,0]] 
            represents ["Tony","Sam","Anna", "Anna"] respectively.
    """
    print("label is being transfered to vectors...")
    if not isinstance(data, list):
        data = [data]
    labels_vecs = [[1 if look_up_table[d] == k else 0 for k in range(len(look_up_table))] for d in data] 
    print("label transfering finished!")    
    return labels_vecs

# get data
attr_names, label_names, train_fraction, \
    train_x, train_y, validate_x, validate_y = get_data(exclude_attr_columns=["Time", "Protocol", "Source address", "Destination", "TSF timestamp", "Qos Control Field"])

# define model
model = build_neral_network_model(in_shape=[None, len(attr_names)])

# transform label data into required type
look_up_table = assign_id_for_label(train_y) # create look up table for each label
label_list = get_label_list(look_up_table) # using this we can find label by id

train_y = transform_label_to_vector(train_y, look_up_table) # transform label type from str to vector
validate_y = transform_label_to_vector(validate_y, look_up_table)

model.fit(train_x, train_y, n_epoch=10000, batch_size=128, show_metric=True)
pred = model.predict(validate_x)