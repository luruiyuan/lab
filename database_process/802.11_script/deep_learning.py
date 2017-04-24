# coding: utf-8
# tensorflow/titantic

from __future__ import absolute_import, division, print_function

from classification import get_data_label as get_data, get_classification_accuracy as accuracy

import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='3'

from os import path, makedirs

import tensorflow as tf
import tflearn
tflearn.init_graph(num_cores=8, gpu_memory_fraction=0.6)

import time

def gen_nets_by_config(config, timestamp):
    # read cofiguration
    title = config['title']
    dropout = config['dropout']
    hidden_act = config['hidden_act']
    hidden_layer_num = config['hidden_layer_num']
    out_act = config['out_act']
    label_num = config['label_num']
    learning_rate = config['learning_rate']
    optimizer = config['optimizer']
    node_num = config['node_num']
    loss = config['loss']
    shape = config['shape']
    
    print("generating %s..." % title)

    # generate network
    net  = tflearn.input_data(shape=shape)

    # hidden layer
    for i in range(hidden_layer_num):
        node_num = node_num << 1 if i <= hidden_layer_num // 2 else node_num >> 1
        # debug
        # fc = "net = tflearn.fully_connected(net, %d, activation='%s')" % (node_num, hidden_act)
        # dr = "net = tflearn.dropout(net, %f)" % (dropout)
        # print(fc)
        # print(dr)
        exec("net = tflearn.fully_connected(net, %d, activation='%s')" % (node_num, hidden_act))

    exec("net = tflearn.dropout(net, %f)" % (dropout)) # avoid overfit

    net = tflearn.fully_connected(net, label_num, activation=out_act)
    net = tflearn.regression(net, learning_rate=learning_rate, loss=loss, optimizer=optimizer)

    # define different log path for each model in order to ensure concurrence
    log_path = path.join('./models/', timestamp, title, 'tflearn_logs/')
    makedirs(log_path)
    dnn = tflearn.models.dnn.DNN(net, tensorboard_dir=log_path, best_val_accuracy=0.5)
    print("generating %s succeeded!" % title)
    return dnn

def title2model_conf(title):
    l = title.split('_')
    hidden_layer = int(l[0])
    shape = [None, 9]
    dropout = 0.2
    hidden_activation = l[2]
    out_activation = l[-1]
    label_num = 31
    learning_rate = 0.0005
    node_num = 16
    optimizer = 'adam'
    loss = 'categorical_crossentropy'

    return {
        "title": title,
        "dropout": dropout,
        "shape": shape,
        "hidden_act": hidden_activation,
        "hidden_layer_num": hidden_layer,
        "out_act": out_activation,
        "label_num": label_num,
        "learning_rate": learning_rate,
        "optimizer": optimizer,
        "node_num": node_num,
        "loss": loss,
    }

def build_neural_network_model(title, label_num, timestamp):
    print("building network: %s..." % title)
    # models = list(map(gen_nets_by_config, list(map(title2model_conf, titles)))) # generate multiple models from title
    model = gen_nets_by_config(title2model_conf(title), timestamp) # generate model from title
    print("building network: %s succeeded!" % title)
    
    return model

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



def get_timestamp():
    return time.strftime('%Y-%m-%d_%H:%M:%S', time.localtime())
    
def save_models(*, timestamp, titles=[], models=[]):
    """
    Saving models and return the root dir path of models
    """

    if not isinstance(titles, list):
        titles = [titles]
    if not isinstance(models, list):
        models = [models]
    
    if len(titles) == 1:
        print("saving model: %s..." % titles[0])
    else:
        print("saving %d models..." % len(models))
    
    root = path.join("./models/", timestamp)  # root dir
    
    
    for title, model in zip(titles, models):
        relative_path = path.join(root, title)
        if not path.exists(relative_path):
            makedirs(relative_path) # create dir
        model.save(path.join(relative_path, title)) # save model
    
    if len(titles) == 1:
        print("saving model: %s finished" % titles[0])
    else:
        print("saving %d models finished!" % len(models))
    
    return root

def load_models(titles, models):
    pass
    # """
    # not finished yet.
    # """
    # pass
    # print("loading model from file...")
    # model = build_neural_network_model(in_shape=[None, 9])
    # model.load("model")
    # print("loading model finished!")

def get_dl_accuracy(*, predict_labels, correct_labels): 
    print("start deep learning accuracy calculation...")
    accuracy = .0
    for pre, corr in zip(predict_labels, correct_labels):
        accuracy += 1 if list(pre).index(1) == corr.index(1) else 0 # predict labels has ndarrays
    accuracy /= len(predict_labels)
    print("accuracy calculation finished!\n  accuracy:", accuracy)
    return accuracy


def train_validate_worker(msg_que, train_x, train_y, validate_x, validate_y, title, evaluate_func, epoch, batch_size, timestamp):
    
    model = build_neural_network_model(title, 31, timestamp)
    
    res = {"title": title}
    # train model: model cannot be transfered to parent process
    print(title,"training...")
    model.fit(train_x, train_y, n_epoch=epoch, batch_size=batch_size, show_metric=True)
    print(title, "training finished!")

    # save model
    save_models(timestamp=timestamp, titles=[title], models=[model])

    #validate
    print(title,"predicting...")
    predict_labels = model.predict_label(validate_x)
    print(title,"predicting finished!")
    res["result"] = predict_labels

    # evaluate
    print(title,"evaluating...")
    res["evaluate"] = evaluate_func(predict_labels=predict_labels, correct_labels=validate_y)
    print(title,"evaluating finished!")
    
    # put res into message queue
    msg_que.put(res)


# The model trained using tflearn cannot be pickled, thus cannot be transformed
# between prcess and subprocess.
# Even We start concurrent from the begining: build_neural_network_model, we will find that each training
# won't release screen handle until current training is finished, which causes other processes quit.
# Thus, tensorflow is not appropriate to train multiple model in the mean time.

def train_validate_manager(train_x, train_y, validate_x, validate_y, titles, evaluate_func):
    from multiprocessing import cpu_count, Pool, Manager
    import time

    # define a result queue
    q = Manager().Queue() # multiprocessing manager queue

    # init process pool
    pool = Pool(cpu_count())

    # init training params
    epoch = 6000
    batch_size = 128
    timestamp = get_timestamp()

    whole_start = time.time()
    
    print("dynamiclly generating %d nets..." % len(titles))
    
    # # It seems that multiprocessing TFLearn cannot process different model in the mean time
    # # multiprocessing
    # print("starting multiprocess_train_validate...\nprocess pool size: %d model number: %d" % (cpu_count(), len(titles)))
    
    # for title in titles:
    #     pool.apply_async(train_validate_worker, args=(q,train_x, train_y, validate_x, validate_y, title, evaluate_func, epoch, batch_size, timestamp))
    # # waiting for processing finished
    # pool.close()
    # pool.join()
    
    singal processing
    for title in titles:
        # reset graph, otherwise these code will raise:# feed_dict[net_inputs[i]] = x;IndexError: list index out of range
        tf.reset_default_graph() 
        train_validate_worker(q,train_x, train_y, validate_x, validate_y, title, evaluate_func, epoch, batch_size, timestamp)

    whole_end = time.time()
    print("whole time for training, validation and evaluating: %.3f seconds." % (whole_end - whole_start))
    # gather results from result queue
    titles, predict_res, evaluates = [], [], []
    while not q.empty():
        res = q.get()
        predict_res.append(res["result"])
        titles.append(res["title"])
        evaluates.append(res["evaluate"])
    
    return titles, predict_res, evaluates

def train_validate():
    # get data
    attr_names, label_names, train_fraction, \
        train_x, train_y, validate_x, validate_y = get_data(exclude_attr_columns=["Time", "Protocol", "Source address", "Destination", "TSF timestamp", "Qos Control Field"])

    # transform label data into required type
    look_up_table = assign_id_for_label(train_y) # create look up table for each label
    label_list = get_label_list(look_up_table) # using this we can find label by id

    train_y = transform_label_to_vector(train_y, look_up_table) # transform label type from str to vector
    validate_y = transform_label_to_vector(validate_y, look_up_table)

    # define structure and titles of models

    titles = [
        "3_hidden_relu_out_softmax",
        "4_hidden_relu_out_softmax",
        "5_hidden_relu_out_softmax",
        "6_hidden_relu_out_softmax",
        "7_hidden_relu_out_softmax",
        "8_hidden_relu_out_softmax",
        "9_hidden_relu_out_softmax",
        "10_hidden_relu_out_softmax",
        "11_hidden_relu_out_softmax",
        "12_hidden_relu_out_softmax",
        "13_hidden_relu_out_softmax",
        "14_hidden_relu_out_softmax",
        "15_hidden_relu_out_softmax",
    ]

    # training and validating
    titles, predict_res, evaluates = train_validate_manager(train_x, train_y, validate_x, validate_y, \
                                titles, get_dl_accuracy)
    
    return attr_names, label_names, train_fraction, train_x, train_y, \
            validate_x, validate_y, titles, predict_res, evaluates

def main():
    from classification import print_res
    attr_names, label_names, train_fraction, train_x, train_y, \
            validate_x, validate_y, titles, \
            predict_res, evaluate_res = train_validate()
    
    print_res(titles, evaluate_res)


if __name__ == '__main__':
    main()