# coding: utf-8
# tensorflow/titantic

from __future__ import absolute_import, division, print_function

from classification import get_data_label as get_data, get_classification_accuracy as accuracy

import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='3'

from os import path, makedirs

import tensorflow as tf
import tflearn
tflearn.init_graph(gpu_memory_fraction=0.8)

import time

def gen_nets_by_config(config, timestamp, net_code):
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

    # define different log path for each model in order to ensure concurrence
    log_path = path.join('./models/', timestamp, title, 'tflearn_logs/')
    makedirs(log_path)
    # generate network
    lines = [s.strip() for s in net_code.split('\n')]
    for i in range(len(lines) - 1):
        print("line:", lines[i])
        exec(lines[i])

    print("generating %s succeeded!" % title)
    return eval(lines[-1])
    # return model # model must be defined in the code lines in the net_code list

def title2model_conf(title):
    l = title.split('_')
    hidden_layer = int(l[0])
    shape = [None, 9]
    dropout = 0.2
    hidden_activation = l[2]
    out_activation = l[-1]
    label_num = 31
    learning_rate = 1
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

def build_neural_network_model(title, net_code, timestamp):
    print("building network: %s..." % title)
    
    model = gen_nets_by_config(title2model_conf(title), timestamp, net_code) # generate model from title

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
    
def save_models(*, timestamp, titles=[], models=[], codes=[]):
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
    
    
    for title, model, code in zip(titles, models, codes):
        relative_path = path.join(root, title)
        if not path.exists(relative_path):
            makedirs(relative_path) # create dir
        ch_path = path.join(relative_path, title) # child path
        model.save(ch_path) # save model
        with open(ch_path+"_code.txt", 'w') as f:
            f.write(code)
    
    if len(titles) == 1:
        print("saving model: %s finished" % titles[0])
    else:
        print("saving %d models finished!" % len(models))
    
    return root

def load_models(titles, models):
    pass
    # not finished yet.

def get_dl_accuracy(*, predict_labels, correct_labels): 
    print("start deep learning accuracy calculation...")
    accuracy = .0

    for pre, corr in zip(predict_labels, correct_labels):
        pre_list = list(pre)
        accuracy += 1 if pre_list.index(max(pre_list)) == corr.index(1) else 0 # predict labels has ndarrays
    accuracy /= len(predict_labels)

    print("accuracy calculation finished!\n  accuracy:", accuracy)
    return accuracy


def train_validate_worker(msg_que, train_x, train_y, validate_x, validate_y, title, code, evaluate_func, epoch, batch_size, timestamp):
    
    # train model: model cannot be transfered to parent process    
    model = build_neural_network_model(title, code, timestamp)
    
    res = {"title": title}
    print(title,"training...")
    model.fit(train_x, train_y, n_epoch=epoch, batch_size=batch_size, show_metric=True)
    print(title, "training finished!")

    #validate
    print(title,"predicting...")
    predict_labels = model.predict(validate_x)

    print(title,"predicting finished!")
    res["result"] = predict_labels

    # saving model
    save_models(timestamp=timestamp, titles=[title], models=[model], codes=[code])

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

def train_validate_manager(train_x, train_y, validate_x, validate_y, titles, codes, evaluate_func):
    from multiprocessing import cpu_count, Pool, Manager
    import time

    # define a result queue
    q = Manager().Queue() # multiprocessing manager queue

    # init process pool
    pool = Pool(cpu_count())

    # init training params
    epoch = 1000
    batch_size = 256
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
    
    # singal processing
    for title, code in zip(titles, codes):
        # reset graph, otherwise these code will raise:# feed_dict[net_inputs[i]] = x;IndexError: list index out of range, 
        # and model generation must after the default process.
        tf.reset_default_graph()
        train_validate_worker(q,train_x, train_y, validate_x, validate_y, title, code, evaluate_func, epoch, batch_size, timestamp)

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

def train_validate(*,train_fraction=0.6):
    # get data
    attr_names, label_names, train_fraction, \
        train_x, train_y, validate_x, validate_y = get_data(exclude_attr_columns=["Time", "Protocol", "Source address", "Destination", "TSF timestamp", "Qos Control Field"],train_fraction=train_fraction)

    # transform label data into required type
    look_up_table = assign_id_for_label(train_y) # create look up table for each label
    label_list = get_label_list(look_up_table) # using this we can find label by id

    train_y = transform_label_to_vector(train_y, look_up_table) # transform label type from str to vector
    validate_y = transform_label_to_vector(validate_y, look_up_table)

    # define structure and titles of models

    titles = [
        "10_hidden_layer_out_softmax",
        "15_hidden_layer_out_softmax",
        "20_hidden_layer_out_softmax",
    ]

    codes = [
        """
        net  = tflearn.input_data(shape=[None, 9])
        net = tflearn.fully_connected(net, 32, activation="relu")
        net = tflearn.fully_connected(net, 64, activation="relu")
        net = tflearn.fully_connected(net, 128, activation="relu") # add a layer
        net = tflearn.fully_connected(net, 256, activation="relu") # add a layer
        net = tflearn.fully_connected(net, 512, activation="relu") # add a layer
        net = tflearn.fully_connected(net, 256, activation="relu") # add a layer
        net = tflearn.fully_connected(net, 128, activation="relu") # add a layer
        net = tflearn.fully_connected(net, 64, activation="relu")
        net = tflearn.fully_connected(net, 64, activation="relu")
        net = tflearn.fully_connected(net, 32, activation="relu")
        net = tflearn.fully_connected(net, 31, activation="softmax")
        net = tflearn.regression(net, learning_rate=0.0001)
        tflearn.DNN(net, tensorboard_dir=log_path)""",

        """
        net  = tflearn.input_data(shape=[None, 9])
        net = tflearn.fully_connected(net, 32, activation="relu")
        net = tflearn.fully_connected(net, 64, activation="relu")
        net = tflearn.fully_connected(net, 64, activation="relu")
        net = tflearn.fully_connected(net, 128, activation="relu")
        net = tflearn.fully_connected(net, 128, activation="relu") # add a layer
        net = tflearn.fully_connected(net, 256, activation="relu") # add a layer
        net = tflearn.fully_connected(net, 512, activation="relu") # add a layer
        net = tflearn.fully_connected(net, 512, activation="relu") # add a layer
        net = tflearn.fully_connected(net, 256, activation="relu") # add a layer
        net = tflearn.fully_connected(net, 256, activation="relu") # add a layer
        net = tflearn.fully_connected(net, 128, activation="relu") # add a layer
        net = tflearn.fully_connected(net, 128, activation="relu") # add a layer
        net = tflearn.fully_connected(net, 64, activation="relu")
        net = tflearn.fully_connected(net, 64, activation="relu")
        net = tflearn.fully_connected(net, 32, activation="relu")
        net = tflearn.fully_connected(net, 31, activation="softmax")
        net = tflearn.regression(net, learning_rate=0.0001) 
        tflearn.DNN(net, tensorboard_dir=log_path)""",

        """
        net  = tflearn.input_data(shape=[None, 9])
        net = tflearn.fully_connected(net, 32, activation="relu")
        net = tflearn.fully_connected(net, 64, activation="relu")
        net = tflearn.fully_connected(net, 64, activation="relu")
        net = tflearn.fully_connected(net, 128, activation="relu")
        net = tflearn.fully_connected(net, 128, activation="relu") # add a layer
        net = tflearn.fully_connected(net, 256, activation="relu") # add a layer
        net = tflearn.fully_connected(net, 256, activation="relu") # add a layer
        net = tflearn.fully_connected(net, 512, activation="relu") # add a layer
        net = tflearn.fully_connected(net, 512, activation="relu") # add a layer
        net = tflearn.fully_connected(net, 1024, activation="relu") # add a layer
        net = tflearn.fully_connected(net, 1024, activation="relu") # add a layer
        net = tflearn.fully_connected(net, 512, activation="relu") # add a layer
        net = tflearn.fully_connected(net, 512, activation="relu") # add a layer
        net = tflearn.fully_connected(net, 256, activation="relu") # add a layer
        net = tflearn.fully_connected(net, 256, activation="relu") # add a layer
        net = tflearn.fully_connected(net, 128, activation="relu") # add a layer
        net = tflearn.fully_connected(net, 128, activation="relu") # add a layer
        net = tflearn.fully_connected(net, 64, activation="relu")
        net = tflearn.fully_connected(net, 64, activation="relu")
        net = tflearn.fully_connected(net, 32, activation="relu")
        net = tflearn.fully_connected(net, 31, activation="softmax")
        net = tflearn.regression(net, learning_rate=0.0001) 
        tflearn.DNN(net, tensorboard_dir=log_path)""",
    ]

    # training and validating
    titles, predict_res, evaluates = train_validate_manager(train_x, train_y, validate_x, validate_y, \
                                titles, codes, get_dl_accuracy)
    
    return attr_names, label_names, train_fraction, train_x, train_y, \
            validate_x, validate_y, titles, predict_res, evaluates

def main():
    from classification import print_res
    attr_names, label_names, train_fraction, train_x, train_y, \
            validate_x, validate_y, titles, \
            predict_res, evaluate_res = train_validate(train_fraction=0.5)
    
    print_res(titles, evaluate_res)


if __name__ == '__main__':
    main()