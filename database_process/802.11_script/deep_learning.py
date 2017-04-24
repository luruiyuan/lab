# coding: utf-8
# tensorflow/titantic

from __future__ import absolute_import, division, print_function

from classification import get_data_label as get_data, get_classification_accuracy as accuracy

import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='3'

import tflearn
tflearn.init_graph(num_cores=8, gpu_memory_fraction=0.6)

import time

def gen_nets_by_config(config):
    # read cofiguration
    dropout = config['dropout']
    hidden_act = config['hidden_act']
    hidden_layer_num = config['hidden_layer_num']
    out_act = config['out_act']
    label_num = config['label_num']
    larning_rate = config['larning_rate']
    optimizer = config['optimizer']
    node_num = config['node_num']
    loss = config['loss']
    shape = config['shape']
    
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
        # execute
        exec("net = tflearn.fully_connected(net, %d, activation='%s')" % (node_num, hidden_act))
        exec("net = tflearn.dropout(net, %f)" % (dropout))

    net = tflearn.fully_connected(net, label_num, activation=out_act)
    net = tflearn.regression(net, learning_rate=larning_rate, loss=loss, optimizer=optimizer)
    
    return tflearn.DNN(net)

def title2model_conf(title):
    l = title.split('_')
    hidden_layer = int(l[0])
    shape = [None, 9]
    dropout = 0.2
    hidden_activation = l[2]
    out_activation = l[-1]
    label_num = 31
    larning_rate = 0.00015
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
        "larning_rate": larning_rate,
        "optimizer": optimizer,
        "node_num": node_num,
        "loss": loss,
    }

def build_neural_network_model(label_num):
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
    print("dynamiclly generating %d nets..." % len(titles))
    models = list(map(gen_nets_by_config, list(map(title2model_conf, titles)))) # generate model from title
    print("generating %d nets succeeded!" % len(titles))
    
    return titles, models



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

# # get data
# attr_names, label_names, train_fraction, \
#     train_x, train_y, validate_x, validate_y = get_data(exclude_attr_columns=["Time", "Protocol", "Source address", "Destination", "TSF timestamp", "Qos Control Field"])

# # define model
# titles, models = build_neural_network_model(31)

# # transform label data into required type
# look_up_table = assign_id_for_label(train_y) # create look up table for each label
# label_list = get_label_list(look_up_table) # using this we can find label by id

# train_y = transform_label_to_vector(train_y, look_up_table) # transform label type from str to vector
# validate_y = transform_label_to_vector(validate_y, look_up_table)

# model.fit(train_x, train_y, n_epoch=1, batch_size=128, show_metric=True)
# pred = model.predict(validate_x)

# print(pred[0])

# print("saving model...")
# model.save("model")
# print("saving finished!")

def get_timestamp():
    return time.strftime('%Y-%m-%d_%H:%M:%S', time.localtime())
    
def save_models(*, titles=[], models=[]):
    """
    Saving models and return the root dir path of models
    """
    from os import path, makedirs
    print("saving %d models..." % len(models))
    timestamp = get_timestamp()
    root = path.join("./models/", get_timestamp())  # root dir
    
    if not isinstance(titles, list):
        titles = [titles]
    if not isinstance(models, list):
        models = [models]
    
    for title, model in zip(titles, models):
        relative_path = path.join(root, title)
        makedirs(relative_path) # create dir
        model.save(path.join(relative_path, title)) # save model
    
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

def get_dl_accuracy(*, predic_labels, correct_labels): 
    print("start deep learning accuracy calculation...")
    accuracy = .0
    for pre, corr in zip(predict_labels, correct_labels):
        accuracy += 1 if list(pre).index(1) == corr.index(1) else 0 # predict labels has ndarrays
    accuracy /= len(predict_labels)
    print("accuracy calculation finished!\n  accuracy:", accuracy)
    return accuracy

def train_validate_worker(msg_que, train_x, train_y, validate_x, validate_y, title, model, evaluate_func):
    res = {"title": clf_name}
    # train
    print(title,"training...")
    model.fit(train_x, train_y, n_epoch=1, batch_size=128, show_metric=True)
    print(title, "training finished!")
    res["model"] = model

    #validate
    print(title,"predicting...")
    predict_labels = model.predict_label(validate_x)
    print(title,"predicting finished!")
    res["result"] = predict_labels

    # evaluate
    print(title,"evaluating...")
    res["evaluate"] = evaluate_func(predict_labels=predict_label, correct_labels=validate_y)
    print(title,"evaluating finished!")
    
    # put res into message queue
    msg_que.put(res)

def train_validate_manager(train_x, train_y, validate_x, validate_y, titles, models, evaluate_func):
    from multiprocessing import cpu_count, Pool, Manager
    import time

    # define a result queue
    q = Manager().Queue() # multiprocessing manager queue

    # init process pool
    pool = Pool(cpu_count())

    print("starting multiprocess_train_validate...\nprocess pool size: %d model number: %d" % (cpu_count(), len(titles)))
    whole_start = time.time()
    # multiprocessing
    for title, model in zip(titles, models):
        pool.apply_async(train_validate_worker, args=(q,train_x, train_y, validate_x, validate_y, title, model, evaluate_func))
    # waiting for processing finished
    print("processing pool is closing!")
    pool.close()
    pool.join()
    
    whole_end = time.time()
    print("whole time for training, validation and evaluating: %.3f seconds." % (whole_end - whole_start))
    # gather results from result queue
    titles, dl_models, predict_res, evaluates = [], [], [], []
    while not q.empty():
        res = q.get()
        dl_models.append(res["model"])
        predict_res.append(res["result"])
        titles.append(res["name"])
        evaluates.append(res["evaluate"])
    
    return dl_models, titles, predict_res, evaluates

def train_validate():
    # get data
    attr_names, label_names, train_fraction, \
        train_x, train_y, validate_x, validate_y = get_data(exclude_attr_columns=["Time", "Protocol", "Source address", "Destination", "TSF timestamp", "Qos Control Field"])

    # define structure of models
    titles, models = build_neural_network_model(31)

    # save models
    save_models(titles=titles, models=models)

    # transform label data into required type
    look_up_table = assign_id_for_label(train_y) # create look up table for each label
    label_list = get_label_list(look_up_table) # using this we can find label by id

    train_y = transform_label_to_vector(train_y, look_up_table) # transform label type from str to vector
    validate_y = transform_label_to_vector(validate_y, look_up_table)

    # training and validating
    dl_models, titles, predict_res, evaluates = train_validate_manager(train_x, train_y, validate_x, validate_y, \
                                titles, models, get_dl_accuracy)
    
    return attr_names, label_names, train_fraction, train_x, train_y, \
            validate_x, validate_y, titles, dl_models, predict_res, evaluates

def main():
    from classification import print_res
    attr_names, label_names, train_fraction, train_x, train_y, \
            validate_x, validate_y, titles, dl_models, \
            predict_res, evaluates = train_validate()
    
    print_res(titles, evaluate_res)


if __name__ == '__main__':
    main()