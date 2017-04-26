d = {"x": "hehe", "y":"xixi"}
l,n = ["1,2,3"],["4,5,6"]
for i, j in zip(l,n):
    print(i, " ", j)

x=1
print ((3 if x==1 else 2) + 2)

s = ("this is %s" + "这是 %d") %("bullshit", 5)
print(s)

k = [l,n]
x1 = ['1,2,3','4,5,6']
x1.append("7+8+9")
print("k:",k)
print(",".join(['1,2,3','4,5,6']))
print(x1)
x = 1
def res(y):
    return y
print(res(5))

x,y = (4,5) if False else (5,4)
print(x,y)

x1.extend(k)
print("hehe", "xixi")

["1,2,3"],["4,5,6"]

m = {"123": "456"}
print("123" in m)
print("456" in m)
print((ord('b') - ord('a')))

s = set([1,2,3])
print(s)
print(3 is not None)

s = "Ab1234"
print(s.lower().startswith("ab12"))

s = ["123","234"]
print(", ".join(s))

def test():
    return 1,2,3

a = test()
print(a)

from collections import OrderedDict as od
a = od()
for i in range(10):
    a[i] = {i:"i"}
print(a)
print(len(a))

a = b = -1
print(a, b)

a, b = (1,1) if False else (2,1)

print(a, b)

a = set

import os
import sys
print(os.getcwd())
print(os.path.abspath("."))
print(__file__)
print("os.path.dirname(p=__file__)", os.path.dirname(p=__file__))
s = __file__
print("%r" %s)

print(os.path.abspath("./database_process"))
s = os.path.dirname(p=__file__)+'/data_preprocess/'
print(os.path.abspath(s))
print(". path:", os.path.abspath("."))
print("s:",s)
print("__file__:", __file__) # __file__ 在不同系统上是不同的
sys.path.append(s)
import db_process
print("ok")

def array_split(data, split_num=5000):
    print(split_num)
    arrays = []
    tmp = []
    for i, row in enumerate(data):
        tmp.append(row)
        i += 1
        if i == len(data) or i % split_num == 0 and i > 0:
            arrays.append(tmp)
            tmp = []
    return arrays


data = [
    [1,2,3],
    [4,5,6],
    [7,8,9]
]
split = array_split(data, 2)
print(split)

def array_merge(data):
    print("merging dataset into one piece...")
    merge = [d for array in data for d in array]
    print("merging finished!")
    return merge    

merge = array_merge(split)
print(merge)

# 测试并发编程
# from multiprocessing import Process
# import os

# def run_proc(name):
#     print("Run child process %s (%s)..." % (name, os.getppid()))

# if __name__ == "__main__":
#     print("Parent proccess %s." % os.getppid())
#     p = Process(target=run_proc, args=('test',))
#     print("child will start.")
#     p.start()
#     p.join()
#     print("child process end.")

# 测试进程池
from multiprocessing import Pool
import os, time, random

def long_time_task(name):
    print("在子进程中")
    raise Exception("hehe")
    print("Run task %s (%s)..." % (name, os.getpid()))
    start = time.time()
    time.sleep(random.random() * 3) # 随机延时一定长度
    end = time.time()
    print("Task %s runs %.2f seconds." % (name, end - start))
    return name + "cao"

if __name__ == "__main__":
    print("Parent process %s." % os.getpid())
    p = Pool(4)
    for i in range(5):
        p.apply_async(long_time_task, args=(i, ))
    p.close()
    p.join()
    print("all subprocesses done.")

import os
from multiprocessing import cpu_count, Queue
q = Queue()
cpu = cpu_count()
global results
pool = Pool(cpu)

# for i in range(cpu):
#     pass

# print("cpu 数目:",cpu_count())
# q.put(["1,2,3"])
# q.put(["1"])

# print(q.get())
# print(q.get_nowait())

d = {j: i + 1 for (i, j) in enumerate(set([-1,-2,-3]))}
print(d)
print(len(d))
print([d[i] for i in [-1,-2,-2,-3]])

data = ["111","222","333"]
look_up_table = {j: i for (i, j) in enumerate(set(data))}
print("look_up_table:",look_up_table)

x = [[1 if look_up_table[d] == k else 0 for k in range(len(look_up_table))] for d in data]
print(x)
# label_vectors = [1 if look_up_table[j] == k else 0 for d in data for k in len(look_up_table)]
# print(label_vectors)
x = [0.5,0.5]
print(x.index(max(x)))

def generate_multi_hidden_layer_nets_by_title(title):
    import tflearn
    t = title.split('_')
    layer_num, hid_act, out_act, label_num = int(t[0]), t[2], t[-3], int(t[-1])
    print(layer_num, hid_act, out_act, label_num)
    node_num = 16
    h = "net = tflearn.fully_connected(net, %d, activation='%s')" % (node_num, hid_act)
    net = eval("tflearn.fully_connected(net, %d, activation='%s')" % (label_num, out_act))
    print(o)
    print(s)
    net = tflearn.regression(net, learning_rate=0.0001)
    # return tflearn.DNN(net)

# generate_multi_hidden_layer_nets_by_title("3_hidden_relu_out_sofmax_labelnum_5")

def title2model_conf(title):
    l = title.split('_')
    hidden_layer = int(l[0])
    shape = [None, 9]
    dropout = 0.2
    hidden_activation = l[2]
    out_activation = l[-1]
    label_num = 31
    larning_rate = 0.0002
    node_num = 16
    optimizer = 'adma'
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

print(list(map(title2model_conf, ["3_hidden_relu_out_sofmax"])))

conf = title2model_conf("3_hidden_relu_out_sofmax")
print("%f" % 0.34)


def gen_nets_by_config(config):
    import tflearn
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
        print("net = tflearn.fully_connected(net, %d, activation='%s')" % (node_num, hidden_act))
        print("net = tflearn.dropout(net, %f)" % (dropout))
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
    larning_rate = 0.0002
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

# conf = title2model_conf("3_hidden_relu_out_softmax")
# model = gen_nets_by_config(conf)

import time
s = time.strftime('%m-%d_%H:%M:%S', time.localtime())
print(s)

l = dir()
print(l)

# import tflearn
# net = tflearn.input_data(shape=[None, 1])
# net = tflearn.fully_connected(net, 32, activation="relu")
# net = tflearn.fully_connected(net, 64, activation="relu")
# net = tflearn.fully_connected(net, 128, activation="relu")
# net = tflearn.fully_connected(net, 64, activation="relu")
# net = tflearn.fully_connected(net, 32, activation="relu")
# net = tflearn.fully_connected(net, 2, activation="softmax")
# net = tflearn.regression(net, learning_rate=0.0009)
# model = tflearn.DNN(net)

# train_x = [[1],[2]]
# train_y = [[1,0],[0,1]]

# model.fit(train_x, train_y, n_epoch=1000, batch_size=128, show_metric=True)


# model.load("./database_process/802.11_script/model")
# model.save("./database_process/802.11_script/model")

# input(">")
# res = model.predict_label([[1],[2],[1]])
# print("res:", res)
# for i in res:
#     print(list(i))
# input("<")

def get_timestamp():
    import time
    return time.strftime('%Y-%m-%d_%H:%M:%S', time.localtime())


def save_models(*, titles=[], models=[]):
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

# print(save_models(titles=["test","hehe"], models=[model, model]))

# net = tflearn.fully_connected(net, 32, activation="relu")
# net = tflearn.fully_connected(net, 64, activation="relu")
# net = tflearn.fully_connected(net, 128, activation="relu")
# net = tflearn.fully_connected(net, 64, activation="relu")
# net = tflearn.fully_connected(net, 32, activation="relu")
# net = tflearn.fully_connected(net, 31, activation="softmax")
# net = tflearn.regression(net, learning_rate=0.0001)

# return tflearn.DNN(net)


print(os.path.join("/usr/hehe", "title/"))

def get_dl_accuracy(*, predict_labels, correct_labels): 
    print("start deep learning accuracy calculation...")
    accuracy = .0

    for pre, corr in zip(predict_labels, correct_labels):
        pre_list = list(pre)
        accuracy += 1 if pre_list.index(max(pre_list)) == corr.index(1) else 0 # predict labels has ndarrays
    accuracy /= len(predict_labels)

    print("accuracy calculation finished!\n  accuracy:", accuracy)
    return accuracy


def train_validate_worker(msg_que, train_x, train_y, validate_x, validate_y, title, model, evaluate_func, epoch, batch_size, timestamp):
    
    
    res = {"title": title}
    # train model: model cannot be transfered to parent process
    print(title,"training...")
    model.fit(train_x, train_y, n_epoch=epoch, batch_size=batch_size, show_metric=True)
    print(title, "training finished!")

    #validate
    print(title,"predicting...")
    predict_labels = model.predict(validate_x)

    pre = model.predict(train_x)
    evaluate_func(predict_labels=pre, correct_labels=train_y)

    print(title,"predicting finished!")
    res["result"] = predict_labels

    # evaluate
    print(title,"evaluating...")
    res["evaluate"] = evaluate_func(predict_labels=predict_labels, correct_labels=validate_y)

    # res["evaluate"] = evaluate_func(predict_labels=predict_labels, correct_labels=train_y)
    print(title,"evaluating finished!")
    
    # put res into message queue
    msg_que.put(res)

s = """
print("shit");
print("fuck);
"""

s = "print(\"shit\");"
s1 = "print('fuck')"


s =  """
net  = tflearn.input_data(shape=[None, 9])
net = tflearn.fully_connected(net, 32, activation="relu")
net = tflearn.fully_connected(net, 64, activation="relu")
net = tflearn.fully_connected(net, 128, activation="relu")
net = tflearn.fully_connected(net, 128, activation="relu") # add a layer
net = tflearn.fully_connected(net, 64, activation="relu")
net = tflearn.fully_connected(net, 32, activation="relu")
net = tflearn.fully_connected(net, 31, activation="softmax")"""
print(s.split('\n'))

for t in s.split('\n'):
    exec(t)