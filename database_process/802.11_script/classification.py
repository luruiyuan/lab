# -*- coding: utf-8 -*-

# os.path.dirname returns the path of the directory.
# eg: /home/luruiyuan/python/Codes/machine learning/database_process/802.11_script
# and we can use relative path to import our own python file
import sys
import os
# __file__ is different. In Vscode, __file__ is the whole path
# in CentOS or Ubuntu, only contains the source file name
# sys.path.append(os.path.dirname(p=__file__)+'/../data_preprocess/') # add package for my Ubuntu
sys.path.append(os.path.abspath(".")+'/../data_preprocess/') # add package for centOS server

import db_process as db

from db_process import __check_db_is_set__ as checkdb, __check_space_in_column_name__ as checksp, __check_space_in_multi_column_names__ as checkmulsps
from str2num import str2int as str2num

# multiprocess_train_validate_manager
from multiprocessing import cpu_count, Pool, Manager
import os, time

import numpy as np
# import matplotlib.pyplot as plt
#@profile
def get_all_column_names_by_table(*, conn, database="alu", table="data"):
    """
    fetch all columns' names and return a list contains these names.
    """
    sql = "show columns from %s;" % checkdb(database, table)
    cursor = db.excute_has_resultset_sqls(conn, sql)
    column_names = [m["Field"] for m in cursor.fetchall()]
    cursor.close()
    return column_names

def del_item_in_list(item_list, *del_items, generate_copy=True):
    """
    1. Delete item in list directly and return it if generate_copy = Flase
    2. If generate_copy = True (by default), this funciton will return a copy of
    original list and delete items in the copy list instead of the original one.
    """
    return_list = item_list[:] if generate_copy else item_list
    for item in items:
        for _ in range(return_list.count(item)):
            return_list.remove(item)
    return return_list

#@profile
def get_attr_names_label_names_by_table(*, conn, database="alu", table="data", attr_include_columns=None, \
        attr_exculde_columns=None, label_include_columns=None, label_exclude_columns=None):
    """
    Return attribute names list and label names list.
    This function will put column names that start with 'label' into lsit of label names.
    Others will be put into lsit of attribute names.
    """
    print("generate attribute/label names...")
    # init
    a_in_set = None
    a_ex_set = None
    l_in_set = None
    l_ex_set = None

    if attr_include_columns is not None:
        a_in_set = set(attr_include_columns)
    if attr_exculde_columns is not None:
        a_ex_set = set(attr_exculde_columns)
    if label_include_columns is not None:
        l_in_set = set(label_include_columns)
    if label_exclude_columns is not None:
        l_ex_set = set(label_exclude_columns)
    
    # get all column names
    columns = get_all_column_names_by_table(conn=conn, database=database, table=table)

    attr_names, label_names = [], []
    # choose columns
    for col in columns:
        if (col.lower().startswith("label") or (l_in_set is not None and col in l_in_set)) and \
                (l_ex_set is None or col not in l_ex_set):
            label_names.append(col)
        elif a_ex_set is None or (a_ex_set is not None and col not in a_ex_set):
            attr_names.append(col)

    print("generate attribute/label names succeeded!")
    return attr_names, label_names

#@profile
def get_attr_value_label_value_by_table(*, conn, database="alu", table="data",
        cluster_columns = None, \
        attr_exculde_columns=None, label_exclude_columns=None):
    """
    Put all the data into 2d tensors.
    Cluster_columns must be a list even there is only one column in it.
    Return: 
        @param data_rows: all the training and testing data without labels
        @param labels_values: all the labels
        @param cluster_values: all the cluster values used to cluster data

    If attr_exculde_columns=None and label_exclude_columns=None, the whole columns will selected.
    Otherwise, the value of excludes columns in attributes set and label set will be dropped.

    Note: 
        cluster_columns is a set of column names that used to generate list 'cluster_values'.
        The 'cluster_values' list is used to cluster data later in the function named 
        'split_fraction_for_train_validate'. The length of 'cluster_values' must be equal to 
        the length of 'data_rows' in 'split_fraction_for_train_validate' so that to reduce the
        database access times. Thus, it is not apropriate to use set here.
    
    For example:
        Suppose we have 3 students: Sam, Amy, Tom. Their age are 12, 12, 15, and their weight are
        60 kg, 50kg, 30kg. The cluster_columns is ["name", "age"]. 
        The function 'split_fraction_for_train_validate'
        will divide data twice:
            1. For the first time, data will be divided into 3 sets: Sam's data, Amy's data, and Tom's data.
            2. For the second time, data will be divided into 2 sets: data of the sutdents whose age are 12,
               15 respectively.
    """

    data_rows, labels_values, cluster_values = [], [], []

    attr_names, label_names = get_attr_names_label_names_by_table(conn=conn,
                                    database=database, table=table,
                                    attr_exculde_columns=attr_exculde_columns,
                                    label_exclude_columns=label_exclude_columns)
    if cluster_columns is None:
        cluster_columns = label_names
    sql = "select * from %s;" % checkdb(database, table)
    cursor = db.excute_has_resultset_sqls(conn, sql)
    for m in cursor.fetchall():
        data_rows.append([m[name] for name in attr_names])
        labels_values.append([m[label] for label in label_names])
        cluster_values.append([m[id] for id in cluster_columns])
    cursor.close()
    return data_rows, labels_values, cluster_values

#@profile
def split_fraction_for_train_validate(train_fraction, clusters, data_rows, labels, index):
    """
    @Params:
        train_fraction: 0~1, represents the fraction of training set
        clusters: cluster indeies
        data_rows: datas in the shape of 2-d list
        labels: a list contains all the values of each label
        index: the index used to cluster data
    
    Return:
        @param train: The training set (2-d list)
        @param validate: The validation set (2-d list)

        Note:
            The shap of training and validation set are shown as follows:
                train: [[data_row, train_label], [data_row2, train_label2]...]
                validate: [[data_row, validate_label], [data_row2, validate_label2]...]

    Split the data set in to training set and validation set according to train_fraction,
    and cluster data according to specific cluster columns. 
    
    These cluster values is generated by fuction named
    'get_attr_value_label_value_by_table'
    """
    print("clustering data set according to cluster labels...")
    from collections import OrderedDict
    split_dict = OrderedDict()
    for clu, row, label in zip(clusters, data_rows, labels):
        cl = clu[index] # list to str
        if cl in split_dict:
            split_dict[cl].append([row, label])
        else:
            split_dict[cl] = [[row, label]]
    print("clustering data set finished!\n%d clusters generated!" % len(split_dict))

    # split data set
    train, validate = [], []
    for _, tensor in split_dict.items():
        train_num = int(len(tensor) * train_fraction)
        for i, sample in enumerate(tensor):
            if i < train_num:
                train.append(sample)
            else:
                validate.append(sample)

    return train, validate

#@profile
def get_classification_accuracy(*, predict_labels, correct_labels):
    print("start accuracy calculation...")
    accuracy = .0
    for pre, corr in zip(predict_labels, correct_labels):
        accuracy += 1 if pre == corr else 0
    accuracy /= len(predict_labels)
    print("accuracy calculation finished!\n  accuracy:", accuracy)
    return accuracy

def transpose(dataset):
    # np.transpose will convert number to str
    print("data is being transposed...")
    data = [[] for i in range(len(dataset[0]))]
    for row in dataset:
        for j, value in enumerate(row):
            data[j].append(value)
    print("data transposed succeeded!")        
    return data

#@profile
def type_transform(dataset):
    """
    transform data from str to number
    """
    print("data type is being transformed!")
    numbers = set([1,2,3,4,6,8,9]) # columns' indeies converted to number
    data = transpose(dataset)
    
    for i, column in enumerate(data):
        if i in numbers:
            for j, value in enumerate(column):
                if isinstance(value, (int, float)):
                    break # skip this this column of the table
                if value == '':
                    data[i][j] = 0
                else:
                    data[i][j] = float(int(value, 16)) if value.startswith('0x') else float(value)

    print("data type transform finished!")
    return transpose(data)

def array_split(data, split_num=5000):
    print("dividing dataset into %d per set..." % split_num)
    arrays = []
    tmp = []
    for i, row in enumerate(data):
        tmp.append(row)
        i += 1
        if i == len(data) or i % split_num == 0 and i > 0:
            arrays.append(tmp)
            tmp = []
    print("dividing dataset into %d per set finished!" % split_num)    
    return arrays

def array_merge(data):
    print("merging dataset into one piece...")
    merge = [d for array in data for d in array] # 先循环 array in data, 再循环 d in array
    print("merging finished!")
    return merge

def train_validate_worker(msg_que, train_x, train_y, validate_x, validate_y, clf_name, classifier, evaluate_func):
    """
    concurrent trainning, validation and evaluate
    """
    res, start = {"name": clf_name}, time.time()
    # train
    print(clf_name,"training...")
    print( classifier.fit(train_x, train_y) ) # 训练
    print(clf_name, "training finished! Time: %.3f seconds." % (time.time() - start))
    res["clf"] = classifier
    
    # validate
    start = time.time()
    print(clf_name,"predicting...")
    predict_label = list(classifier.predict(validate_x)) # 验证
    print(clf_name, "predicting finished! Time: %.3f seconds." % (time.time() - start))
    res["result"] = predict_label
    
    # evaluate
    start = time.time()    
    print(clf_name,"evaluating...")
    res["evaluate"] = evaluate_func(predict_labels=predict_label, correct_labels=validate_y)
    print(clf_name,"evaluating finished! Time: %.3f seconds." % (time.time() - start))
    
    # put res into message queue
    print("is here")
    msg_que.put(res)
    print("finished yes ")


def multiprocess_train_validate_manager(train_x, train_y, validate_x, validate_y, clf_names, classifiers, evaluate_func):
    """
    manage the concurrence of trainning, prediction and evaluate
    """
    # define a result queue
    q = Manager().Queue() # multiprocessing manager queue

    # calculate process time

    # init process pool
    pool = Pool(cpu_count())

    print("starting multiprocess_train_validate_manager...\nprocess pool size: %d clf number: %d" % (cpu_count(), len(clf_names)))
    whole_start = time.time()
    # multiprocessing
    for name, clf in zip(clf_names, classifiers):
        pool.apply_async(train_validate_worker, args=(q,train_x, train_y, validate_x, validate_y, name, clf, evaluate_func))
    # waiting for processing finished
    print("processing pool is closing!")
    pool.close()
    pool.join()
    
    whole_end = time.time()
    print("whole time for training, validation and evaluating: %.3f seconds." % (whole_end - whole_start))
    # gather results from result queue
    clf_names, clfs, predict_res, evaluates = [], [], [], []
    while not q.empty():
        res = q.get()
        clfs.append(res["clf"])
        predict_res.append(res["result"])
        clf_names.append(res["name"])
        evaluates.append(res["evaluate"])
    
    return clfs, clf_names, predict_res, evaluates



#@profile
def train_validate(*, conn=None, database="alu", table="data", classifier, clf_names, evaluate=get_classification_accuracy,\
        train_fraction=0.6, cluster_column_names=None, exclude_attr_columns=None, \
        exclude_label_columns=None, k_cross_validation=0):
    """
    Training model and validate the model.

    Return:
        attr_names: the column names choosen as attributes
        label_names: the column names choosen as categories
        train_fraction: the fraction used to split data set
        classifier: the classifier used to classify data
        train_x: list of trainning feature labels
        train_y: list of trainning category labels
        validate_x: list of validation feature labels
        validate_y: list of validation category labels
        evaluate_res: evaluate result. Usually a float between 0~1 describes the probability of correct identification.

    Param:
        conn: database connection
        databse: databse name
        table: table name
        classifier: the classifier specified to classify data
        evaluate: the evaluate function which will be called to evaluate the model we trained.
                  If the evaluate method is not specified, the default evaluate function named
                  'get_classification_accuracy' defined above will be called to evaluate the model.
        train_fraction: the fraction describes the number of data used to train
        cluster_column_names: the columns used to cluster data. Further explanation can be found in
                  the doc of function 'get_attr_value_label_value_by_table' defined above.
        exclude_attr_columns: columns that shouldn't be condidered as attributes. None is OK.
        exclude_label_columns: columns that shouldn't be condidered as labels. None is OK.
        
        Note:
            The defination of parameters of evaluate function must be defined the same as follos:
            
                function_name(*, predict_labels, correct_labels)
            
            For the evaluate will be called as follows:
                evaluate_res = evaluate(predict_labels=predict_label, correct_labels=validate_y)
    """

    in_flag = False
    if conn is None or not conn.open:
        conn = db.create_connection()
        in_flag = True # flag if the connection is opened in this function
    
    # get attributes column names and label column names
    attr_names, label_names = get_attr_names_label_names_by_table(conn=conn, database=database,
                                    table=table, attr_exculde_columns=exclude_attr_columns,
                                    label_exclude_columns=exclude_label_columns)
    # get all the data and cluster columns
    data_rows, labels_values, cluster_values = get_attr_value_label_value_by_table(conn=conn, database=database, \
                                                    table=table, cluster_columns=cluster_column_names, \
                                                    attr_exculde_columns=exclude_attr_columns, \
                                                    label_exclude_columns=exclude_attr_columns)

    # close connection
    if in_flag:
        db.close_connection()

    # transform data from str type to their original type
    data_rows = type_transform(data_rows)

    # split data set for training and validating
    trains, validates = split_fraction_for_train_validate(train_fraction, cluster_values, data_rows, cluster_values, 1)
    
    # split feature labels and category labels
    # x is the list of feature labels, and y is the list of category labels
    train_x, train_y = [t[0] for t in trains], [t[1][1] for t in trains]
    validate_x, validate_y = [v[0] for v in validates], [v[1][1] for  v in validates]
    
    # normalize trainning set and validate set 归一化训练集和验证集
    train_n_x, validate_n_x = normalization(train_x, validate_x)

    # train and validate using multiprocessing
    clfs, clf_names, predict_res, evaluates = multiprocess_train_validate_manager(train_n_x, \
                                                train_y, validate_n_x, validate_y, clf_names, \
                                                classifier, evaluate)

    return attr_names, label_names, train_fraction, train_x, train_y, \
            validate_x, validate_y, clf_names, clfs, predict_res, evaluates

#@profile
def max_min_normalization(data):
    """
    normalization dataset.
    If the dataset contains str, it will be transformed to float between 0~1
    """
    data_set = []
    for d in data:
        tp_data = transpose(d) # transposed data
        hash = {}
        for col in tp_data:
            if isinstance(col[0], (int, float)):
                    continue
            cols = list(set(col))
            rg = len(cols)
            for i, j in enumerate(cols):
                hash[j] = i + 1
            for i, d in enumerate(col): # transform str to float
                col[i] = float(hash[d])/ rg
        
        data_set.append(transpose(tp_data))
    return data_set

#@profile
def normalization(*data, normalize_method="max_min"):
    print("data normalization...")
    res =  eval(normalize_method+"_normalization(data)") # dynamically call normalization
    print("normalization finished!")
    return res

#@profile
def init_classifiers():
    """
    Return classifiers names and classifiers
    """
    print("import pakgs...")
    # import matplotlib.pyplot as plt
    from matplotlib.colors import ListedColormap
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler
    from sklearn.datasets import make_moons, make_circles, make_classification
    from sklearn.neural_network import MLPClassifier
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.svm import SVC, LinearSVC
    from sklearn.gaussian_process import GaussianProcessClassifier
    from sklearn.gaussian_process.kernels import RBF
    from sklearn.tree import DecisionTreeClassifier
    from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
    from sklearn.naive_bayes import GaussianNB
    from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
    print("import done!")

    print("init classifers...")
    # svc_default = SVC()
    # C = 1.0 SVM regularization parameter
    # svc = SVC(kernel='linear', C=C)
    # rbf_svc = SVC(kernel='rbf', gamma=0.7, C=C)
    # poly_svc = SVC(kernel='poly', degree=3, C=C)
    # lin_svc = LinearSVC(C=C)

    # title for the plots

    names = ["Nearest Neighbors",
        # "Linear SVM", 
        "RBF SVM", 
        # "Gaussian Process",
        "Decision Tree", 
        "Random Forest", 
        "Neural Net", 
        "AdaBoost",
        "Naive Bayes", 
        "QDA"]

    # titles = ['my_SVC with default settings',
    #         # 'my_SVC with linear kernel',
    #         # 'my_LinearSVC (linear kernel)',
    #         'my_SVC with RBF kernel',
    #         'my_SVC with polynomial (degree 3) kernel']
    
    # names.extend(titles)

    # add classifiers to list
    classifiers = [
    KNeighborsClassifier(30),
    # SVC(kernel="linear", C=0.025),
    SVC(gamma=2, C=1),
    # GaussianProcessClassifier(1.0 * RBF(1.0), warm_start=True),
    DecisionTreeClassifier(max_depth=5),
    RandomForestClassifier(max_depth=5, n_estimators=10, max_features=1),
    MLPClassifier(alpha=1),
    AdaBoostClassifier(),
    GaussianNB(),
    QuadraticDiscriminantAnalysis()]

    # classifiers.append(svc_default)
    # classifiers.append(svc)
    # classifiers.append(lin_svc)    
    # classifiers.append(rbf_svc)
    # classifiers.append(poly_svc)
    print("init classifers done!")

    return names, classifiers

def print_res(classifier_names, evaluate_results):
    for c_name, res in zip(classifier_names, evaluate_results):
        print(c_name, res)


def main():
    # initialize
    names, classifiers = init_classifiers()

    # train and validate
    attr_names, label_names, train_fraction, train_x, train_y, validate_x, \
        validate_y, clf_names, clfs, predict_res, \
        evaluate_res = train_validate(train_fraction=0.6, classifier=classifiers, \
        clf_names=names, exclude_attr_columns=["Time", "Protocol", " Source address", "Destination", "TSF timestamp", "Qos Control Field"])
    
    print_res(clf_names, evaluate_res)

if __name__ == '__main__':
    main()