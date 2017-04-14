# -*- coding: utf-8 -*-

# os.path.dirname returns the path of the directory.
# eg: /home/luruiyuan/python/Codes/machine learning/database_process/802.11_script
# and we can use relative path to import our own python file
import sys
import os
print(os.path.dirname(p=__file__))
# __file__ is different. In Ubuntu, __file__ is the whole path
# in CentOS, only contains the source file name
# sys.path.append(os.path.dirname(p=__file__)+'/../data_preprocess/') # add package for my computer
sys.path.append(os.path.abspath(".")+'/../data_preprocess/') # add package for server

import db_process as db

from db_process import __check_db_is_set__ as checkdb, __check_space_in_column_name__ as checksp, __check_space_in_multi_column_names__ as checkmulsps
from str2num import str2int as str2num

import numpy as np
# import matplotlib.pyplot as plt

dicts = None

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

def get_attr_value_label_value_by_table(*, conn, database="alu", table="data",
        cluster_columns = ["Source address"], \
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
    sql = "select * from %s;" % checkdb(database, table)
    cursor = db.excute_has_resultset_sqls(conn, sql)
    for m in cursor.fetchall():
        data_rows.append([m[name] for name in attr_names])
        labels_values.append([m[label] for label in label_names])
        cluster_values.append([m[id] for id in cluster_columns])
    
    return data_rows, labels_values, cluster_values

def split_fraction_for_train_validate(train_fraction, clusters, data_rows, labels):
    """
    @Params:
        train_fraction: 0~1, represents the fraction of training set
        clusters: cluster indeies
        data_rows: datas in the shape of 2-d list
        labels: a list contains all the values of each label
    
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
        cl = clu[0] # list to str
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

def convert2num(tensors, *, trans_func=str2num):
    """
    transform object to num. Users must specify how to transform.
    """

    res = tensors[:]

    global dicts
    dicts = [dict() for _ in res[0]] # dict for each column

    print("spliting data set...")
    for n in res:
        for i, value in enumerate(n):
            if isinstance(value, str):
                if value.isdigit():
                    n[i] = float(value)
                else:
                    n[i] = trans_func(value, dicts[i])
    print("spliting finished!")
    return res # return a 2-d tensor

def get_classification_accuracy(*, predict_labels, correct_labels):
    print("start accuracy calculation...")
    accuracy = .0
    for pre, corr in zip(predict_labels, correct_labels):
        accuracy += 1 if pre == corr else 0
    accuracy /= len(predict_labels)
    print("accuracy calculation finished!\n  accuracy:", accuracy)
    return accuracy

def train_validate(*, conn=None, database="alu", table="data", classifier, clf_names, evaluate=get_classification_accuracy,\
        train_fraction=0.6, cluster_column_names=["Source address"], exclude_attr_columns=None, \
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
                                                    table=table, cluster_columns=cluster_column_names,
                                                    attr_exculde_columns=exclude_attr_columns,
                                                    label_exclude_columns=exclude_attr_columns)
    # split data set for training and validating
    trains, validates = split_fraction_for_train_validate(train_fraction, cluster_values, data_rows, cluster_values)
    
    # split feature labels and category labels
    # x is the list of feature labels, and y is the list of category labels
    train_x, train_y = convert2num([t[0] for t in trains]), [t[1][0] for t in trains]
    validate_x, validate_y = convert2num([v[0] for v in validates]), [v[1][0] for v in validates]
    
    # 归一化
    train_x = normalization(train_x)

    evaluate_results = []
    if not isinstance(classifier, list):
        classifier = [classifier]
    for clf, c_name in zip(classifier, clf_names):
        # train
        print(c_name,"training...")        
        print( clf.fit(train_x, train_y) )
        print(c_name, "predicting finished!")
        print("training finished!")
        
        # validate
        print(c_name,"predicting...")
        predict_label = list(clf.predict(validate_x))
        print(c_name, "predicting finished!")
        
        # evaluate model
        print(c_name,"evaluating...")
        evaluate_res = evaluate(predict_labels=predict_label, correct_labels=validate_y)
        evaluate_results.append(evaluate_res)
        print(c_name,"evaluating finished!")

    if in_flag:
        db.close_connection()
    return attr_names, label_names, train_fraction, classifier, train_x, train_y, validate_x, validate_y, evaluate_results

def get_max_min(x, *, min_func=None, max_func=None, value2num_func=str2num):
    """
    return 2 lists. The max and min of attributes in x (x must be a 2-d tensor)
    """
    max_list, min_list = [], []

    data = np.transpose(x) # 矩阵转置
    for attrs in data:
        # values = list(map(value2num_func, attrs)) # 直接哈希时使用

        values = attrs # map 计数时使用
        min_attr, max_attr = min(values), max(values)

        max_list.append(max_attr)
        min_list.append(min_attr)
    
    return max_list, min_list

def max_min_normalization(data, value2num_func):
    """
    极值归一化
    data 必须为2维张量, 形式如同数据库, 每行为一个记录
    返回归一化之后的数据
    """
    from collections import OrderedDict as Dict
    max_min = Dict

    min_attrs, max_attrs = get_max_min(data)
    data_tran = np.transpose(data)

    for diction, max_num, min_num, col in zip(dicts, min_attrs, max_attrs, data_tran):
        rg = max_num - min_num
        for i, d in enumerate(col):
            col[i] = 1.0 if rg == 0 else float(value2num_func(d, diction) - min_num) / rg
    
    return np.transpose(data_tran)

def normalization(data, normalize_method="max_min", value2num_func=str2num):
    print("data normalization...")
    res =  eval(normalize_method+"_normalization(data, value2num_func)") # dynamically call normalization
    print("normalization finished!")
    return res


def init_classifiers():
    """
    Return classifiers names and classifiers
    """
    print("import pakgs...")
    import numpy as np
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
    svc_default = SVC()
    C = 1.0  # SVM regularization parameter
    svc = SVC(kernel='linear', C=C)
    rbf_svc = SVC(kernel='rbf', gamma=0.7, C=C)
    poly_svc = SVC(kernel='poly', degree=3, C=C)
    lin_svc = LinearSVC(C=C)

    # title for the plots

    names = ["Nearest Neighbors", "Linear SVM", "RBF SVM", "Gaussian Process",
         "Decision Tree", "Random Forest", "Neural Net", "AdaBoost",
         "Naive Bayes", "QDA"]

    titles = ['my_SVC with default settings',
            'my_SVC with linear kernel',
            'my_LinearSVC (linear kernel)',
            'my_SVC with RBF kernel',
            'my_SVC with polynomial (degree 3) kernel']
    
    names.extend(titles)

    # add classifiers to list
    classifiers = [
    KNeighborsClassifier(3),
    SVC(kernel="linear", C=0.025),
    SVC(gamma=2, C=1),
    GaussianProcessClassifier(1.0 * RBF(1.0), warm_start=True),
    DecisionTreeClassifier(max_depth=5),
    RandomForestClassifier(max_depth=5, n_estimators=10, max_features=1),
    MLPClassifier(alpha=1),
    AdaBoostClassifier(),
    GaussianNB(),
    QuadraticDiscriminantAnalysis()]

    classifiers.append(svc_default)
    classifiers.append(svc)
    classifiers.append(rbf_svc)
    classifiers.append(poly_svc)
    classifiers.append(lin_svc)
    print("init classifers done!")

    return names, classifiers

def print_res(classifier_names, evaluate_results):
    for c_name, res in zip(classifier_names, evaluate_results):
        print(c_name, res)


def test():
    x = [[1, 2, 3, "abc"], ["ack", "heheda","megmegd"], [4,5,6,"def"]]
    print(normalization(x))
    # # y = x[:, :2]
    # print("hehe", x[:,0])
    # print(x[:,0].max())

    # print(np.shape(max_min_normalization(x)))
    # print(np.shape(normalization(x)))

def main():
    # initialize
    names, classifiers = init_classifiers()

    # train and validate
    attr_names, label_names, train_fraction, classifier, train_x, train_y, validate_x, \
        validate_y, evaluate_res = train_validate(train_fraction=0.8, classifier=classifiers, \
                clf_names=names, exclude_attr_columns=["Time", "Source address", "TSF Timestamp"])
    
    print_res(names, evaluate_res)

if __name__ == '__main__':
    main()
    # test()    