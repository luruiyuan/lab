# -*- coding: utf-8 -*-
import pymysql as pm

__CONNECTION__ = None

def __check_space_in_multi_column_names__(*columns):
    # Test case
    # col = ", ".join(checkmulsps(*attr_names)) + ", " + ", ".join(checkmulsps(*label_names))

    if isinstance(columns, str):
        return __check_space_in_column_name__(columns)
    cols = []
    for col in columns:
        cols.append(__check_space_in_column_name__(col))
    return cols

def __check_space_in_column_name__(column):
    """
    check if there is ' ' in column name
    if column name has ' ', add ``
    """
    if column is None:
        return None
    return column if ' ' not in column and '/' not in column else '`' + column + '`'

def __check_db_is_set__(db, table):
    """
    check if there is ' ' in column name
    if column name has ' ', add ``
    return modified string
    """
    return table if db is None else " %s.%s " % (db, table)

def create_connection(*, host="localhost",port=3306, user="root", passwd="root", charset='utf8', cursorclass=pm.cursors.DictCursor):
    """
    start a new connection.

    Test case for create_connection
    # conn = create_connection(host="localhost",port=3306, user="root", passwd="root")
    # print("connection: ", conn)
    # print("isopen: ", conn.open)
    # conn1 = create_connection(host="localhost",port=3306, user="root", passwd="root")
    # print("connection: ", conn1)
    # print("isopen: ", conn1.open)
    """
    global __CONNECTION__ # modify global var
    if __CONNECTION__ is None or not __CONNECTION__.open: # if connection is open, the field open is true
        print("createing connection...")
        __CONNECTION__ = pm.connect(host=host, port=port, user=user, passwd=passwd, charset=charset, cursorclass=cursorclass)
        print("create succeeded!")    
    return __CONNECTION__

def get_connection():
    """
    Return the connection. If the connection is not initialized, users should init manualy by
    'create_connection' funciton.
    """
    return __CONNECTION__

def close_connection():
    if __CONNECTION__ is not None and __CONNECTION__.open:
        __CONNECTION__.commit()
        print("connection is closing...")
        __CONNECTION__.close()
        print("connection is closed succeeded!")
    
    else:
        print("connection has been closed!")


def excute_update_sqls(conn, *sqls, batch_size=16):
    """
    Excute sqls, commit changes and close cursor finally.
    If you need to get resultSet, use 'excute_has_resultset_sqls' which will not close cursor.

    Note that * sqls is a mutable parameter, thus users cannot just use a list.
        If you want to use list, please use *list
    
    For example:
        sqls = ["select * from sh.rawdata", "select * from sh.rawdata", "select * from sh.rawdata"]
        excute_update_sqls(create_connection(host="localhost",port=3306, user="root", passwd="root"), *sqls) # * is needed
    Otherwise the each sql will be converted to a list: ["select * from sh.rawdata"],
    which causes TypeError: can't concat bytes to list.

    Test case:
    # sql = "select * from alu.raw_data"
    # sqls = []
    # for i in range(3):
    #     sqls.append(sql)
    #     print(sqls[i])
    # excute_update_sqls(create_connection(host="localhost",port=3306, user="root", passwd="root"), *sqls)
    """
    print("excuting update sqls ...")
    with conn.cursor() as cursor:
        if isinstance(sqls, str):
            cursor.execute(sql)
        else:
            for i, sql in enumerate(sqls):
                print(sql)
                cursor.execute(sql)
                if i % batch_size == 0:
                    conn.commit()
    conn.commit()
    print("excuting update succeeded")

def excute_has_resultset_sqls(conn, *sqls, batch_size=16):
    """
    Excute select sqls and will not close cursor, the cursor will be returned.
    If you don't need to get resultSet, use 'excute_update_sqls' which will close cursor automatically.

    Note that * sqls is a mutable parameter, thus users cannot just use a list.
        If you want to use list, please use *list
    
    For example:
        sqls = ["select * from sh.rawdata", "select * from sh.rawdata", "select * from sh.rawdata"]
        excute_has_resultset_sqls(create_connection(host="localhost",port=3306, user="root", passwd="root"), *sqls) # * is needed
    Otherwise the each sql will be converted to a list: ["select * from sh.rawdata"],
    which causes TypeError: can't concat bytes to list.

    Test case:
    # sql = "select * from alu.raw_data"
    # sqls = []
    # for i in range(3):
    #     sqls.append(sql)
    #     print(sqls[i])
    # excute_has_resultset_sqls(create_connection(host="localhost",port=3306, user="root", passwd="root"), *sqls)
    """
    print("excuting select sqls ...")
    cursor = conn.cursor()
    if isinstance(sqls, str):
        cursor.execute(sql)
    else:
        for i, sql in enumerate(sqls):
            print(sql)
            cursor.execute(sql)
            if i % batch_size == 0:
                conn.commit()
    conn.commit()
    print("excuting select succeeded")
    return cursor

def generate_update_sqls(table,*, db=None, **kw):
    """
    return a list contains the sqls used to update values of specific columns. Users must specify following parameters:
    @param string table: the name of the table
    @param dict kw: each condition is represented by the key of the dict, and value is a dict which
        contains the column and the value

        Note: 
            1.If the key starts with "None" (a string None, but not the Object None), the where clause
            will be ignored. If the MySQL runs with "safe-update-mode", the lack of where clause will cause
            update failure.

            2. For pyhton < 3.6, the dict isn't ordered, so please not try to excute it one by one if your python version < 3.6.
            If you have to use ordered dict, please use OrderedDict instead.

    For example, the kw may be like this:
    table: sampleTable, {"id=1": {"name": "Sam", "age": 12},
                         "id=2": {"name": "Tony", "sex": "male"},
                         "None-key": {"class": "Class-1", "Establish_time": "2017-1-1"}}
    and the sqls are shown as follows:
    
    1."update sampleTable set name = 'Sam', age = 12 where id = 1;"
    2."update sampleTable set name = 'Tony' sex = 'male' where id = 2;"
    3."update sampleTable set class = 'Class-1' Establish_time = '2017-1-1';# " This may fail if mysql runs with "safe-update-mode"

    Test case:
    # sqls = generate_update_sqls("TestTable", **{"id=1": {"name": "Sam", "age": 12},
    #                                 "id=2": {"name": "Tony", "sex": "male"},
    #                                 "None-key": {"class": "Class-1", "Establish_time": "2017-1-1"}})
    """
    # meta-sqls
    update_sql = " update %s set "
    value_sql_str = " %s = '%s' "
    value_sql_Notstr = " %s = %s "
    where_sql = " where %s "
    end = ";"

    sqls = []
    update = update_sql % __check_db_is_set__(db,table)    
    for condition, values in kw.items(): # for each condition
        where = "" if condition.startswith("None") else where_sql % condition
        set_clauses = []
        for attr, val in values.items(): # get value and condition of each case
            set_clause = (value_sql_str if isinstance(val, str) else value_sql_Notstr) % (__check_space_in_column_name__(attr), val)
            set_clauses.append(set_clause)
        set_sql = ", ".join(set_clauses)
        sql = update + set_sql + where + end # generate sql
        sqls.append(sql)
        # print(sql) # used to debug

    return sqls