# -*- coding: utf-8 -*-

# add and import my database tools
import sys
sys.path.append('/home/luruiyuan/python/Codes/machine learning/database_process/data_preprocess') # add package to system path temperarily
import db_process as db
from db_process import __check_db_is_set__, __check_space_in_column_name__


def delete_by_null_column_value(conn, *, database=None, table, column, pk=" No >= 0 "):
    """
    delete records that specific column is null or ""
    """
    print("excuting delete null records...")
    col = __check_space_in_column_name__(column)
    sql = "delete from %s where %s and (%s is null or %s = \"\");" \
            % (__check_db_is_set__(database, table), pk, col, col)
    print(sql)
    db.excute_update_sqls(conn, sql)
    print("excuting delete null records succeeded!")

def wash_duplicate_data_by_column(conn, *, database=None, table, column, pk=" No >= 0 "):
    """
    return the unique value set and the duplicate value map of the specific column.
    pk is the primary key in database and in the condition that used to delete.
    
    For example:
        column: src
        value: IntelCor_ca:d3:5c, IntelCor_ca:d3:5c (94:65:9c:ca:d3:5c) (TA), 192.167.1.1, 192.168.1.2
        
        As shown above, IntelCor_ca:d3:5c (94:65:9c:ca:d3:5c) (TA) is duplicated MAC address, and should be
        replaced by IntelCor_ca:d3:5c (the shortter one). This function can replace it automatically,
    """

    col = __check_space_in_column_name__(column)    
    sql = "select distinct %s from %s;" % (col, __check_db_is_set__(database, table))
    cursor = db.excute_has_resultset_sqls(conn, sql)

    rows = cursor.fetchall()

    # find duplicate data
    source = set()
    change = dict()
    for row in rows:
        issubstr = False
        src = row[column]
        for item in source:
            if src in item or item in src:# is substring
                issubstr = True
                min_src, del_src = (src, item) if len(src) < len(item) else (item, src)
                change[min_src] = del_src # minimum src as key, rest as value.
                if item == del_src:
                    source.remove(item)
                    source.add(src)
                break
        if not issubstr:
            source.add(src)
    
    # wash duplicate data
    params = {}
    for remain, remove in change.items():
        factor = pk + " and (%s = '%s')" % (__check_space_in_column_name__(column), remove)
        params[factor] = {column: remain}
    sqls = db.generate_update_sqls(table, db=database, **params)
    db.excute_update_sqls(conn, *sqls)

    return source, change

# def 

def wash_data(*columns, conn, database='802_11_exprmt', table='raw_data', pk=" No >= 0 "):

    for column in columns:
        delete_by_null_column_value(conn, database=database, table=table, column=column, pk=pk)
        wash_duplicate_data_by_column(conn, database=database, table=table, column=column, pk=pk)


def union_2_table_data(*,conn, database1, table1="raw_data", database2, table2="raw_data", target_db="test", target_table="test", order_by="TSF Timestamp"):
    check_db = __check_db_is_set__
    sql = "create table %s as (select * from %s) union (select * from %s) order by %s;"\
        % (check_db(target_db, target_table), check_db(database1, table1), check_db(database2, table2),__check_space_in_column_name__(order_by))
    db.excute_update_sqls(conn, sql)

def drop_columns_by_name(*, conn, databse, table, columns):
    """
    drop one column or many columns.
    """
    alter = "alter table %s " % __check_db_is_set__(databse, table)
    drop = " drop column %s "
    drop_cols = []
    sql = ""
    if isinstance(columns, str):
        sql = alter + drop % __check_space_in_column_name__(columns) + ";"
    else:
        for col in columns:
            drop_cols.append(drop % __check_space_in_column_name__(col))
        sql = alter + ", ".join(drop_cols) + ";"
    db.excute_has_resultset_sqls(conn, sql)

def add_columns_by_name(*, conn, databse, table, columns, data_types):
    """
    add one column or many columns.
    """
    alter = "alter table %s " % __check_db_is_set__(databse, table)
    add = " add column %s %s "
    sql = ""
    
    if isinstance(columns, str) and isinstance(data_types, str):
        sql = alter + add % (__check_space_in_column_name__(columns), data_types) + ";"
    else:
        adds = []        
        for name, type in zip(columns, data_types):
            adds.append(add % (name, type))
        sql = alter + ", ".join(adds) + ";"
    db.excute_has_resultset_sqls(conn, sql)

def copy_column_in_same_table(*, from_columns, to_columns, conn, database, table):
    """
    copy each value of column 'from_column' into specific column 'to_column'
    """
    sqls = ["set SQL_SAFE_UPDATES = 0;"] # close ssafe-update mode
    update = "update %s set" % __check_db_is_set__(database, table)
    equal = " %s = %s"
    check = __check_space_in_column_name__

    if isinstance(from_columns, str) and isinstance(to_columns, str):
        sql = update + equal % (check(to_columns), check(from_columns)) + ";"
    else:
        tmp = []
        for f, t in zip(from_columns, to_columns):
            tmp.append(equal % (check(t), check(f)))
        sql = update + ", ".join(tmp) + ";"
    sqls.extend([sql, "set SQL_SAFE_UPDATES = 1;"]) # reset to safe-update mode
    db.excute_update_sqls(conn, *sqls)

def main():
    
    conn = db.create_connection()

    columns = ["Source address", "Destination"]
    wash_data(*columns,conn=conn, database='alu')

    union_2_table_data(conn=conn, database1="alu", database2="802_11_exprmt", target_db="alu", target_table="data")

    drop_columns_by_name(conn=conn, databse="alu", table="data", columns="No")
    add_columns_by_name(conn=conn, databse="alu", table="data", columns=["label1"], data_types=["varchar(45)"])

    # copy_column_in_same_table(from_columns=["Protocol", "Type/Subtype"], to_columns=["Source address","Destination"], conn=conn, database="alu", table="data")
    copy_column_in_same_table(from_columns=["Source address"], to_columns=["label1"], conn=conn, database="alu", table="data")

    db.close_connection()


if __name__ == "__main__":
    main()
