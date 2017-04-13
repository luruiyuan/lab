# coding:utf-8


def str2int(s, dictionary):
    if s in dictionary:
        return dictionary[s]
    id = len(dictionary)
    dictionary[s] = id
    return id

def str2num_char(string):
    """
    如果是数字, 返回数字
    否则进行哈希
    """
    if isinstance(string, (int, float)):
        return string

    string = string.lower()
    num = 0
    for s in string:
        n = ord(s)
        num *= 10
        if ord('0') <= n <= ord('9'):
            num += n - ord('0') + 1
        elif ord('a') <= n <= ord('z'):
            num += n - ord('a') + 1
    return num
