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

for i in range(cpu):
    pass

print("cpu 数目:",cpu_count())
q.put(["1,2,3"])
q.put(["1"])

print(q.get())
print(q.get_nowait())

print(list(q))