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

s = __file__
print("%r" %s)

print(os.path.abspath("./database_process"))
s = os.path.dirname(p=__file__)+'/data_preprocess/'
print(os.path.abspath(s))
print("s:",s)
sys.path.append(s)
import db_process
print("ok")