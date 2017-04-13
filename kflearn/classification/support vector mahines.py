# -*- coding: utf-8 -*-
from sklearn.svm import SVC as svc, NuSVC as nusvc, LinearSVC as linearsvc
x = [[0, 0], [1, 1]]
y = [0,1]
y1 = ["class1", "class2"]
clf = svc()
clf.fit(x,y)

print(clf.fit(x,y1))

print(clf.predict([[2.,2.]]))