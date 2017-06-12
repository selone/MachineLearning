import random
import numpy as np
import matplotlib.pyplot as plt
'''
Created on 2017年6月11日

@author: pengx
'''
from Perceptron import Perceptron
from pip._vendor.html5lib.treebuilders.base import Marker
file_object = open(r'E:\javaWorkPlace\MachineLearning\src\test.txt', 'r')
rank = 2
x = np.zeros((150,rank))
y = np.zeros(150)
xx = 0
try:
    for line in file_object:
        yy = 0
        temp = line.split(",")
        for temp1 in temp[1:1 + rank]:
            x[xx][yy] = float(temp1)
            yy += 1
        xx += 1
    
finally:
    file_object.close()

y[0:50] = np.ones(50) * -1
y[50:100] = np.ones(50)
y[100:150]  = np.ones(50) * 3


plt.scatter(x[:50, 0],x[:50, 1], color = 'red', marker = 'o', label = 'setosa')
plt.scatter(x[50:100, 0], x[50:100, 1], color = 'blue', marker = 'x', label = 'versicolor')
plt.legend()
plt.show()


perceptron = Perceptron(0.1,10)
perceptron.fit(x[0:100], y[0:100])
print(perceptron.errors_)

plt.plot(range(1,len(perceptron.errors_) + 1), perceptron.errors_, marker = 'o')
plt.xlabel('epoches')
plt.ylabel('number of misclassifications')
plt.show()