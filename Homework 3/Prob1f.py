import numpy as np
import pandas
import math
from operator import add
from matplotlib import pyplot as plt;

#Import Data
training = pandas.read_csv("digits_training_data.csv", sep=",",header=None)
trainingL = pandas.read_csv("digits_training_labels.csv", sep=",",header=None)
test = pandas.read_csv("digits_test_data.csv", sep=",",header=None)
testL = pandas.read_csv("digits_test_labels.csv", sep=",",header=None)

(tr_row,tr_col) = np.shape(training)
(te_row,te_col) = np.shape(test)

len_tr          = len(trainingL)
len_te          = len(testL)

trainingL[trainingL==4] = -1
trainingL[trainingL==9] = 1
testL[testL==4] = -1
testL[testL==9] = 1

w = np.zeros((tr_col,1))
b = 0
eta = 0.001
C   = 3	
#l;
num_it=100
ErrorT=[]
alphaT=[]

for i in range(num_it):
    alpha = eta/(1+float(i)*eta)
    E=0
    for j in np.random.permutation(tr_row):
        t = np.asarray(trainingL[0][j])
        X = np.asarray(training.iloc[[j]])
        term = 1-t*(np.transpose(w).dot(np.transpose(X))+b)
        wg = -alpha*C*t*np.transpose(X)	
        bg = -alpha*C*t
        wv = alpha*w
        if term<0:
            term  = 0
            wg    = 0
            bg    = 0
        E  = E+C*term
        wg2 = -(wv/tr_row+wg)
        w = map(add,w,wg2)
        w = np.asarray(w)
        b = b-bg
    Error     = np.linalg.norm(w,2)**2/2+E
    ErrorT.append(np.sqrt(Error/tr_row))
    print (np.sqrt(Error/tr_row))

plt.plot(range(num_it),ErrorT,"g")
plt.ylabel('Error')
plt.xlabel('Iteration Number)')
plt.show()



