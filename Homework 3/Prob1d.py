import numpy as np
import pandas
import math
from operator import add
import matplotlib.patches as mpatches
from matplotlib import pyplot as plt;
if "bmh" in plt.style.available: plt.style.use("bmh");

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

wGr = np.zeros((tr_col,1))
wst = np.zeros((tr_col,1))
bGr = 0
bst = 0
eta = 0.001
C   = 3
#l;
num_it=1000
ErrorGr=[]
ErrorSt=[]
Tr_Ac_Ba = []
Te_Ac_Ba = []
Tr_Ac_St = []
Te_Ac_St = []


for i in range(num_it):
    alpha = eta/(1.0+float(i)*eta)
    E=0
    wgrad=0
    bgrad=0
    Grad_ac=tr_row
    for j in range(tr_row):
        t = np.asarray(trainingL[0][j])
        X = np.asarray(training.iloc[[j]])
        term = 1-t*(np.transpose(wGr).dot(np.transpose(X))+bGr)
        wg = -float(C)*float(t)*np.transpose(X)	
        bg = -float(C)*float(t)
        if term<=0:
            term  = 0
            wg    = np.zeros((tr_col,1))
            bg    = 0
        elif term>=1:
            Grad_ac = Grad_ac-1
        wgrad = wgrad+wg
        bgrad = bgrad+bg
    Accuracy = float(Grad_ac)/float(tr_row)
    Tr_Ac_Ba.append(Accuracy)
    wgrad = map(add,wGr,wgrad)
    wGr     = wGr-np.dot(alpha,wgrad)
    bGr     = bGr-alpha*bgrad
    #print "Ba:",Accuracy
    
    
    
    E=0
    Stoch_ac=tr_row
    for j in np.random.permutation(tr_row):
        t = np.asarray(trainingL[0][j])
        X = np.asarray(training.iloc[[j]])
        term = 1-t*(np.transpose(wst).dot(np.transpose(X))+bst)
        wg = -alpha*C*t*np.transpose(X)	
        bg = -alpha*C*t
        wv = alpha*wst
        if term<=0:
            term  = 0
            wg    = 0
            bg    = 0
        elif term>=1:
            Stoc_ac = Stoch_ac-1
        wg2 = -(wv/tr_row+wg)
        wst = map(add,wst,wg2)
        wst = np.asarray(wst)
        bst = bst-bg
    Accuracy = float(Stoch_ac)/float(tr_row)
    Tr_Ac_St.append(Accuracy)
    
    Ba_te_ac = te_row
    St_te_ac = te_row
    for j in range(te_row):
        t = np.asarray(testL[0][j])
        X = np.asarray(test.iloc[[j]])
        term_ba = t*(np.transpose(wGr).dot(np.transpose(X))+bGr)
        term_st = t*(np.transpose(wst).dot(np.transpose(X))+bst)
        if term_ba<=0:
            Ba_te_ac = Ba_te_ac-1
        if term_st<=0:
            St_te_ac = St_te_ac-1
    Accuracy = float(Ba_te_ac)/float(te_row)
    Te_Ac_Ba.append(Accuracy)
    Accuracy = float(St_te_ac)/float(te_row)
    Te_Ac_St.append(Accuracy)
    print i
        
    
    #print "St:",Accuracy


plt.plot(range(num_it),Tr_Ac_Ba)
plt.xlabel('Iteration')
plt.ylabel('Training Accuracy')
plt.show()

plt.plot(range(num_it),Tr_Ac_Ba,"r",range(num_it),Tr_Ac_St,"g",range(num_it),Te_Ac_Ba,"b",range(num_it),Te_Ac_St,"y")
plt.ylabel('Accuracy')
plt.xlabel('Iteration Number)')
red_patch = mpatches.Patch(color='red', label='Training Data-Batch')
green_patch = mpatches.Patch(color='green', label='Training Data-Stochastic')
blue_patch = mpatches.Patch(color='blue', label='Test Data-Batch')
yellow_patch = mpatches.Patch(color='yellow', label='Test Data-Stochastic')
plt.legend(handles=[red_patch,green_patch, blue_patch,yellow_patch])
plt.show()

