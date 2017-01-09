import numpy as np
import pandas
import math
from operator import add
from matplotlib import pyplot as plt;
import sklearn
from sklearn.kernel_ridge import KernelRidge

#Import Data
train = pandas.read_csv("steel_composition_train.csv", sep=",")
test = pandas.read_csv("steel_composition_test.csv", sep=",")

names = ["id","Carbon","Nickel","Manganese","Sulfur","Chromium","Iron","Phosphorus","Silicon"]


data_train = train[names]
targets_train = train["Strength"]

tr_len = len(targets_train)

krr2 = KernelRidge(alpha = 1,kernel="polynomial",degree=2,coef0=1)
krr2.fit(data_train, targets_train)
krr2_score = krr2.score(data_train, targets_train)

K2TR = krr2.predict(data_train)
E= K2TR - targets_train
E = np.asarray(E)
RMSE2 = np.sqrt(np.dot(np.transpose(E),E)/tr_len)


krr3 = KernelRidge(alpha = 1,kernel="polynomial",degree=3,coef0=1)
krr3.fit(data_train, targets_train)
krr3_score = krr3.score(data_train, targets_train)

K3TR = krr3.predict(data_train)
E= K3TR - targets_train
E = np.asarray(E)
RMSE3 = np.sqrt(np.dot(np.transpose(E),E)/tr_len)

krr4 = KernelRidge(alpha = 1,kernel="polynomial",degree=4,coef0=1)
krr4.fit(data_train, targets_train)
krr4_score = krr4.score(data_train, targets_train)


K4TR = krr4.predict(data_train)
E= K4TR - targets_train
E = np.asarray(E)
RMSE4 = np.sqrt(np.dot(np.transpose(E),E)/tr_len)

krrG = KernelRidge(alpha = 1,kernel="rbf",gamma=0.001)
krrG.fit(data_train, targets_train)
krrG_score = krrG.score(data_train, targets_train)


KGTR = krrG.predict(data_train)
E= KGTR - targets_train
E = np.asarray(E)
RMSEG = np.sqrt(np.dot(np.transpose(E),E)/tr_len)

print "The RMSE of the 2 degree polynomial kernel is", RMSE2
print "The RMSE of the 3 degree polynomial kernel is", RMSE3
print "The RMSE of the 4 degree polynomial kernel is", RMSE4
print "The RMSE of the Gaussian kernel is", RMSEG





