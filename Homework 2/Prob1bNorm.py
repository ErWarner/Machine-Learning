import numpy as np
import pandas
import math
import scipy.sparse as sp
import scipy.sparse.linalg as spl
from matplotlib import pyplot as plt;
if "bmh" in plt.style.available: plt.style.use("bmh");

#Get data
train = pandas.read_csv("train_graphs_f16_autopilot_cruise.csv", sep=",",
                           names= ["id","rolling_speed","elevation_speed","elevation_jerk",
"elevation","roll","elevation_acceleration","controller_input"])

test = pandas.read_csv("test_locreg_f16_autopilot_cruise.csv", sep=",",
                           names= ["id","rolling_speed","elevation_speed","elevation_jerk",
"elevation","roll","elevation_acceleration","controller_input"])

names= ["rolling_speed","elevation_speed","elevation_jerk",
"elevation","roll","elevation_acceleration"]


#Slice off the id from the test data
Train = train
Test  = test[1:]

Error= 0
ErrorT =[]
Theta = []
t = []

#Find the mean and standard deviation of training data
mT= Train.mean()
SDT= Train.std()

#Generate theta and the parameter value vector for the training data
for z in range(len(Train)): 
    row =[1]
    for name in names:  
        mN = mT[name]
        SD = SDT[name] 
        Value = (float(Train[name][z])-mN)/SD
        row.append(Value)
    Theta.append(row)
    c = "controller_input"
    ti = (Train[c][z]-mT[c])/SDT[c]
    t.append(float(ti))


Tau = np.logspace(-2, 1, num=10, base=2)


for tau in Tau:
    Error=0
    #Cycle through each test point
    for y in range(1,len(Test)+1):
        r=[]
        R=[]
        #Cycle through the training data
        for z in range(len(Train)): 
            x  = []
            xi = []
            dif = []
            X   = [1]
            #Normalize each value and compute local weight
            for name in names:    
                mN = mT[name]
                SD = SDT[name] 
                x1 = (Train[name][z]-mN)/SD
                x2 = (float(Test[name][y])-mN)/SD
                x.append(float(x2))
                xi.append(float(x1))
                X.append(float(x2))
                dif.append(float(x1)-float(x2))
            nm = (np.dot(np.transpose(dif),dif))
            ri = np.exp(-nm/(2*(float(tau)**2))) 
            r.append(float(ri)/2)
        
        #Generate R
        R = np.diag(r)
        
        #Find the w vector and find the error
        TT = np.transpose(Theta).dot(R)
        TT = np.dot(TT,Theta)
        t  = np.asarray(t)
        w1  = np.dot(np.dot(np.linalg.inv(TT),np.transpose(Theta)),R)
        w   = np.dot(w1,t)

        tt  = (float(Test["controller_input"][y])-mT[c])/SDT[c]
        Er  =(np.dot(np.transpose(w),X)-float(tt))**2/len(Test)
        Error = Error+Er
    #Error for each value of Tau
    ErrorT.append(np.sqrt(Error))   
    print ErrorT      

#Plot the results
plt.plot(Tau,ErrorT)
plt.ylabel('Error')
plt.xlabel('Value of Tau')
plt.show()  
