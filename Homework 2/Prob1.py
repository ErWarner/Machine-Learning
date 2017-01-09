import numpy as np
import pandas
import math
import matplotlib.patches as mpatches
from matplotlib import pyplot as plt;
if "bmh" in plt.style.available: plt.style.use("bmh");

#Import the data
Data = pandas.read_csv("train_graphs_f16_autopilot_cruise.csv", sep=",",
                           names= ["id","rolling_speed","elevation_speed","elevation_jerk",
"elevation,roll","elevation_acceleration","controller_input"])

Test = pandas.read_csv("test_graphs_f16_autopilot_cruise.csv", sep=",",
                           names= ["id","rolling_speed","elevation_speed","elevation_jerk",
"elevation,roll","elevation_acceleration","controller_input"])


names= ["rolling_speed","elevation_speed","elevation_jerk",
"elevation,roll","elevation_acceleration"]


#Shave off the id section
Test = Test[1:]

Error_P=[]
Error_R=[]
Error_Reg_Test=[]
Error_Test=[]
ErrorR_test = []

#Max degree of polynomial
j=6
d = range(j+1)
#Lambda values
lamb = range(-40,21,1)


t2=[]
print d[1:]

for n in d[1:]:
    Theta=[]
    ThetaT=[]
    m=0
    t=[]
    #Generate the Theta Matrix for the Training data
    for x in range(1,len(Data)+1):
        t.append(float(Data["controller_input"][x]))
        row = [1]
        for y in range(len(names)):   		
            for z in range(1,n+1):
                Value = float(Data[names[y]][x])
                row.append(Value**(z))
        Theta.append(row)
    #Generate the Theta Matrix for the Test data    
    for x in range(len(Test)):
        t2.append(float(Test["controller_input"][x]))
        row = [1]
        for y in range(len(names)):   		
            for z in range(1,n+1):
                Value = float(Test[names[y]][x])
                row.append(Value**(z))
        ThetaT.append(row)
    #Compute w without regularization
    m = len(np.dot(np.transpose(Theta),Theta))
    w = np.dot(np.dot(np.linalg.inv(np.dot(np.transpose(Theta),Theta)),np.transpose(Theta)),t)
    
    #Compute w with regularization for n=6
    if n==6:
        for lam in lamb:
            ErrorR=0
            ErrorR_test=0
            lam = np.exp(lam)
            wR = np.dot(np.dot(np.linalg.inv(np.dot(np.transpose(Theta),Theta)
            +lam*np.identity(m)),np.transpose(Theta)),t)
            for x in range(len(Data)):
                Error2 = (np.dot(Theta[x],wR)-t[x])**2#+lam/2*np.dot(np.transpose(wR),wR)
                ErrorR = ErrorR+Error2
            for x in range(len(Test)):
                Error2 = (np.dot(ThetaT[x],wR)-t2[x])**2#+lam/2*np.dot(np.transpose(wR),wR)
                ErrorR_test = ErrorR_test+Error2
            Error_R.append(np.sqrt(ErrorR/len(Data)))
            Error_Reg_Test.append(np.sqrt(ErrorR_test/len(Test)))
    
    #U,S,V= np.linalg.svd(np.dot(np.transpose(Theta),Theta))
    
    #Find the test and training error for the two examples
    ErrorT=0
    Error_T=0
    for x in range(len(Data)):
        Error = (np.dot(Theta[x],w)-t[x])**2
        ErrorT = ErrorT+Error
    for x in range(len(Test)):
        Error_T2 = (np.dot(ThetaT[x],w)-t2[x])**2 
        Error_T= Error_T+Error_T2
    Error_P.append(np.sqrt(ErrorT/len(Data)))
    Error_Test.append(np.sqrt(Error_T/len(Test)))

#plot the results    
print Error_P 
print Error_Test
print Error_Reg_Test
print d[1:] 
plt.plot(d[1:],Error_P,"g",d[1:],Error_Test,"b")
plt.ylabel('Error')
plt.xlabel('Degree of Polynomial')
green_patch = mpatches.Patch(color='green', label='Training Data')
blue_patch = mpatches.Patch(color='blue', label='Test Data')
plt.legend(handles=[green_patch, blue_patch])
plt.show()

plt.plot(lamb,Error_R,"g",lamb,Error_Reg_Test,"b")
plt.ylabel('Error')
plt.xlabel('Value of ln(lambda)')
green_patch = mpatches.Patch(color='green', label='Training Data')
blue_patch = mpatches.Patch(color='blue', label='Test Data')
plt.legend(handles=[green_patch, blue_patch])
plt.show()



"""

print "The train least squares Errors are ",Error_P
print "..."
print "..."
print "..."
print "The train Regularized Errors are ",Error_R
print "..."
print "..."
print "..."
print "The test least squares Errors are ",Error_Test
print "..."
print "..."
print "..."
print "The test Regularized Errors are ",Error_Reg_Test
print "..."
print "..."
print "..."
print "The minimum train error is ",np.amin(Error_P)
print "..."
print "..."
print "..."
print "The minimum train Regularized Errors is ",np.amin(Error_R)
print "..."
print "..."
print "..."
print "The minimum test least squares Errors is ",np.amin(Error_Test)
print "..."
print "..."
print "..."
print "The minimum test Regularized Errors is ",np.amin(Error_Reg_Test)   
"""



