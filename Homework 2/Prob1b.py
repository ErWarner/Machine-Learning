import numpy as np
import pandas
import math
import scipy.sparse as sp
import scipy.sparse.linalg as spl
from matplotlib import pyplot as plt;
if "bmh" in plt.style.available: plt.style.use("bmh");
from scipy.sparse import csr_matrix

train = pandas.read_csv("train_graphs_f16_autopilot_cruise.csv", sep=",",
                           names= ["id","rolling_speed","elevation_speed","elevation_jerk",
"elevation","roll","elevation_acceleration","controller_input"])

test = pandas.read_csv("test_locreg_f16_autopilot_cruise.csv", sep=",",
                           names= ["id","rolling_speed","elevation_speed","elevation_jerk",
"elevation","roll","elevation_acceleration","controller_input"])

names= ["rolling_speed","elevation_speed","elevation_jerk",
"elevation","roll","elevation_acceleration"]

Test = test[1:]
Train = train[1:]

#print Train[0:10]
#PART 1b

Tau = np.logspace(-2, 1, num=10, base=2)

Error= 0
ErrorT =[]
Theta = []
t = []

for z in range(1,len(Train)+1): 
    row =[1]
    for p in range(len(names)):  
        Value = float(Train[names[p]][z])
        row.append(Value)
    Theta.append(row)
    ti = Train["controller_input"][z]
    t.append(float(ti))



for tau in Tau:
	Error=0
	for y in range(1,len(Test)+1):
		r=[]
		R=[]
		for z in range(1,len(Train)+1): 
			x  = []
			xi = []
			dif = []
			X   = [1]
			for name in names:
				x1 = Train[name][z]
				x2 = Test[name][y]
				x.append(float(x2))
				xi.append(float(x1))
				X.append(float(x2))
				dif.append(float(x1)-float(x2))
            #print np.dot(np.transpose(dif),dif)
			nm = (np.dot(np.transpose(dif),dif))
			ri = np.exp(-nm/(2*(float(tau)**2))) 
            #print ri
			r.append(float(ri/2))
            
        #R = sp.diags(r,0)     
		R = np.diag(r)
        
        #TT = np.transpose(Theta).dot(R)
        #TT = np.dot(TT,Theta)
        #U,S,V= np.linalg.svd(R)
		t  = np.asarray(t)
        #print np.shape(TT)
        #w1  = np.dot(np.dot(np.linalg.inv(TT),np.transpose(Theta)),R)
        #w   = np.dot(w1,t)
		#Can use multiply instead of dot phi.transpose()*R*phi
		Theta = csr_matrix(Theta)
		R     = csr_matrix(R)
		t     = csr_matrix(t)
		
		A= Theta.transpose()*R*Theta
		B=(Theta.transpose()*R*t.transpose())
		print np.shape(A)
		print np.shape(B)

		w2   = spl.cgs(A, B, [np.ones((7,1))])
		tt  = (Test["controller_input"][y])
		Er  =(np.dot(np.transpose(w2[0]),X)-float(tt))**2/len(Test)
		Error = Error+Er
	ErrorT.append(np.sqrt(Error))
	print ErrorT     
        
print ErrorT     


plt.plot(Tau,ErrorT)
plt.ylabel('Error')
plt.xlabel('Value of Tau')
plt.show()  