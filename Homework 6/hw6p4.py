import numpy as np
import math
import random

X  = np.array([[0.22,0.19],[0.30,0.17],[0.45,0.20],[0.20,0.18],[0.12,0.17],[0.10,0.15],[0.30,0.15]])
y  = np.array([-1,-1,-1,1,1,1,1])

W0 = 1.0/len(X)*np.ones((1,7))
W= W0[0]
W = W/np.sum(W)
n=7

alpha = np.ones((7,1))
theta1 = np.array([0,-1])
theta0 = 0.17
classifier = np.zeros((n,1))

#--------------------------PART A----------------------------------------#

epsilon =0
j=0

for i in range(0,n):
	classifier[i,0] = np.sign(X[i][0]*theta1[0]+X[i][1]*theta1[1]+theta0+0.0000001)
	if classifier[i] != y[i]:
		j+=1
		epsilon += W[i]
	
alpha = 1.0/2.0*np.log((1.0-epsilon)/epsilon)   
for i in range(0,n):
    W[i]     = W[i]*math.exp(-classifier[i]*alpha*y[i])

W = W/np.sum(W)

print "After one iteration, alpha is ",alpha
print "After one iteration, W is", W


#-----------------------PART B------------------------------------#


theta_array = np.array([2,-1,0.17])
M     = 10

classifier = np.zeros((n,1))
f = np.zeros((n,1))
W0 = 1.0/len(X)*np.ones((1,7))
W= W0[0]

m=0
hT=0

while m<M:   

	min = 100
	thetaT = np.array([-1,1])
	b0     = np.array([np.linspace(0.1, 0.5, 20),np.linspace(0.14, 0.2, 20)])
	
    #Find the optimal theta,b,k
	for k in [0,1]:
		for b in b0[k]:
			for theta in [-1,1]:
				epsilon=0
				for i in range(0,n):
					f[i]  =(X[i][k]*theta+b)
					classifier[i] = np.sign(f[i]+0.0000001)

					if classifier[i] != y[i]:
						epsilon += W[i]  
				if epsilon<min:
					min=epsilon
					theta_min = theta
					k_min     = k
					b_min     = b

				
					
	

	theta1 = theta_min
	b      = b_min
	k      = k_min


    
	epsilon = 0
	j=0
    #Find the classifier values of the optimal parameters
	for i in range(0,n):
		f[i]          =(X[i][k]*theta1+b)
		classifier[i] = np.sign(f[i]+0.0000001)

		if classifier[i] != y[i]:
			epsilon += W[i]   
			j+=1

            

    #Solve for alpha
	alpha = 1.0/2.0*np.log((1.0-epsilon)/epsilon)
    
    
	

    
	for i in range(0,n):
		W[i]     = W[i]*math.exp(-classifier[i]*alpha*y[i])
	W = W/np.sum(W)


	hT += alpha*classifier
	m+=1
    
EL =0


for i in range(0,len(hT)):
    EL += math.exp(-y[i]*(hT[i]))
   

print "The exponential loss is",EL  
print "The hT output is",hT






