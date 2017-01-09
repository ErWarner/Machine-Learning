from __future__ import division
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.mlab as mlab
import math

# Generate the data according to the specification in the homework description

N = 10000

# Here's an estimate of gamma for you
G = lambda x: np.log(np.cosh(x))
gamma = np.mean(G(np.random.randn(10**6)))

s1 = np.sin((np.arange(N)+1)/200)
s2 = np.mod((np.arange(N)+1)/200, 2) - 1
S = np.concatenate((s1.reshape((1,N)), s2.reshape((1,N))), 0)

A = np.array([[1,2],[-2,1]])

X = A.dot(S)

# TODO: Implement ICA using a 2x2 rotation matrix on a whitened version of X

L,V = np.linalg.eig(np.dot(X,np.transpose(X)))
Lam = np.diag(np.matrix(L).A1)

D   = np.sqrt(N)*np.dot(np.sqrt(np.linalg.inv(Lam)),np.transpose(V))

Xtil = np.dot(D,X)

Gam  = np.mean(np.log(np.cosh(np.random.normal(0,1,10**6))))

Theta = np.linspace(0,math.pi/2,50)
Total_Cost=[]


for i in range(0,len(Theta)):
    theta=Theta[i]    
    W = np.array([[math.cos(theta),math.sin(-theta)],[math.sin(theta),math.cos(theta)]])
    Cost = 0
    for j in range(0,len(W)):
        G     = np.mean(np.log(np.cosh(np.dot(W[j],Xtil))))
        Cost += (G-Gam)**2
        
    
    Total_Cost.append(Cost)
    if Cost == np.min(Total_Cost):
        Theta_Min = theta
        W_Min     = W
    
Y   = np.dot(W,Xtil)


print np.shape(Y)
print np.shape(Y[1,:])



plt.plot(range(0,N),Y[0,:],'r',range(0,N),Y[1,:],'g')
plt.ylabel('Y Value')
plt.xlabel('Iteration')
plt.show()

plt.plot(range(0,N),S[0,:],'r',range(0,N),S[1,:],'g')
plt.ylabel('S Value')
plt.xlabel('Iteration')
plt.show()


plt.plot(Theta,Total_Cost)
plt.ylabel('J(y)')
plt.xlabel('Theta')
plt.show()


    
    




