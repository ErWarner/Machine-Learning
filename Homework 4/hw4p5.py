from __future__ import division
import numpy as np
from matplotlib import pyplot as plt
from scipy import stats
import pylab
import math

# Generate the data according to the specification in the homework description

N = 500
x = np.random.rand(N)

pi0 = np.array([0.7, 0.3])
w0 = np.array([-2, 1])
b0 = np.array([0.5, -0.5])
sigma0 = np.array([.4, .3])

y = np.zeros_like(x)
for i in range(N):
    k = 0 if np.random.rand() < pi0[0] else 1
    y[i] = w0[k]*x[i] + b0[k] + np.random.randn()*sigma0[k]


# TODO: Implement the EM algorithm for Mixed Linear Regression based on observed
# x and y values.

pih = np.array([0.5,0.5])
wh  = np.array([1,-1])
bh  = np.array([0,0])
sigmah  = (np.std(y)*np.array([1,1]))
r   = np.zeros((N,2))
Dist=np.zeros((N,2))
pi  = []
pi.append(pih)
p=0
LogL=[]
Q=0
Qold = 1
w=[]
b=[]
sigma=[]

while abs(Q-Qold)>10**(-4):
    Qold = Q
    print "hello"
    for n in range(0,N):
        sum=0
        for j in range(0,2):
            Mu     = wh[j]*x[n]+bh[j]
            Gaus = 1/np.sqrt(2.0*math.pi*sigmah[j]**2)*math.exp(-(y[n]-Mu)**2/(2.0*(sigmah[j]**2)))
            sum = sum+pih[j]*stats.norm(Mu,sigmah[j]).pdf(y[n])
            sum = sum+pih[j]*Gaus
            Dist[n,j] = np.log(stats.norm(Mu,sigmah[j]).pdf(y[n]))
            Dist[n,j] = np.log(Gaus)
        for k in range(0,2):
            Mu     = wh[k]*x[n]+bh[k]
            Gaus = (1/np.sqrt(2.0*math.pi*(sigmah[k]**2)))*math.exp(-(y[n]-Mu)**2/(2.0*sigmah[k]**2))
            r[n,k] = pih[k]*stats.norm(Mu,sigmah[k]).pdf(y[n])/sum
            r[n,k] = pih[k]*Gaus/sum
    print "bye"
	
    #---------------------E-STEP---------------------------%
    Q = -np.sum(r*np.log(pih))+np.sum(r*Dist)
    LogL.append(Q)
    
    #---------------------M-STEP---------------------------%
    pih = r.sum(axis=0)/N
    pi.append(pih)
    
    
    yn    = np.transpose(np.array([y,y]))
    xn    = np.transpose(np.array([x,x]))
    Den   = (r*xn*xn).sum(axis=0)	
    LHS   = (r*yn).sum(axis=0)-((r*yn*xn).sum(axis=0)*xn*r/Den).sum(axis=0)
    RHS   = r.sum(axis=0)-((r*xn).sum(axis=0)*r*xn/Den).sum(axis=0)

    bh    =  LHS/RHS
    b.append(bh)
    wh     = (r*yn*xn-r*bh*xn).sum(axis=0)/Den	
    w.append(wh)
    
    Mult2  = r*((yn-(xn*wh+bh))**2)
    sigmah = np.sqrt(Mult2.sum(axis=0)/r.sum(axis=0))
    sigma.append(sigmah)
	
    
    print (pih,bh,sigmah,wh)
    print Q-Qold
    print "-------------------"

    
    p=p+1
    if p>300:
        break


print "The estimated pi parameters are",(pih[1],pih[0])
print "The estimated w parameters are",(wh[1],wh[0])
print "The estimated b parameters are",(bh[1],bh[0])
print "The estimated sigma parameters are",(sigmah[1],sigmah[0])


# Here's the data plotted
plt.plot(range(len(LogL)),LogL)
plt.xlabel('Iteration')
plt.ylabel('Log Likelihood Function')
plt.show()

output = xn*wh+bh
print np.shape(output)

plt.plot(range(len(y)),y,range(len(x)),output)
plt.xlabel('Iteration')
plt.ylabel('Pi Value')
plt.show()

pylab.plot(range(len(y)), y, '-b', label='y')
pylab.plot(range(len(x)), output[:,0], '-r', label='k=1')
pylab.plot(range(len(x)), output[:,1], '-g', label='k=2')
pylab.legend(loc='upper left')
pylab.show()

plt.plot(range(len(pi)),pi)
plt.axhline(y=pi0[0], xmin=0, xmax=len(pi), hold=None,color='r')
plt.axhline(y=pi0[1], xmin=0, xmax=len(pi), hold=None,color='r')
plt.xlabel('Iteration')
plt.ylabel('Pi Value')
plt.show()


plt.plot(range(len(w)),w)
plt.axhline(y=w0[0], xmin=0, xmax=len(w), hold=None,color='r')
plt.axhline(y=w0[1], xmin=0, xmax=len(w), hold=None,color='r')
plt.xlabel('Iteration')
plt.ylabel('W Value')
plt.show()

plt.plot(range(len(b)),b)
plt.axhline(y=b0[0], xmin=0, xmax=len(b), hold=None,color='r')
plt.axhline(y=b0[1], xmin=0, xmax=len(b), hold=None,color='r')
plt.xlabel('Iteration')
plt.ylabel('b Value')
plt.show()

plt.plot(range(len(sigma)),sigma)
plt.axhline(y=sigma0[0], xmin=0, xmax=len(sigma), hold=None,color='r')
plt.axhline(y=sigma0[1], xmin=0, xmax=len(sigma), hold=None,color='r')
plt.xlabel('Iteration')
plt.ylabel('Sigma Value')
plt.show()


plt.scatter(x, y, c='r', marker='x')
plt.show()

   