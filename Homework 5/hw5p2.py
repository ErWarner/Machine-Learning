import matplotlib.pyplot as plt
import numpy as np
import matplotlib.mlab as mlab
import math
from scipy.stats import gaussian_kde
from numpy.random import normal


mu = np.array([1,1])
sigma = np.matrix([[1, 0.5],[0.5,1]])


s_ab = sigma[0,0]-sigma[0,1]*sigma[1,0]/sigma[1,1]
s_ba = sigma[1,1]-sigma[0,1]*sigma[1,0]/sigma[0,0]




CSig = np.array([s_ab,s_ba])
n=0
x1 = 0
Max = 5001
X1 = np.zeros((Max,1))
X2 = np.zeros((Max,1))


while n<Max:
   
    #Sample x2
    Mu2  = mu[1]+sigma[0,1]/sigma[0,0]*(x1-mu[0])
    x2   = np.random.normal(Mu2,np.sqrt(s_ba),1)
    X2[n] = x2
    
    #Sample x1
    Mu1   = mu[0]+sigma[0,1]/sigma[1,1]*(x2-mu[1])
    x1   = np.random.normal(Mu1,np.sqrt(s_ab),1)
    X1[n]=x1

    
    n=n+1
 
print "------------"

x = np.linspace(-5,6,100)

plt.plot(x,mlab.normpdf(x,mu[0],np.sqrt(sigma[0,0])))
plt.hist(X1,bins=25,normed=True)
plt.title("Gaussian Histogram")
plt.xlabel("Value")
plt.ylabel("p(x1)")
plt.show()


plt.plot(x,mlab.normpdf(x,mu[1],np.sqrt(sigma[1,1])))
plt.hist(X2,bins=25,normed=True)
plt.title("Gaussian Histogram")
plt.xlabel("Value")
plt.ylabel("p(x2)")
plt.show()


