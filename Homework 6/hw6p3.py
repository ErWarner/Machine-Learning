import cPickle as pickle
import numpy as np
from itertools import izip
import random
import sklearn
from sklearn import linear_model


train = open('train.pkl','rb')
Dict = pickle.load(train)

X = np.reshape(Dict['data'],(50000,3,1024))
Y = Dict['labels']

print Y[0:100]



M = 1024
K = 100
D = 50
N = 10


def sigmoid(x):
    return 1 / (1+np.exp(-x))




def forwardprop(x, t, A, S, W):

    # ---------- make your implementation here -------------
    P=0
    for i in range(0,2):
        y = np.transpose(sigmoid(A[:,M]+np.transpose(np.dot(A[:,0:M],x[i]))))
        z = np.transpose(sigmoid(S[:,K]+np.transpose(np.dot(S[:,0:K],y))))
        P += np.transpose(np.exp(W[:,D]+np.transpose(np.dot(W[:,0:D],z))))
    P_den = np.sum(P)
    P = P/P_den
    J = -np.log(P[t])
    # -------------------------------------------------

    return y, z, P, J, P_den



def backprop(x, t, A, S, W):

    y, z, P, J, P_den = forwardprop(x, t, A, S, W)
    I = np.zeros((N, 1), dtype=np.float)
    I[t] = 1
	
    grad_W = np.zeros(np.shape(W))
    grad_S = np.zeros(np.shape(S))
    grad_A = np.zeros(np.shape(A))
    
    # ---------- make your implementation here -------------    
    for m in range (0,2):
    
        for i in range(0,D):
            grad_W[t,i] += -z[i] + z[i]*P[t]
        grad_W[t,D] += -1+P[t]

    
        dEdzk=0    
        for i in range(0,N):
            dEdzk += W[i,t]*(np.exp(W[i,D]+np.dot(W[i,0:D],z)))/P_den
        for k in range(0,D):
            for j in range(0,K):
                grad_S[k,j]+= -(W[t,k]-dEdzk)*z[k]*(1-z[k])*y[j]
            grad_S[k,K] += -(W[t,k]-dEdzk)*z[k]*(1-z[k])
     
        dEdz=0 
        for k in range(0,D):
            dEdz += -(W[t,k] - dEdzk)*z[k]*(1-z[k])*S[k,t]          
    
    
        for i in range(0,K):
            for j in range(0,M):
                grad_A[i,j] += dEdz*y[i]*(1-y[i])*x[m][j]
            grad_A[i,M] += dEdz*y[i]*(1-y[i])
    
    
    
    
    # -------------------------------------------------

    return grad_A, grad_S, grad_W
    
   
    
A = np.random.rand(K, M+1)*0.1-0.05
S = np.random.rand(D, K+1)*0.1-0.05
W = np.random.rand(N, D+1)*0.1-0.05
alpha = 0.0001

for j in range(0,50000):  
    
    
    grad_A, grad_S, grad_W = backprop(X[j], Y[j], A, S, W)
    A = A+alpha*grad_A      
    S = S+alpha*grad_S      
    W = W+alpha*grad_W 
    
    if j%1000==0:
        print j

T=[]  


 
        
for j in range(0,50):


    y, z, P, J, P_den = forwardprop(X[j], Y[j], A, S, W) 
    
    print "----------------------------------"
    print P
    print Y[j]
    
    for i in range(0,len(P)):
        if P[i]==np.min(P):
            T.append(i)
    if j%1000==0:
        print j

print T[0:100]
print Y[0:100]



          