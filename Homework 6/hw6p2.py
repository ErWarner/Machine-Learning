import numpy as np
from itertools import izip
import random
from decimal import *


epsilon = 10e-4
M = 100
K = 50
D = 30
N = 10
nCheck = 1000

def sigmoid(x):
    return 1 / (1+np.exp(-x))




def forwardprop(x, t, A, S, W):

    # ---------- make your implementation here -------------

    y = np.transpose(sigmoid(A[:,M]+np.transpose(np.dot(A[:,0:M],x))))
    z = np.transpose(sigmoid(S[:,K]+np.transpose(np.dot(S[:,0:K],y))))
    P = np.transpose(np.exp(W[:,D]+np.transpose(np.dot(W[:,0:D],z))))
    P_den = np.sum(P)
    P = P/P_den
    J = -np.log(P[t])
    # -------------------------------------------------

    return y, z, P, J, P_den


def backprop(x, t, A, S, W):

	y, z, P, J, P_den = forwardprop(x, t, A, S, W)
	I = np.zeros((N, 1), dtype=np.float)
	I[t] = 1
	
    
    # ---------- make your implementation here -------------    
    
	grad_W = np.zeros(np.shape(W))
	
	for i in range(0,D):
		grad_W[t,i] = -z[i] + z[i]*P[t]
	grad_W[t,D] = -1+P[t]

	grad_S = np.zeros(np.shape(S))
	dEdzk=0 

	dEdzk = np.zeros((D))
	dEdzk2 = np.zeros((D))
    
	for k in range(0,D):
		for i in range(0,N):
				dEdzk[k] += W[i,k]*(np.exp(W[i,D]+np.dot(W[i,0:D],z)))/P_den
		for j in range(0,K):	
			grad_S[k,j]= -(W[t,k]-dEdzk[k])*z[k]*(1-z[k])*y[j]
		grad_S[k,K] = -(W[t,k]-dEdzk[k])*z[k]*(1-z[k])
   
	
	dEdz=0
	dEdz2=0
	
	grad_A = np.zeros(np.shape(A))
    
	for i in range(0,K):
		dEdz=0
		for k in range(0,D):
			dEdz += -(W[t,k] - dEdzk[k])*z[k]*(1-z[k])*S[k,i]
		for j in range(0,M):
			grad_A[i,j] = dEdz*y[i]*(1-y[i])*x[j]
		grad_A[i,M] = dEdz*y[i]*(1-y[i])
    
    
    
    
    # -------------------------------------------------

	return grad_A, grad_S, grad_W

def gradient_check():	
    A = np.random.rand(K, M+1)*0.1-0.05
    S = np.random.rand(D, K+1)*0.1-0.05
    W = np.random.rand(N, D+1)*0.1-0.05
    x, t = np.random.rand(M, 1)*0.1-0.05, np.random.choice(range(N), 1)[0]

    grad_A, grad_S, grad_W = backprop(x, t, A, S, W)
    errA, errS, errW = [], [], []

    for i in range(nCheck):

        # ---------- make your implementation here -------------
        idx_x, idx_y = random.randint(0,K-1), random.randint(0,M)
        # numerical gradient at (idx_x, idx_y)
        A[idx_x,idx_y]+=epsilon
        y, z, P, J, P_den = forwardprop(x, t, A, S, W)
        A[idx_x,idx_y]-=2*epsilon
        y, z, P, J2, P_den = forwardprop(x, t, A, S, W)
        
        numerical_grad_A = (J-J2)/(2*epsilon)
        A[idx_x,idx_y]+=epsilon
        
        errA.append(np.abs(grad_A[idx_x, idx_y] - numerical_grad_A))

        
        
        idx_x, idx_y = random.randint(0,D-1), random.randint(0,K)       
        
        # numerical gradient at (idx_x, idx_y)
        S[idx_x,idx_y]+=epsilon
        y, z, P, J, P_den = forwardprop(x, t, A, S, W)
        S[idx_x,idx_y]-=2*epsilon
        y, z, P, J2, P_den = forwardprop(x, t, A, S, W)

        
        numerical_grad_S = (J-J2)/(2*epsilon)

        S[idx_x,idx_y]+=epsilon


        errS.append(np.abs(grad_S[idx_x, idx_y] - numerical_grad_S))
        
        

        idx_x, idx_y = random.randint(0,N-1), random.randint(0,D)
        # numerical gradient at (idx_x, idx_y)
        W[idx_x,idx_y]+=epsilon
        y, z, P, J, P_den = forwardprop(x, t, A, S, W)
        W[idx_x,idx_y]-=2*epsilon
        y, z, P, J2, P_den = forwardprop(x, t, A, S, W)
        numerical_grad_W = (J-J2)/(2*epsilon)
        
        

        W[idx_x,idx_y]+=epsilon
       
        errW.append(np.abs(grad_W[idx_x, idx_y] - numerical_grad_W))
        # -------------------------------------------------
     

    print 'Gradient checking A, MAE: %0.15f' % Decimal(np.mean(errA))
    print 'Gradient checking S, MAE: %.15f' % np.mean(errS)
    print 'Gradient checking W, MAE: %.15f' % np.mean(errW)
    



if __name__ == '__main__':
    gradient_check()