from __future__ import division
import numpy as np
import itertools 
import operator
import matplotlib.pyplot as plt

# Generate the data according to the specification in the homework description
# for part (b)

A0 = np.array([[0.5, 0.2, 0.3], [0.2, 0.4, 0.4], [0.4, 0.1, 0.5]])
phi0 = np.array([[0.8, 0.2], [0.1, 0.9], [0.5, 0.5]])
pi0 = np.array([0.5, 0.3, 0.2])

X = []

for _ in xrange(5000):
	z = [np.random.choice([0,1,2], p=pi0)]
	for _ in range(3):
		z.append(np.random.choice([0,1,2], p=A0[z[-1]]))
	x = [np.random.choice([0,1], p=phi0[zi]) for zi in z]
	X.append(x)

# TODO: Implement Baum-Welch for estimating the parameters of the HMM


p     = 0
Alpha = []
sum   = np.zeros((3,1))
Beta  = [] 
Xi    = []
Gamma = []
Num = np.zeros((3,3))
Den = np.zeros((3,3))
Nump = 0
Denp = 0

#------------INITIALIZE PARAMETERS---------------#

Ai = np.random.uniform(0,1,(3,3))
row_sums = Ai.sum(axis=1)
Ai        = Ai / row_sums[:, np.newaxis]


phii = np.random.uniform(0,1,(3,2))
row_sums = phii.sum(axis=1)
phii        = phii / row_sums[:, np.newaxis]


pii = np.random.uniform(0,1,(1,3))
pii = pii/np.sum(pii)
pii = pii[0]


#Number of observation points
N  = np.array([500,1000,2000,5000])
#Initialize error arrays
Error =[]
Errors=[]
#Number of iterations of EM array
Finish = 50

x_seq = list(itertools.product([0,1], repeat=4))
z_seq = list(itertools.product([0,1,2], repeat=4))

for j in range(0,4):


    Total = N[j]
	#Initial parameters
    A  = Ai
    phi = phii
    pi  = pii
    Error=[]
    for n in range(0,Finish):
        print"--------------------------------------------------"
        print n
        print Total
        Num = np.zeros((3,3))
        Den = np.zeros((3,3))
        Nump = 0
        Denp = 0
        Num_phi = np.zeros((3,2))
        Den_phi = np.zeros((3,1))
        Total_Prob = 0
        Est_Prob=0
        Diff   = 0
        
        
                    
        for p in range(0,Total):
            Alpha=[]
            Beta=[]
            Gamma=[]
            Xi=[]
            obs = X[p]   
            PX2   = 0
            for i in z_seq:   
                Prob2 = pi[i[0]]*phi[i[0],obs[0]]
                for j in range(1,4):
                    prev = i[j-1]
                    current = i[j]
                    Prob2 *= A[prev,current]*phi[current,obs[j]]

                PX2+=Prob2


    #--------------Initialize Alpha, Beta-------------------#
            alpha = np.reshape(pi*phi[:,obs[0]],(3,1))   
            Alpha.append(alpha)
        
            beta       = np.ones((3,1))        
            Beta.append(beta)

    #---------------FORWARD PROCEDURE------------------------
            for i in range(1,4):
                for j in range(0,3): 
                    sum[j]     = np.dot(np.transpose(alpha),A[:,j]) 
    
                alpha = np.transpose(phi[:,obs[i]]*np.transpose(sum))            
                Alpha.append(alpha)
            
    #---------------BACKWARD PROCEDURE-------------------------
            for i in reversed(range(0,3)):
                sum  = np.zeros((3,1))
                for k in range(0,3):
                    for j in range(0,3):
                        sum[k]+= beta[j]*A[k,j]*phi[j,obs[i+1]]

            
                beta = np.reshape(sum,(3,1))
                Beta.append(beta)
        
    
            Beta= Beta[::-1]
    
    #---------------TEMPORARY VALUES-------------------------
        
    
            for i in range(0,4):
                gamma = np.transpose(Alpha[i])*np.transpose(Beta[i])
                #gamma = gamma/np.sum(gamma)
                gamma = gamma/PX2
                Gamma.append(gamma)
        
                if i!=3:
                    xi = np.zeros((3,3))
                    for k in range(0,3):
                        for j in range(0,3):
                            xi[k,j] = Alpha[i][k]*A[k,j]*Beta[i+1][j]*phi[j,obs[i+1]]                  
                    #Xi.append(xi/np.sum(Alpha[3]))
                    Xi.append(xi/PX2)
                
                
    #-----SOLVE FOR NUMERATOR AND DENOMINATOR TERMS OF PARAMETERS-------    
            Nump += Gamma[0][0]
            Denp += np.sum(Gamma[0][0])     
        
    
            for i in range(0,3):
                for j in range(0,3):
                    for m in range(0,3):
                        Num[i,j]+=Xi[m][i,j]
                        Den[i,j]+=Xi[m][i,0]
                        Den[i,j]+=Xi[m][i,1]
                        Den[i,j]+=Xi[m][i,2]
                    
        
            for i in range(0,4):        
                if obs[i]==0:
                    Num_phi[0,0] += Gamma[i][0][0]
                    Num_phi[1,0] += Gamma[i][0][1]
                    Num_phi[2,0] += Gamma[i][0][2]
                else:
                    Num_phi[0,1] += Gamma[i][0][0]
                    Num_phi[1,1] += Gamma[i][0][1]
                    Num_phi[2,1] += Gamma[i][0][2]       
		
            Phi_Sum   = np.sum(Gamma,axis=0) 
         
        
            Den_phi[0] += Phi_Sum[0][0]
            Den_phi[1] += Phi_Sum[0][1]
            Den_phi[2] += Phi_Sum[0][2]
        

        
		#----------FIND P(X) AND P(X')-----------	
        x_seq = list(itertools.product([0,1], repeat=4))
        z_seq = list(itertools.product([0,1,2], repeat=4))

        Diff = 0
        for x in x_seq:
            PXP=0
            PX=0
            for i in z_seq:
                Prob = pi0[i[0]]*phi0[i[0],x[0]]     
                Prob2 = pi[i[0]]*phi[i[0],x[0]]
                for j in range(1,4):
                    prev = i[j-1]
                    current = i[j]
                    Prob *= A0[prev,current]*phi0[current,x[j]]
                    Prob2 *= A[prev,current]*phi[current,x[j]]
        
                PXP+=Prob
                PX+=Prob2
            Diff  +=abs(PXP-PX)/2       
        

        Error.append(Diff)
		#-----------SOLVE FOR NEW PARAMETERS------------
        A   = np.divide(Num,Den)
        phi = np.divide(Num_phi,Den_phi)
        pi  = np.divide(Nump,Denp)   
    
	
		#print pi
		#print "................"
		#print A 
		#print "................"
		#print phi
		#print "................"
        print Diff
    Errors.append(Error)

    
### Plot the results

n=range(0,Finish)

for i in xrange(0, 4):
  plt.plot(n, Errors[i], label = "N = " + str(N[i]), linewidth=2)


plt.xlabel("Iteration" )
plt.ylabel("Difference")
plt.legend(loc='upper right')
plt.show()
    
    











