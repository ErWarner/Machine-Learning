
import numpy as np
import itertools
import operator


pi = np.array([0.5,0.3,0.2])
A  = np.matrix([[0.5, 0.2,0.3],[0.2,0.4,0.4],[0.4,0.1,0.5]])
phi = np.matrix([[0.8,0.2],[0.1,0.9],[0.5,0.5]]) 

obs  = np.array([0,1,0,1])


##FIND MOST PROBABLE SEQUENCE

x = [0,1,2]
It = [p for p in itertools.product(x, repeat=4)]
Total={}
Total_Prob=0

for i in It:
    string = str(i)
    Prob = pi[i[0]]*phi[i[0],obs[0]]
    Uncond_Prob = pi[i[0]]
    Likelihood = phi[i[0],obs[0]]
    for j in range(1,4):
        prev = i[j-1]
        current = i[j]
        Prob = A[prev,current]*phi[current,obs[j]]*Prob
        Uncond_Prob = A[prev,current]*Uncond_Prob
        Likelihood  = Likelihood*phi[current,obs[j]]
        
    Total_Prob=Total_Prob+Prob
    Total[string] = (Prob,Uncond_Prob,Likelihood)

Result = {key:value[0]/Total_Prob for key, value in Total.items()}    
    


sorted_Total = sorted(Result.items(), key=operator.itemgetter(1),reverse=True)
print sorted_Total[0:3]
print Total_Prob