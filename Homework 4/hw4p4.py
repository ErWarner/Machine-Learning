from __future__ import division
from scipy.ndimage import imread
import numpy as np
from matplotlib import pyplot as plt
import random

# Load the mandrill image as an NxNx3 array. Values range from 0.0 to 255.0.
mandrill = imread('mandrill.png', mode='RGB').astype(float)
N = int(mandrill.shape[0])

M = 2
k = 64

# Store each MxM block of the image as a row vector of X
X = np.zeros((N**2//M**2, 3*M**2))
for i in range(N//M):
    for j in range(N//M):
        X[i*N//M+j,:] = mandrill[i*M:(i+1)*M,j*M:(j+1)*M,:].reshape(3*M**2)

# TODO: Implement k-means and cluster the rows of X, then reconstruct the
# compressed image using the cluster center for each block, as specified in
# the homework description.


Mean    = np.random.choice(range(0,len(X)),size=(k,1),replace=False)
Initial = (X[Mean])


Diff = X[0]-Initial
Norm = np.sum(np.abs(Diff)**2,axis=-1)**(1./2)
Index  = np.where(Norm==Norm.min())
Num  = Index[0]



p=1

Count = np.zeros((1,k))
Total = np.zeros((k,3*M**2))
ErrorT=[]

while p<2:
    Count = 0.00001*np.ones((k,1))
    Error = 0
    Total = np.zeros((k,3*M**2))
    Final = np.zeros((len(X),3*M**2))
    for i in range(0,len(X)):
        xi    = np.array(X[i])
        Diff  = xi-Initial
        Norm  = np.sum(np.abs(Diff)**2,axis=-1)**(1./2)
        Index = np.where(Norm==Norm.min())
        Error = Error+Norm[Index]
        Num   = Index[0]
        Count[Num] = Count[Num]+1
        Total[Num] = Total[Num]+xi
        Final[i] = Initial[Num][0]
        
    Initial  = np.divide(Total,Count)
    ErrorT.append(Error[0])
    p=p+1


print np.shape(Final)
    
mandrill2 = np.zeros(np.shape(mandrill))    
for i in range(N//M):
    for j in range(N//M):
        mandrill2[i*M:(i+1)*M,j*M:(j+1)*M,:]=Final[i*N//M+j,:].reshape(M,M,3)    

(length,width,height) = np.shape(mandrill)

Abs_Error=0

for i in range(0,length):
    for j in range(0,width):
        for r in range(0,height):
            Abs_Error = abs(mandrill[i][j][r]-mandrill2[i][j][r])
 
Abs_Error = Abs_Error/(3*255*N**2)
print "The Relative Mean of the Absolute Error is",Abs_Error
            
    
# To show a color image using matplotlib, you have to restrict the color
# color intensity values to between 0.0 and 1.0. For example,

print np.shape(ErrorT)
print type(ErrorT)
print ErrorT[0:10]
print np.vstack(ErrorT[0:10])
plt.plot(range(len(ErrorT)),ErrorT)
plt.xlabel('Iteration')
plt.ylabel('K Means Objective Function')
plt.show()
plt.imshow(mandrill2/mandrill2.max())
plt.show()
plt.imshow(mandrill/255)
plt.show()
plt.imshow(mandrill/255-mandrill2/mandrill2.max()+(128,128,128))
plt.show()
