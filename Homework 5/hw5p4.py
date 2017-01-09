import numpy as np
import pandas
import sklearn
import cPickle
from collections import Counter
import math
import scipy
import scipy.ndimage


#Import Data
digits = pandas.read_csv("train_noised.csv",header=None)
clean = pandas.read_csv("train_clean.csv",header=None)
test = pandas.read_csv("test_noised.csv",header=None)

						   
(r,c) = np.shape(digits)
del digits[0]
del clean[0]
del test[0]


Final=[]
Error = 0
Error2 = 0
ErrorT=[]

cleaned = np.array(clean[:][1:])
cleaned = cleaned.astype(float)  
cleaned = cleaned/255.0

digit = np.array(digits[:][1:])
digit = digit.astype(float)  


image = np.array(digits[:][1:])       
image = image.astype(float)

image = image/255.0
imfft = np.fft.fft2(image)





  
for i in range(imfft.shape[0]):

    kx = i/float(imfft.shape[0])
    if kx>0.5: 
        kx = kx-1
	
    for j in range(imfft.shape[1]):
        
        ky = j/float(imfft.shape[1])
        if ky>0.5: 
            ky = ky-1
				
			# Get rid of all the low frequency stuff - in this case, features whose wavelength is larger than about 20 pixels
        if (kx*kx + ky*ky< 0.015*0.015):
            imfft[i,j] = 0
        

# Transform back


newimage = 1.0*((np.fft.ifft(imfft)).real)
#newimage  = 15*newimage/np.max(newimage)


	
newimage = np.minimum(newimage, 1.0)
newimage = np.maximum(newimage, 0.0)

newimage = newimage*255.0
cleaned  = cleaned*255.0    

Error =np.linalg.norm(newimage-cleaned,'fro')/np.sqrt(r*c)
print Error
	
#newimage=newimage*255.0

"""   

Gauss = scipy.ndimage.percentile_filter(image,50,(50,50))
GH    = image-Gauss
GH    = GH

GH    = np.minimum(GH,1.0)
GH    = np.maximum(GH,0.0)
GH    = GH*255.0


#---------------TEST DATA------------------#

test_image = np.array(test[:][1:])       
test_image = test_image.astype(float)
test_image = test_image/255.0



Gauss2 = scipy.ndimage.gaussian_filter(test_image,100)
(l,w)  = np.shape(Gauss2)
GH2    = test_image-Gauss2
GH2    = GH2

GH2    = np.minimum(GH2,1.0)
GH2    = np.maximum(GH2,0.0)
GH2    = GH2*255.0





    
Test = np.reshape(GH2,(l*w))



p=0
Names=[]

for i in range(0,l):
    for j in range(0,w):
        row = str(i)
        col = str(j)
        Names.append(row+'_'+col)



#ErrorG =np.linalg.norm(GH-cleaned,'fro')/np.sqrt(r*c)


#print ErrorG
   
#Write to a Dataframe
df = pandas.DataFrame(Test, Names,columns=['Val'])

#df.to_csv('erwaner_test_data.csv')        
"""   




	
