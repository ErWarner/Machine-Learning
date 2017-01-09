from matplotlib import pyplot
import matplotlib as mpl
import numpy as np
from time import time
import pandas



#Import Data
data_train = pandas.read_csv("digits_training_data.csv", sep=",",header=None)
targets_train = pandas.read_csv("digits_training_labels.csv", sep=",",header=None)
data_test = pandas.read_csv("digits_test_data.csv", sep=",",header=None)
targets_test = pandas.read_csv("digits_test_labels.csv", sep=",",header=None)


targets_train=targets_train.rename(columns={0:'digit'})
targets_test=targets_test.rename(columns={0:'digit'})

train = [data_train,targets_train]
test = [data_test,targets_test]

train = pandas.concat(train,axis=1)
test  = pandas.concat(test,axis=1)



train4 = train[train['digit']==4]
train9 = train[train['digit']==9]

P4    = float(len(train4))/float(len(train))
P9    = 1.0-P4
test4 = test[test['digit']==4]
test9 = test[test['digit']==9]

Mean_Train4 = train4.mean()
Mean_Train9 = train9.mean()
Mean_Test4 = test4.mean()
Mean_Test9 = test9.mean()

matrix = data_train
mean =  np.mean(matrix,axis=0)
# make a mean matrix the same shape as data for subtraction
mean_mat = np.outer(np.ones((1000,1)),mean)

cov = matrix - mean_mat
Cov = np.dot(cov.T,cov)/(1000 -1)

MTr4 = np.asarray(Mean_Train4[0:676])
MTr9 = np.asarray(Mean_Train9[0:676])

print MTr4
print Mtr9



Gamma4 = -0.5*np.dot(np.dot(np.transpose(MTr4),np.linalg.pinv(Cov)),MTr4)
Gamma9 = -0.5*np.dot(np.dot(np.transpose(MTr9),np.linalg.pinv(Cov)),MTr9)



Beta4  = np.dot(np.linalg.pinv(Cov),MTr4)
Beta9  = np.dot(np.linalg.pinv(Cov),MTr9)

Tr4 = train4[0:676]
Tr9 = train9[0:676]
Te4 = test4[0:676]
Te9 = test9[0:676]

ProbTr4 = data_train.apply(lambda x: P4*np.exp(np.dot(np.transpose(Beta4),x)+Gamma4)/(np.exp(np.dot(np.transpose(Beta4),x)+Gamma4)+
    np.exp(np.dot(np.transpose(Beta9),x)+Gamma9)), axis=1)
ProbTr9 = data_train.apply(lambda x: P9*np.exp(np.dot(np.transpose(Beta9),x)+Gamma9)/(np.exp(np.dot(np.transpose(Beta4),x)+Gamma4)+
    np.exp(np.dot(np.transpose(Beta9),x)+Gamma9)), axis=1)
    
ProbTe4 = data_test.apply(lambda x: P4*np.exp(np.dot(np.transpose(Beta4),x)+Gamma4)/(np.exp(np.dot(np.transpose(Beta4),x)+Gamma4)+
    np.exp(np.dot(np.transpose(Beta9),x)+Gamma9)), axis=1)
ProbTe9 = data_test.apply(lambda x: P9*np.exp(np.dot(np.transpose(Beta9),x)+Gamma9)/(np.exp(np.dot(np.transpose(Beta4),x)+Gamma4)+
    np.exp(np.dot(np.transpose(Beta9),x)+Gamma9)), axis=1)
    
ProbTr=ProbTr4 
ProbTe=ProbTe4
ProbTr[ProbTr4>ProbTr9]=4
ProbTr[ProbTr4<ProbTr9]=9
ProbTe[ProbTe4>ProbTe9]=4
ProbTe[ProbTe4<ProbTe9]=9

Train_Error = 1.0-sum(targets_train['digit']==ProbTr)/float(len(targets_train))
Test_Error = 1.0-sum(targets_test['digit']==ProbTe)/float(len(targets_test))

print "The training accuracy is ", 1-Train_Error
print "The test accuracy is ", 1-Test_Error


Te = ProbTe -targets_test['digit']

p=1
Imag = []
for i in range(len(Te)):
    if (Te[i] != 0):
        Imag.append(i)
        p=p+1
        if p>5:
            break
            
            
           
            

def show_image(image):


    fig = pyplot.figure()
    ax = fig.add_subplot(1,1,1)
    imgplot = ax.imshow(image, cmap=mpl.cm.Greys)
    imgplot.set_interpolation('nearest')
    ax.xaxis.set_ticks_position('top')
    ax.yaxis.set_ticks_position('left')
    pyplot.show()
 
trainData = np.loadtxt('digits_training_labels.csv', dtype=np.uint8, delimiter=',')

    # Visualize sample image

   
def load_data():
    # load training data
    trainLabels = np.loadtxt('digits_training_labels.csv', dtype=np.uint8, delimiter=',')
    trainData = np.loadtxt('digits_training_data.csv', dtype=np.uint8, delimiter=',')

    # Visualize sample image
    u,v = np.shape(trainData)
    imgHeight = np.sqrt(v)
    #show_image(trainData[1].reshape((imgHeight, imgHeight)))

    # load test data
    testData = np.loadtxt('digits_test_data.csv', dtype=np.uint8, delimiter=',')
    u,v = np.shape(testData)
    imgHeight = np.sqrt(v)
    for i in Imag:
        show_image(testData[i].reshape((imgHeight, imgHeight)))



load_data()


