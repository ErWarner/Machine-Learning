from matplotlib import pyplot
import matplotlib as mpl
import numpy as np
from time import time
import pandas

# Import datasets, classifiers and performance metrics
from sklearn import datasets, svm, pipeline
from sklearn.kernel_approximation import (RBFSampler,
                                          Nystroem)
from sklearn.decomposition import PCA
from sklearn.kernel_ridge import KernelRidge
from sklearn import preprocessing
from sklearn.cross_validation import StratifiedShuffleSplit
from sklearn.grid_search import GridSearchCV

# The digits dataset
digits = datasets.load_digits(n_class=9)

#Import Data
data_train = pandas.read_csv("digits_training_data.csv", sep=",",header=None)
targets_train = pandas.read_csv("digits_training_labels.csv", sep=",",header=None)
data_test = pandas.read_csv("digits_test_data.csv", sep=",",header=None)
targets_test = pandas.read_csv("digits_test_labels.csv", sep=",",header=None)

targets_train[targets_train==4] = -1
targets_train[targets_train==9] = 1
targets_test[targets_test==4] = -1
targets_test[targets_test==9] = 1

data_train = np.asarray(data_train)
targets_train = np.asarray(targets_train)
data_test = np.asarray(data_test)
targets_test = np.asarray(targets_test)


len_tr  = len(targets_train)
len_te  = len(targets_test)

targets_train = np.reshape(targets_train,len_tr)
targets_test = np.reshape(targets_test,len_te)

scaler = preprocessing.StandardScaler().fit(data_train)
data_train = scaler.transform(data_train)
data_test  = scaler.transform(data_test)

kernel_svm = svm.SVC(kernel='rbf',C=3,gamma=1e-10,tol=1e-6)
kernel_svm.fit(data_train, targets_train)
kernel_svm_score_train = kernel_svm.score(data_train, targets_train)
kernel_svm_score = kernel_svm.score(data_test, targets_test)

print kernel_svm_score



C_range = np.logspace(6, 18, 10)
gamma_range = np.logspace(-15, -5, 10)
param_grid = dict(gamma=gamma_range, C=C_range)
cv = StratifiedShuffleSplit(targets_train, n_iter=5, test_size=0.2, random_state=42)
grid = GridSearchCV(svm.SVC(), param_grid=param_grid, cv=cv)

grid.fit(data_train, targets_train)
gridB = grid.best_params_
gam = gridB['gamma']
C2   = gridB['C']

kernel_svm = svm.SVC(C=C2,gamma=gam, kernel='rbf')
kernel_svm.fit(data_train, targets_train)
kernel_svm_score_train = kernel_svm.score(data_train, targets_train)
kernel_svm_score = kernel_svm.score(data_test, targets_test)

print "The training score is ",kernel_svm_score_train
print "The test score is ",kernel_svm_score


ProbTe = kernel_svm.predict(data_test)
Te = ProbTe -targets_test


p=1
Imag = []

for i in range(0,499):
	if (Te[i]==2):
		Imag.append(i)
		p=p+1
		if p>5:
			break
	elif Te[i]==-2:
		Imag.append(i)
		p=p+1
		if p>5:
			break
            
print Imag            
            
            

def show_image(image):
    """
    Render a given numpy.uint8 2D array of pixel data.
    """
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
