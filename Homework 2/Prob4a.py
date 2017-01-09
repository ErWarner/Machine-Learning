import numpy as np
import pandas
from collections import Counter
import math
import re

#import the data
train = pandas.read_csv("spambase.train", sep=',',
		names=np.array(map(str,range(58))))
		
test = pandas.read_csv("spambase.test", sep=',',
		names=np.array(map(str,range(58))))
		
nums = range(57)
names = np.array(map(str,nums))

#Set values above the median to 2, and below to 1
for col in names:
	
	med = float(np.median(train[col]))
	
	column = train[col]
	col2 = column-med
	column.loc[col2>0]=2
	column.loc[col2<=0]=1
	train.loc[:,col]=column
	
	column2 = test[col]
	col3 = column2-med
	column2.loc[col3>0]=2
	column2.loc[col3<=0]=1
	test.loc[:,col]=column2
	
#Separate into Normal and Spam
Normal = train[train["57"]==0]
Spam   = train[train["57"]==1]

#Find Probabilities of Normal and Spam
PN = float(len(Normal))/float(len(train))
PS = float(len(Spam))/float(len(train))

print PN
print PS
P1GN=dict()
P2GN=dict()
P1GS=dict()
P2GS=dict()
P1 = dict()
P2 = dict()

#Compute the conditional probabilities
for col in names:
	P1GN[col] = float(len(Normal[Normal[col]==1]))/float(len(Normal))
	P2GN[col] = float(len(Normal[Normal[col]==2]))/float(len(Normal))
	P1GS[col] = float(len(Spam[Spam[col]==1]))/float(len(Spam))
	P2GS[col] = float(len(Spam[Spam[col]==2]))/float(len(Spam))
P1   = float(len(train[train[col]==1]))/float(len(train))
P2   = float(len(train[train[col]==2]))/float(len(train))
 
Decision = []
Error    = []	

#Compute the Naive Bayes
for line in range(len(test)):
	PN1 = P1
	PS1 = P2	
	for col in names:
		C= test[col]
		if test[col][line]==1:
			PN1 = PN1*P1GN[col]
			PS1 = PS1*P1GS[col]
		else:
			PN1 = PN1*P2GN[col]
			PS1 = PS1*P2GS[col]
	EV = PN*PN1+PS*PS1
	PNormal = PN1/EV
	PSpam   = PS1/EV
	if PNormal>PSpam:
		d=0
		Decision.append(d)
	else:
		d=1
		Decision.append(d)
	Error.append(abs(d-test["57"][line]))
	
print "The misclassification percentage is ", float(sum(Error))/float(len(test))
print "The percentage of spam is ",float(sum(test["57"]))/float(len(test))
print "The percentage of normal email is ",1.0-float(sum(test["57"]))/float(len(test))
print "If I had guessed normal email every time, I would be correct ", 1.0-float(sum(test["57"]))/float(len(test)), " of the time"
print "Therefore the algorithm improves the results ",(1.0-float(sum(Error))/float(len(test)))-(1.0-float(sum(test["57"]))/float(len(test))), " percent"
