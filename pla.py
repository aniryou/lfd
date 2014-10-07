#! /usr/bin/python

import sys
from numpy import random, linalg
import numpy
import math
import matplotlib.pyplot as plt

def sign(x):
	return 1 if x>=0 else -1

def classify(x,y,slope,intercept):
	m=y/x
	x_=intercept/(m-slope)
	y_=m*x_
	v1=numpy.array([x,y])
	v2=numpy.array([x_,y_])
	if(numpy.dot(v1,v2)<0):
		return -1
	r=linalg.norm(v1)
	r_=linalg.norm(v2)
	return sign(r-r_)

def generate(N, slope, intercept):
	X=random.uniform(-1.,1.,size=(N,2))
	Y=numpy.empty(N)
	for i in range(0,N):
		Y[i]=classify(*X[i],slope=slope,intercept=intercept)
	return X,Y

def pla(X,Y):
	from random import shuffle
	W=numpy.zeros(3)
	itercount=0
	while True:
		misclassified = []
		indices = range(0,len(X))
		shuffle(indices)
		for i in indices:
			x,y=X[i]
			v = numpy.array([1,x,y])
			p = numpy.dot(W,v)
			clf = sign(p)
			if(clf!=Y[i]):
				itercount += 1
				misclassified.append(v)
				W = W - clf*v
		if(len(misclassified)==0):
			return itercount, W
		
def predict(X,W):
	predictions=[]
	for v in X:
		v_=numpy.concatenate(([1], v))
		predictions.append(sign(numpy.dot(v_,W)))
	return numpy.array(predictions)

def accuracy(Y,Y_pred):
	test=[]
	for i in range(0,len(Y)):
		test.append(1 if Y[i]==Y_pred[i] else 0)
	return numpy.mean(test)

def perceptron_training (x1, y1, x2, y2, training_size):
	slope=(y2-y1)/(x2-x1)
	intercept=y1-slope*x1
	X,Y=generate(2*training_size,slope,intercept)
	X_train=X[:training_size]
	Y_train=Y[:training_size]
	X_test=X[training_size:]
	Y_test=Y[training_size:]
	iter_count,W=pla(X_train,Y_train)
	Y_pred=predict(X_test,W)
	disagreement=1-accuracy(Y_test,Y_pred)
	return (iter_count, disagreement)

def main(tests, sample_size, sample_tests=100):
	iteration_counts = []
	disagreements = []
	for t in range(1,tests+1):
		x1 = random.uniform (-1, 1)
		y1 = random.uniform (-1, 1)
		x2 = random.uniform (-1, 1)
		y2 = random.uniform (-1, 1)
		for j in range(1, sample_tests+1):
			iteration_count, disagreement = perceptron_training (x1, y1, x2, y2, sample_size)
			iteration_counts.append(iteration_count)
		disagreements.append(disagreement)
	print numpy.mean(iteration_counts), numpy.mean(disagreements)

if __name__=='__main__':
	if(len(sys.argv)<3):
			print "usage: python pla.py NUM_TESTS NUM_TRAIN"
			sys.exit(-1)
	NUM_TESTS = int(sys.argv[1])
	NUM_TRAIN = int(sys.argv[2])
	main(NUM_TESTS, NUM_TRAIN)