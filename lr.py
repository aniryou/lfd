#! /usr/bin/python

import sys
from numpy import random, linalg
import numpy
import math
from numba import autojit

@autojit
def sign(x):
	return 1 if x>=0 else -1

@autojit
def classify(x,y,slope,intercept):
	'''
	classify point (x,y) against line (y = slope*x + intercept)
	point in region with origin is classified negative
	point in region without origin is classified positive
	'''
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
	
@autojit	
def predict(X,W):
	'''
	predict classification of points in X, using perceptron params W
	'''
	predictions=[]
	for v in X:
		v_=numpy.concatenate(([1], v))
		predictions.append(sign(numpy.dot(v_,W)))
	return numpy.array(predictions)

@autojit
def accuracy(Y_act,Y_pred):
	'''
	perceptron accuracy, Y_actual vs Y_predicted
	'''
	test=[]
	for i in range(0,len(Y_act)):
		test.append(1 if Y_act[i]==Y_pred[i] else 0)
	return numpy.mean(test)

@autojit
def generate_data(N, slope, intercept):
	'''
	generate N points classified as positive/negative
	using line defined by slope and intercept
	'''
	X=random.uniform(-1.,1.,size=(N,2))
	Y=numpy.empty(N)
	for i in range(0,N):
		Y[i]=classify(*X[i],slope=slope,intercept=intercept)
	return X,Y

@autojit
def linear_regression(X,Y):
	'''
	Linear regression with least squares
	'''
	M = numpy.asmatrix(numpy.c_[numpy.ones(X.shape[0]), X])
	W = M.T.dot(M).I.dot(M.T).dot(Y).A1
	return W

@autojit
def train_test(x1, y1, x2, y2, sample_size, test_size):
	'''
	train perceptron over given line, for sample_size points
	'''
	slope=(y2-y1)/(x2-x1)
	intercept=y1-slope*x1
	X,Y=generate_data(sample_size+test_size,slope,intercept)
	X_train=X[:sample_size]
	Y_train=Y[:sample_size]
	X_test=X[sample_size:]
	Y_test=Y[sample_size:]
	W=linear_regression(X_train,Y_train)
	Y_pred_train=predict(X_train,W)
	Y_pred_test=predict(X_test,W)
	E_in=1-accuracy(Y_train,Y_pred_train)
	E_out=1-accuracy(Y_test,Y_pred_test)
	return (E_in, E_out)

def main(num_expt, sample_size, num_test=100, test_size=1000):
	'''
	train perceptron using sample_size points, test sample_tests*tests times
	output average number of iterations, average disagreement with target function
	'''
	iteration_counts = []
	E_in = []
	E_out = []
	for t in range(num_expt):
		x1 = random.uniform (-1, 1)
		y1 = random.uniform (-1, 1)
		x2 = random.uniform (-1, 1)
		y2 = random.uniform (-1, 1)
		for j in range(num_test):
			ein, eout = train_test(x1, y1, x2, y2, sample_size, test_size)
			E_in.append(ein)
			E_out.append(eout)
	print numpy.mean(E_in), numpy.mean(E_out)

if __name__=='__main__':
	if(len(sys.argv)<5):
			print "usage: python pla.py NUM_EXPT TRAIN_SIZE NUM_TEST TEST_SIZE"
			sys.exit(-1)
	NUM_EXPT = int(sys.argv[1])
	TRAIN_SIZE = int(sys.argv[2])
	NUM_TEST = int(sys.argv[3])
	TEST_SIZE = int(sys.argv[4])
	main(NUM_EXPT, TRAIN_SIZE, NUM_TEST, TEST_SIZE)
