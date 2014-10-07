#! /usr/bin/python
#
# This is support material for the course "Learning from Data" on edX.org
# http://lionoso.org/learningfromdata/
#
# The software is intented for course usage, no guarantee whatsoever
# Date: Sep 24, 2014
#
# Template for a LIONoso parametric table script.
#
# Generates a table based on input parameters taken from another table or from user input
#
# Syntax:
# When called without command line arguments:
#    number_of_inputs
#    name_of_input_1 default_value_of_input_1
#    ...
#    name_of_input_n default_value_of_input_n
# Otherwise, the program is invoked with the following syntax:
#    script_name.py input_1 ... input_n table_row_number output_file.csv
# where table_row_number is the row from which the input values are taken (assume it to be 0 if not needed)
#
# To customize, modify the output message with no arguments given and insert task-specific code
# to insert lines (using tmp_csv.writerow) in the output table.

import sys
import os
from numpy import random
import numpy
import math

#
# If there are not enough parameters, optionally write out the number of required parameters
# followed by the list of their names and default values. One parameter per line,
# name followed by tab followed by default value.
# LIONoso will use this list to provide a user friendly interface for the component's evaluation.
#
if len(sys.argv) < 2:
	sys.stdout.write ("2\nNumber of tests\t1000\nNumber of training points\t10\n")
	sys.exit(0)

#
# Retrieve the input parameters, the input row number, and the output filename.
#
in_parameters = [float(x) for x in sys.argv[1:-2]]
in_rownumber = int(sys.argv[-2])
out_filename = sys.argv[-1]

#
# Retrieve the output filename from the command line; create a temporary filename
# and open it, passing it through the CSV writer
#
tmp_filename = out_filename + "_"
tmp_file = open(tmp_filename, "w")

#############################################################################
#
# Task-specific code goes here.
#

def sign(x):
	return 1 if x>=0 else -1

def mod(v):
	return math.sqrt(sum(v*v))

def dot(v1,v2):
	return sum(v1*v2)/(mod(v1)*mod(v2))

def classify(x,y,slope,intercept):
	m=y/x
	x_=intercept/(m-slope)
	y_=m*x_
	v1=numpy.array([x,y])
	v2=numpy.array([x_,y_])
	r=mod(v1)
	r_=mod(v2)
	return -1 if(dot(v1,v2))<0 else sign(r-r_)

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
			p = sum(W * v)
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
		predictions.append(sign(sum(v_*W)))
	return numpy.array(predictions)

def accuracy(Y,Y_pred):
	test=[]
	for i in range(0,len(Y)):
		test.append(1 if Y[i]==Y_pred[i] else 0)
	return numpy.mean(test)

# The following function is a stub for the perceptron training function required in Exercise1-7 and following.
# It currently generates random results.
# You should replace it with your implementation of the
# perceptron algorithm (we cannot do it otherwise we solve the homework for you :)
# This functon takes the coordinates of the two points and the number of training samples to be considered.
# It returns the number of iterations needed to converge and the disagreement with the original function.
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


tests = int(in_parameters[0])
points = int(in_parameters[1])

# Write the header line in the output file, in this case the output is a 3-columns table containing the results
# of the experiments
# The syntax  name::type  is used to identify the columns and specify the type of data
header = "Test number::label,Number of iterations::number,Disagreement::number\n"
tmp_file.write (header)


# Repeat the experiment n times (tests parameter) and store the result of each experiment in one line of the output table
for t in range(1,tests+1):
	x1 = random.uniform (-1, 1)
	y1 = random.uniform (-1, 1)
	x2 = random.uniform (-1, 1)
	y2 = random.uniform (-1, 1)
	iterations, disagreement = perceptron_training (x1, y1, x2, y2, points)
	line = str(t) + ',' + str(iterations) + ',' + str(disagreement) + '\n'
	tmp_file.write (line)

#
#############################################################################

#
# Close all input files and the temporary output file.
#
tmp_file.close()

#
# Rename the temporary output file into the final one.
# It's important that the output file only appears when it is complete,
# otherwise LIONoso might read an incomplete table.
#
os.rename (tmp_filename, out_filename)