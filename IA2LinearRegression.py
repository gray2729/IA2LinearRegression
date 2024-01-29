"""
File name:  LinearRegression.py
Author:     Isaac Gray
Date:  	    02/12/2023
Class: 	    DSCI 440 ML
Assignment: IA 2
Purpose:    This program takes a data set and finds the weight vector 
            optimizes the Sum on Square Error.
"""

import numpy as np
import matplotlib.pyplot as pyplot

"""
Function:    fReadfile
Description: reads the contents of a file, seperating the all but 
             last columns into a matrix array and the last column 
             into a vector array
Input:       filename - the name of the file that will be read
Output:      X - matrix that holds the features
             Y - vector that holds the target feature
"""
def fReadfile(fileName):
    X = []
    Y = []
    
    file = open(fileName, "r")

    for line in file:
        features = [1]
        line = line.split()
        for num in range(len(line)-1):
            features.append(float(line[num]))
        X.append(features)
        Y.append(float(line[-1]))

    file.close()

    X = np.array(X)
    Y = np.array(Y)
    
    return X, Y

"""
Function:    fDeterminew
Description: determines vector w through normal equations
Input:       X - matrix holding the features
             y - vector holding the true values of the training set
Output:      w - vector holding the parameters for the 
                 linear function
"""
def fDeterminew(X, y):
    XTranspose = np.transpose(X)
    XInverse = np.linalg.inv(np.matmul(XTranspose, X))
    
    w = np.matmul(XTranspose, y)
    w = np.matmul(XInverse, w)
    return w



"""
Function:    fDetermineSSE
Description: determines the Sum of Square Error
Input:       X - matrix holding the features
             y - vector holding the true values of the training set
             w - vector holding the parameters for the 
                 linear function
Output:      SSE - value of objective function J(w)
"""
def fDetermineSSE(X, y, w):
    wInverse = np.transpose(w)
    runningSum = 0
    for index in range(y.size):
        runningSum += (y[index] - np.matmul(wInverse, X[index]))**2
    SSE = (1/2)*runningSum
    return (SSE)



"""
Function:    fL2w
Description: determines optimal vector w through normal equations 
             with regulatization for certain lambda
Input:       X - matrix holding the features
             y - vector holding the true values of the training set
             lamda - the lamdba for regularization
Output:      w - vector holding the parameters for the 
                 linear function
"""
def fL2w(X, y, lamda):
    XTranspose = np.transpose(X)
    RegXMatrix = np.matmul(XTranspose, X)
    
    matrixSize = np.size(RegXMatrix, 1)
    Identity = np.identity(matrixSize)
    RegTerm = lamda*Identity
    
    XInverse = np.linalg.inv(RegTerm + RegXMatrix)
    
    w = np.matmul(XTranspose, y)
    w = np.matmul(XInverse, w)
    return w



"""
Function:    fL2SSE
Description: calculates the SSE of the regularized data
Input:       X - matrix holding the features
             y - vector holding the true values of the training set
             w - vector holding the parameters for the 
                 linear function
             lamda - the lambda for the regularization term
Output:      SSE - the SSE of the regularized data which comes from
                   the objective function J(w) and the 
                   regularization term
"""
def fL2SSE(X, y, w, lamda):
    wInverse = np.transpose(w)
    RegSSE = 0
    RegTerm = 0
    
    for index in range(y.size):
        RegSSE += (y[index] - np.matmul(wInverse, X[index]))**2
    for index in range(w.size):
        RegTerm += (w[index])**2
    
    SSE = (1/2)*(RegSSE+lamda*RegTerm)
    return SSE



"""
Function:    main
Description: Opens and retieves the data from file, finds the weight vector w 
             and Sum of Squares Error SSE for training and testing data
Input:       None
Output:      None
"""


trainingFile = "housing_train.txt"
testingFile = "housing_test.txt"

trainingX, trainingY = fReadfile(trainingFile)
testingX, testingY = fReadfile(testingFile)

w = fDeterminew(trainingX, trainingY)
trainingSSE = fDetermineSSE(trainingX, trainingY, w)
testingSSE = fDetermineSSE(testingX, testingY, w)

print("Optimal w:", w)
print("Training SSE: ", trainingSSE)
print("Testing SSE: ", testingSSE)

lambdaRange = [0.00001, 0.0001, 0.001, 0.01, 0.1, 1, 10, 100, 1000, 10000]
trainSSEs = []
trainSSEReg = []
testSSEs = []
testSSEReg = []
eucNorms = []

for lamda in lambdaRange:
    w = fL2w(trainingX, trainingY, lamda)
    
    trainingSSE = fL2SSE(trainingX, trainingY, w, lamda)
    trainSSEReg.append(trainingSSE)
    
    trainingSSE = fDetermineSSE(trainingX, trainingY, w)
    trainSSEs.append(trainingSSE)
    
    testingSSE = fL2SSE(testingX, testingY, w, lamda)
    testSSEReg.append(testingSSE)
    
    testingSSE = fDetermineSSE(trainingX, trainingY, w)
    testSSEs.append(testingSSE)
    
    
    wSum = 0
    for index in range(w.size):
        wSum += (w[index])**2
    eucNorms.append(wSum**(1/2))


pyplot.scatter(lambdaRange, trainSSEReg, label='Weighted', c='red')
pyplot.scatter(lambdaRange, trainSSEs, label='Unweighted', c='blue')
pyplot.xlabel('Lambda')
pyplot.ylabel('SSE values')
pyplot.title('Training SSEs vs Lambda')
pyplot.legend()
pyplot.xscale("log")
pyplot.show()

pyplot.scatter(lambdaRange, testSSEReg, label='Weighted', c='red')
pyplot.scatter(lambdaRange, testSSEs, label='Unweighted', c='blue')
pyplot.xlabel('Lambda')
pyplot.ylabel('SSE values')
pyplot.title('Testing SSEs vs Lambda')
pyplot.legend()
pyplot.xscale("log")
pyplot.show()

pyplot.scatter(lambdaRange, eucNorms, label='Euc Norms of ws')
pyplot.xlabel('Lambda')
pyplot.ylabel('Euc Norms of ws')
pyplot.title('Weight Values of ws')
pyplot.xscale("log")
pyplot.show()