#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May  9 06:59:15 2018

@author: Shivendra
This is pereceptron illustration code for training
Solves an arbutrary classification problem
"""

import numpy as np
import matplotlib.pyplot as plt
#%matplotlib inline

#The actual input data is first two columns. 
#The third one is bias added for the perceptron model
inputs = np.array([
    [-3,4,-1],
    [2,1,-1],
    [4, -1, -1],
    [2, 5, -1],
    [7, 4, -1],

])

#This is the actual output expected from percepron for above dataset
#This perceptron solves a classification problem in two dimension
y = np.array([-1,-1,-1,1,1])

#Lets visualize the dataset with expected outcome
for d, sample in enumerate(inputs):
    # Plotting the negative outcomes
    if d < 3:
        plt.scatter(sample[0], sample[1], s=120, marker='_', linewidths=2)
    # Here we Plot the positive samples
    else:
        plt.scatter(sample[0], sample[1], s=120, marker='+', linewidths=2)

# Print a possible hyperplane, that is seperating the two classes.
plt.plot([-2,6],[7,1])

#Stochastic Gradient descent function for the perceptron
def perceptron_sgd_function(X, Y):
    '''
    This function trains perceptron and plots the total loss in each epoch.
    
    :param X: data samples
    :param Y: data labels
    :return: weight vector(weights) as a numpy array
    '''
    w = np.zeros(len(inputs[0]))
    eta = 1
    epochs = 20

    for t in range(epochs):
        for i, x in enumerate(X):
            print("Before weight change: ")
            print(inputs[i], w, Y[i])
            print("My SUm product function * target class value is....")
            print((np.dot(X[i], w)*Y[i]))
            if (np.dot(X[i], w)*Y[i]) <= 0:
                print(np.dot(X[i], w)*Y[i])
                print("Negative product for: ..")
                print(X[i], w, Y[i])
                w = w + eta*X[i]*Y[i]
                print("Changed weight: ..")
                print(X[i], w)
                input("enter any key to continue")

    return w

w = perceptron_sgd_function(inputs,y)
print(w)

def perceptron_sgd_plot(X, Y):
    '''
    This function trains perceptron and plots the total loss in each epoch.
    
    :param X: data samples
    :param Y: data labels
    :return: weight vector(weights) as a numpy array
    '''
    w = np.zeros(len(X[0]))
    eta = 1
    n = 20
    errors = []

    for t in range(n):
        total_error = 0
        for i, x in enumerate(X):
            if (np.dot(X[i], w)*Y[i]) <= 0:
                total_error += (np.dot(X[i], w)*Y[i])
                w = w + eta*X[i]*Y[i]
        errors.append(total_error*-1)
        
    plt.plot(errors)
    plt.xlabel('Epoch')
    plt.ylabel('Total Loss')
    
    return w
print(perceptron_sgd_plot(inputs,y))

for d, sample in enumerate(inputs):
    # Plot the negative samples
    if d < 3:
        plt.scatter(sample[0], sample[1], s=120, marker='_', linewidths=2)
    # Plot the positive samples
    else:
        plt.scatter(sample[0], sample[1], s=120, marker='+', linewidths=2)

# Add our test samples

plt.scatter(1,2, s=120, marker='_', linewidths=2, color='red')
plt.scatter(6,3, s=120, marker='+', linewidths=2, color='blue')

# Print the hyperplane calculated by perceptron_sgd()
x2=[w[0],w[1],-w[1],w[0]]
x3=[w[0],w[1],w[1],-w[0]]

x2x3 =np.array([x2,x3])
X,Y,U,V = zip(*x2x3)
ax = plt.gca()
ax.quiver(X,Y,U,V,scale=1, color='blue')


