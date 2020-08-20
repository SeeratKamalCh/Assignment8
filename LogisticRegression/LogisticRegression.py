
from scipy.io import loadmat
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import math
from math import sin
from math import cos
import numpy as np
from scipy import linalg

"""
This program is a logistic regression classifier that trains the model to learn
the weights and then classifies the test data based on the learned weights
"""

# define global variables
eta = 0.01
tol = 0.001
samples = 0
features = 0

# This function reads the data from mat file
def read_Data():
    # Load data
    data = loadmat('Data.mat')
    # Read Test and Train data arrays
    XTrain = data['XTrain']
    yTrain = data['yTrain']
    XTest = data['XTest']
    yTest = data['yTest']
    # set total training samples and features in global variables
    global samples
    global features
    samples = XTrain.shape[0]
    features = XTrain.shape[1]
    # Append a column of ones in the Train and Test feature arrays for weight w0
    XTrain = np.insert(XTrain, 0, 1, axis=1)
    XTest = np.insert(XTest, 0, 1, axis=1)
    # normalize the values of arrays in a range to avoid overflow error for np.exp
    XTrain = exp_normalize(XTrain)
    XTest = exp_normalize(XTest)
    return XTrain, yTrain, XTest, yTest

# This function normalizes the large float data
def exp_normalize(x):
    b = x.max()
    y = np.exp(x - b)
    return y / y.sum()

# This function trains/ calculates the new weights until convergence
def calculate_gradient_ascent(XTrain, yTrain):
    global features
    global tol
    global eta
    # Initialize the wights array
    wHat = np.zeros((features + 1, 1))
    converged = False
    old_val = 0
    new_val = 0
    i = 0
    # Run until convergence
    while (converged == False and i < 1000):
        print(i)
        i += 1
        # get new gradient values
        gradient = calculate_gradient(XTrain, yTrain, wHat)
        # update the weights
        wHat = update_weights(wHat, gradient, eta)
        # calculate new objective value for new weights
        new_val = objective_function(yTrain, XTrain, wHat)
        # check if the function has converged or not
        converged = check_convergence(old_val, new_val, tol)
    return wHat

# This function calculates the gradient for Train data for changed weights
def calculate_gradient(XTrain, yTrain, wHat):
    # get linear function value
    linear_val = calculate_linear(XTrain, wHat)
    # get sigmoid function value
    sigmoid_val = sigmoid(linear_val, 0)
    # put in formula of gradient = sigma(yi - theta(z) * xi) 
    diff = yTrain - sigmoid_val
    gradient = np.matmul(diff.transpose(), XTrain)
    return gradient.transpose()

# This function calculates the linear function val = w0 + w1*x1 + .... = w0 + sigma(wi*xi)
def calculate_linear(XTrain, wHat):
    val = np.matmul(XTrain, wHat)
    return val

# This function calculates the sigmoid value on the linear function calculated
def sigmoid(x, category):
    val = 0
    # for category 0 the function returns this value
    if category == 0:       
        val = 1 / (1 + np.exp(-x))
    else:
        # for category 1 the function returns this value
        val = np.exp(-x) / (1 + np.exp(-x))
    return val

# This function calculates the objective value for logistic regression model
def objective_function(yTrain, XTrain, wHat):
    # get linear function value of train data for given weights
    linear_val = calculate_linear(XTrain, wHat)
    # get sigmoid value for the linear function value calculated for category 0
    sigmoid_val = sigmoid(linear_val, 0)
    #apply formula : sigma(-yTrain * log(theta(z)) - (1 - yTrain) * log(1 - theta(z)))
    element_1 = np.matmul(-yTrain, np.log10(sigmoid_val).transpose())
    element_2 = np.matmul(1 - yTrain, np.log10(1 - sigmoid_val).transpose())
    obj_val = np.sum(element_1 - element_2)
    return obj_val

# This function is to update the weights according to calculated gradient values
def update_weights(wHat, gradient, eta):
    # weight = weight + delta_weight
    wHat = wHat - eta * gradient
    return wHat

# This function is to check if the function has converged or not
def check_convergence(old_val, new_val, tolerance_threshold):
    # get the difference of new and old value of the objective function
    convergence_diff = np.abs(old_val - new_val)
    # if it is less than the tolerance threshold for convergence then return true else false
    if convergence_diff < tolerance_threshold:
        return True
    else:
        return False
    return

# This function is to predict the labels on the test dataset
def predict_labels(XTest, wHat):
    # get number of test dataset samples
    samples = XTest.shape[0]
    # initialize the predictions list
    predictions = []
    for i in range(XTest.shape[0]):
        # for every test dataset sample calculate probabilities for both the categories P(Y=0|X) and P(Y=1|X)
        linear_val = calculate_linear(XTest[i, :], wHat)
        sigmoid_val_0 = sigmoid(linear_val, 0)
        sigmoid_val_1 = sigmoid(linear_val, 1)
        # whichever category value is bigger append label of that category
        if sigmoid_val_0 > sigmoid_val_1:
            predictions.append(0)
        else:
            predictions.append(1)
    predictions = np.array(predictions)
    return predictions.reshape(samples, 1)

# This function is to count the correctly labelled test datasets
def correct_classified(predictions, yTest):
    diff = np.abs(predictions - yTest)
    # select indices where the difference is 0
    correct_classified = diff[diff == 0]
    # return number of correctly labelled samples alongwith total number of test samples
    return correct_classified.shape[0], yTest.shape[0]

# This function is the main driver function
def main():
    # Read dataset for test and train samples
    XTrain, yTrain, XTest, yTest = read_Data()
    # Train the model to learn the weights
    wHat = calculate_gradient_ascent(XTrain, yTrain)
    # make predictions of the test dataset using weights calculated in the training
    predictions = predict_labels(XTest, wHat)
    # get correctly labelled and total test samples
    correct, total = correct_classified(predictions, yTest)
    print("Total test samples: ", total)
    print("correct: ", correct)
    return

main()
    