
from scipy.io import loadmat
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import math
from math import sin
from math import cos
import numpy as np
from scipy import linalg


"""
This program is a naive bayes classifer that trains the model on train data
and then classifies the documents in the test set
"""

# define global variables
ALPHA = 2
BETA = 1
# count of total words in economist and onion categories
count_e = 0
count_o = 0
# count of documents in economist and onion categories
count_e_docs = 0
count_o_docs = 0

# This function reads the data from mat file
def read_Data():
    data = loadmat('DataA.mat')
    vocab = data['Vocabulary']
    global count_e
    global count_o
    global count_e_docs
    global count_o_docs
    vocabulary = []
    for word in vocab:
        vocabulary.append(word[0][0])
    words_economist = np.zeros((len(vocabulary), ))
    words_onions = np.zeros((len(vocabulary), ))
    train_set = data['XTrain']
    train_label = data['yTrain']
    test_set = data['XTest']
    test_label = data['yTest']
    # define counts of the vocabulary words in every category 
    for j in range(train_set.shape[0]): 
        # document label
        print(j)
        doc_label = train_label[j]
        if doc_label == 1:
            count_e_docs += 1
        else:
            count_o_docs += 1
        for i in range(len(vocabulary)):
            # i = index of word number
            # get single word from train_set index
            word = train_set[j, i]
            if doc_label == 1 and word == 1:
                words_economist[i] += 1
                count_e += 1
            elif doc_label == 2 and word == 1:
                words_onions[i] += 1
                count_o += 1
    return words_economist, words_onions, test_set, test_label

# This function trains the naive bayes classifier on both the categories
def train(words_economist, words_onions):
    # get probabilities for economist
    global count_e
    global count_o
    prob_economist = MAP_estimate(words_economist, count_e)
    # get probabilities for onions
    prob_onions = MAP_estimate(words_onions, count_o)
    return prob_economist, prob_onions

# This function calculates the MAP estimation of words in a category
def MAP_estimate(words_list, count):
    prob_count = []
    for i in range(len(words_list)):
        prob_count.append((words_list[i] + ALPHA - 1) / (count + ALPHA + BETA - 2))
    return prob_count

# This function classifies the test data from the trained model
def classify(test_set, prob_economist, prob_onions):
    predicted = []
    global count_e_docs
    global count_o_docs
    # calculate prior probabilities for economist and onions
    prior_prob_e = count_e_docs / (count_e_docs + count_o_docs)
    prior_prob_o = count_o_docs / (count_e_docs + count_o_docs)
    for i in range(test_set.shape[0]):  
        # check prob for economist p (economist | document)
        print(i)
        cal_e_prob = math.log(prior_prob_e)
        for j in range(test_set.shape[1]):
            word = test_set[i, j]
            if word == 1:
                cal_e_prob += math.log(prob_economist[j])
        cal_o_prob = math.log(prior_prob_o)
        # check prob for onions p (onions | document)
        for j in range(test_set.shape[1]):
            word = test_set[i, j]
            if word == 1:
                cal_o_prob += math.log(prob_onions[j])
        # get the maximum probability category
        if cal_e_prob > cal_o_prob:
            predicted.append(1)
        else:
            predicted.append(2)
    return predicted

# This function calculates the error percentage from the predicted and original labels
def calculate_error(predicted_label, test_label):
    total = len(predicted_label)
    correct = 0
    for i in range(len(predicted_label)):
        predicted = predicted_label[i]
        actual = test_label[i]
        if predicted == actual:
            correct += 1
    return (correct / total) * 100

# This function is the main driver function
def main():
    words_economist, words_onions, test_set, test_label = read_Data()
    prob_economist, prob_onions = train(words_economist, words_onions)
    predicted_label = classify(test_set, prob_economist, prob_onions)
    error = calculate_error(predicted_label, test_label)
    print(error)
    return

main()
    