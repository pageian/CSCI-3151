'''
    Logistic Regression Gradient Descent
'''
import pandas as pd
import numpy as np
import math
from scipy.stats import pearsonr
import matplotlib.pyplot as plt

def validate(data, target, weight, bias):
    z = np.dot(data.iloc[:,0], weight[0]) + np.dot(data.iloc[:,1], weight[1]) + bias
    pred = sigmoid(z)

    accuracy = 1 - (np.sum(abs(pred - target)) / len(data.iloc[:,0]))
    return accuracy

def cross_entropy(prediction, target):
    return np.subtract(np.dot(-target, np.log(prediction)), np.dot(np.subtract(1, target), np.log(np.subtract(1, prediction))))

def log_cross_entropy(z, target):
    return np.dot(target, np.log(1 + math.e ** (-z))) + np.dot(1 - target, np.log(1 + math.e ** (-z)))

def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def graph(data, target):
    for i in range(len(target)):
        if(target.iloc[i] == 0):
            plt.scatter(data.iloc[i,0], data.iloc[i,1], color='red')
        else:
            plt.scatter(data.iloc[i,0], data.iloc[i,1], color='blue')
    plt.show()

def graph_result(data, target, weight, bias):
    for i in range(len(target)):
        if(target.iloc[i] == 0):
            plt.scatter(data.iloc[i,0], data.iloc[i,1], color='red')
        else:
            plt.scatter(data.iloc[i,0], data.iloc[i,1], color='blue')
    #plt.scatter(range(len(data.iloc[:,0])), target, color='red')
    z = np.dot(data.iloc[:,0], weight[0]) + np.dot(data.iloc[:,1], weight[1]) + bias
    plt.plot(range(max(data.iloc[:,0])), z, color='green')
    plt.show()
'''
    log regr gradient trainer
    returns final input weights
'''
def LRGradDesc(data, target, data_test, target_test, weight_init, bias_init, learning_rate, max_iter):
    
    print("### Training Model ###")
    N = len(data.iloc[:,0])
    current_weight = [weight_init, weight_init]
    current_bias = bias_init

    for i in range(max_iter):
        z = np.dot(data.iloc[:,0], current_weight[0]) + np.dot(data.iloc[:,1], current_weight[1]) + current_bias
        pred = sigmoid(z)
        diff = pred - target
        #TODO: use correct cost function
        cost = cross_entropy(pred, target)
        log_cost = log_cross_entropy(z, target)
        current_weight[0] = current_weight[0] - ((learning_rate/N) * np.sum(data.iloc[:,0] * diff))
        current_weight[1] = current_weight[1] - ((learning_rate/N) * np.sum(data.iloc[:,1] * diff))
        current_bias = current_bias - ((learning_rate/N) * np.sum(diff))

        train_accuracy = 1 - (np.sum(abs(diff)) / N)
        test_accuracy = validate(data_test, target_test, current_weight, current_bias)

        if i % 200 == 0:
           print("[Iteration " + str(i) + "] : " + str(cost) + ", " + str(log_cost) + ", " + str(test_accuracy))
 
    return current_weight, current_bias

'''
    pearson coeff feature selection
    returns top 2 feature indices
'''
def feature_selection(data, target):
    correlations = []
    for i in range(0, len(data.iloc[1,:])):
        correlations.append(pearsonr(data.iloc[:,i], target)[0])
    
    sorted_correl = sorted(correlations, reverse=True)
    return correlations.index(sorted_correl[0]), correlations.index(sorted_correl[1])

if __name__ == "__main__":

    # get file data
    data_file = pd.read_csv("data-sets/diabetes.csv")
    X = data_file.iloc[1:,:8] # Features
    y = data_file.iloc[1:,8] # Target variable

    # TODO: must be a randomized selection
    # spltting train and test data
    split_index = int(math.floor(len(y) * 0.8))
    X_train, X_test = X.iloc[:split_index], X.iloc[split_index:]
    y_train, y_test = y.iloc[:split_index], y.iloc[split_index:]
    graph(X_train, y_train)
    # get optimal pearson features
    key_features = feature_selection(X_train, y_train)

    # train model
    weight, bias = LRGradDesc(X_train.iloc[:, [key_features[0], key_features[1]]], y_train, X_test.iloc[:, [key_features[0], key_features[1]]], y_test, 0, -10, 0.001, 2000)
    graph_result(X_test, y_test, weight, bias)