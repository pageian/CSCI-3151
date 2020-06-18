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
    return np.dot(-target, np.log(prediction)) - np.dot(1 - target, np.log(1 - prediction))

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

    y = - (bias + np.dot(weight[0], data.iloc[:,0])) / weight[1]
    plt.plot(data.iloc[:,0], y, color='green')
    plt.xlabel(data.columns.values[0])
    plt.ylabel(data.columns.values[1])
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
    theta = np.zeros((data.shape[1], 1))

    for i in range(max_iter):
        z = np.dot(data.iloc[:,0], current_weight[0]) + np.dot(data.iloc[:,1], current_weight[1]) + current_bias
        pred = sigmoid(z)
        diff = pred - target
        
        cost = cross_entropy(pred, target)
        current_weight[0] = current_weight[0] - ((learning_rate/N) * np.sum(data.iloc[:,0] * diff))
        current_weight[1] = current_weight[1] - ((learning_rate/N) * np.sum(data.iloc[:,1] * diff))
        current_bias = current_bias - ((learning_rate/N) * np.sum(diff))

        train_acc = 1 - (np.sum(abs(diff)) / N)
        val_acc = validate(data_test, target_test, current_weight, current_bias)

        if i % 1000 == 0:
           print("[Iteration " + str(i) + "] : " + str(cost) + ", " + str(train_acc) + ", " + str(val_acc))
 
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
    shuffled_data = data_file.sample(frac=1)
    X = shuffled_data.iloc[1:,:8] # Features
    y = shuffled_data.iloc[1:,8] # Target variable

    # spltting train and test data
    split_index = int(math.floor(len(y) * 0.8))
    X_train, X_test = X.iloc[:split_index], X.iloc[split_index:]
    y_train, y_test = y.iloc[:split_index], y.iloc[split_index:]
    
    # get optimal pearson features
    key_features = feature_selection(X_train, y_train)

    # train model
    weight, bias = LRGradDesc(X_train.iloc[:, [key_features[0], key_features[1]]], y_train, X_test.iloc[:, [key_features[0], key_features[1]]], y_test, 0, -10, 0.001, 10000)
    print("Final weights: " + str(bias) + ", " + str(weight[0]) + ", " + str(weight[1]))
    graph_result(X_test.iloc[:, [key_features[0], key_features[1]]], y_test, weight, bias)