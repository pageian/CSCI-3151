'''
    Logistic Regression Gradient Descent w/ ROC curve
'''
import pandas as pd
import numpy as np
import math
from scipy.stats import pearsonr
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve 
from sklearn.linear_model import LogisticRegression

def validate(data, target, weight, bias):
    z = np.dot(data.iloc[:,0], weight[0]) + np.dot(data.iloc[:,1], weight[1]) + bias
    pred = sigmoid(z)
    accuracy = 1 - (np.sum(abs(pred - target)) / len(data.iloc[:,0]))
    return accuracy

def cross_entropy(prediction, target):
    return np.dot(-target, np.log(prediction)) - np.dot(1 - target, np.log(1 - prediction))

def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def predict(X, weight, bias):
    z = np.dot(X.iloc[:,0], weight[0]) + np.dot(X.iloc[:,1], weight[1]) + np.dot(X.iloc[:,2], weight[2]) + bias
    return sigmoid(z)

def probability(pred):
    reg_odds = np.exp(pred)
    return reg_odds / (1 + reg_odds)
    
'''
    log regr gradient trainer
    returns final input weights
'''
def LRGradDesc(data, target, data_test, target_test, weight_init, bias_init, learning_rate, max_iter):
    
    N = len(data.iloc[:,0])
    current_weight = [weight_init, weight_init, weight_init]
    current_bias = bias_init
    theta = np.zeros((data.shape[1], 1))

    for i in range(max_iter):
        z = np.dot(data.iloc[:,0], current_weight[0]) + np.dot(data.iloc[:,1], current_weight[1]) + np.dot(data.iloc[:,2], current_weight[2]) + current_bias
        pred = sigmoid(z)
        diff = pred - target
        
        cost = cross_entropy(pred, target)
        current_weight[0] = current_weight[0] - ((learning_rate/N) * np.sum(data.iloc[:,0] * diff))
        current_weight[1] = current_weight[1] - ((learning_rate/N) * np.sum(data.iloc[:,1] * diff))
        current_weight[2] = current_weight[2] - ((learning_rate/N) * np.sum(data.iloc[:,2] * diff))
        current_bias = current_bias - ((learning_rate/N) * np.sum(diff))

        train_acc = 1 - (np.sum(abs(diff)) / N)
        val_acc = validate(data_test, target_test, current_weight, current_bias)
 
    return current_weight, current_bias

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
    key_features = [1, 5, 6]

    # train models
    weight, bias = LRGradDesc(X_train.iloc[:, [key_features[0], key_features[1], key_features[2]]],
        y_train, X_test.iloc[:, [key_features[0], key_features[1], key_features[2]]], y_test, 0, -10, 0.001, 2000)

    model = LogisticRegression()
    model.fit(X_train.iloc[:, [key_features[0], key_features[1], key_features[2]]], y_train)
   
    ns_probs = [0 for _ in range(len(y_test))]
    probs = probability(predict(X_test.iloc[:, [key_features[0], key_features[1], key_features[2]]], weight, bias))
    probs_canned = model.predict_proba(X_test.iloc[:, [key_features[0], key_features[1], key_features[2]]])
    probs_canned = probs_canned[:, 1]

    # calculate roc curves
    ns_fpr, ns_tpr, _ = roc_curve(y_test, ns_probs)
    lr_fpr, lr_tpr, _ = roc_curve(y_test, probs)
    # plot the roc curve for the model
    plt.plot(ns_fpr, ns_tpr, linestyle='--', label='No Skill')
    plt.plot(lr_fpr, lr_tpr, marker='.', label='Logistic')
    # axis labels
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Scratch LR Model')
    # show the legend
    plt.legend()
    # show the plot
    plt.show()

    plt.clf()
    # calculate roc curves
    ns_fpr1, ns_tpr1, _ = roc_curve(y_test, ns_probs)
    lr_fpr1, lr_tpr1, _ = roc_curve(y_test, probs_canned)
    # plot the roc curve for the model
    plt.plot(ns_fpr1, ns_tpr1, linestyle='--', label='No Skill')
    plt.plot(lr_fpr1, lr_tpr1, marker='.', label='Logistic')
    # axis labels
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('SKLearn LR Model')
    # show the legend
    plt.legend()
    # show the plot
    plt.show()