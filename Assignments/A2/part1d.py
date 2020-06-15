'''
    Logistic Regression Gradient Descent using sklearn
'''
import pandas as pd
import numpy as np
import math
from scipy.stats import pearsonr
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt

def sigmoid(x):
  return 1 / (1 + math.exp(-x))

def graph_result(data, target, model):
    for i in range(len(target)):
        if(target.iloc[i] == 0):
            plt.scatter(data.iloc[i,0], data.iloc[i,1], color='red')
        else:
            plt.scatter(data.iloc[i,0], data.iloc[i,1], color='blue')
    
    y = - (model.intercept_[0] + np.dot(model.coef_[0][0], data.iloc[:,0])) / model.coef_[0][1]
    plt.plot(data.iloc[:,0], y, color='green')
    plt.xlabel(data.columns.values[0])
    plt.ylabel(data.columns.values[1])
    plt.show()

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

    # get optimal pearson features
    key_features = feature_selection(X_train, y_train)

    # train model
    model = LogisticRegression()
    model.fit(X_train.iloc[:, [key_features[0], key_features[1]]], y_train)
    
    # make predictions
    pred = model.predict(X_test.iloc[:, [key_features[0], key_features[1]]])
    accuracy = accuracy_score(pred, y_test)
    weights = model.coef_
    print(str(weights) + " " + str(model.intercept_))
    print(accuracy)

    graph_result(X_test.iloc[:, [key_features[0], key_features[1]]], y_test, model)

    