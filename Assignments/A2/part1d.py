'''
    Logistic Regression Gradient Descent using sklearn
'''
import pandas as pd
import numpy as np
import math
from scipy.stats import pearsonr
import matplotlib.pyplot as plt

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
    