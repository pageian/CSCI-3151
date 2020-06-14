import pandas as pd
import numpy as np
import math
from scipy.stats import pearsonr

def LRGradDesc(data, target, weight_init, bias_init, learning_rate, max_iter):
    print("CHII")

def feature_selection(data, target):
    correlations = []
    for i in range(0, len(data.iloc[1,:])):
        correlations.append(pearsonr(data.iloc[:,i], target)[0])
    
    sorted_correl = sorted(correlations, reverse=True)
    return correlations.index(sorted_correl[0]), correlations.index(sorted_correl[1])

if __name__ == "__main__":
    data_file = pd.read_csv("data-sets/diabetes.csv")
    X = data_file.iloc[1:,:8] # Features
    y = data_file.iloc[1:,8] # Target variable
    split_index = int(math.floor(len(y) * 0.8))

    X_train, X_test = X.iloc[:split_index], X.iloc[split_index:]
    y_train, y_test = y.iloc[:split_index], y.iloc[split_index:]

    print(feature_selection(X_train, y_train))
    LRGradDesc(X_train, y_train, 0, 0, 0.1, 1000)