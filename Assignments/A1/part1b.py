import pandas as pd
import numpy as np
from sklearn import datasets, linear_model
from sklearn.metrics import mean_squared_error, r2_score

def LRGradDesc(x_train, y_train, x_test, y_test):
    regr = linear_model.LinearRegression()
    regr.fit(x_train, y_train)

    N = len(x_train.iloc[:, 0])
    prediction = regr.predict(x_test)
    diff = np.divide(prediction - y_train, y_train)
    cost = np.sum(diff ** 2) / (2. * N)
    # The coefficients
    print("Cost: %.2f" % cost)
    print('Coefficients: \n', regr.coef_)

if __name__ == "__main__":
    data_train = pd.read_csv("data-sets/housing-prices-dataset/train.csv", usecols=[43,44,80])
    LRGradDesc(data_train.iloc[:,:2], data_train.iloc[:,2], data_train.iloc[:,:2], data_train.iloc[:,2])