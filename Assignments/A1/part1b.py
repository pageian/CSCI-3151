import pandas as pd
import numpy as np
from sklearn import linear_model
import seaborn as sns
import matplotlib.pyplot as plt

def LRGradDesc(x_train, y_train):
    
    tot_area = x_train.iloc[:,0] + x_train.iloc[:, 1]
    plotting_data = pd.DataFrame({
        'Total Square Feet': tot_area,
        'Sale Price': y_train})
    sns.scatterplot(x="Total Square Feet", y="Sale Price", data=plotting_data)
    
    regr = linear_model.LinearRegression()
    regr.fit(x_train, y_train)

    N = len(x_train.iloc[:, 0])
    prediction = regr.predict(x_train)
    diff = np.divide(prediction - y_train, y_train)
    cost = np.sum(diff ** 2) / (2. * N)
    # The coefficients
    print("Final Cost: %.2f" % cost)
    print('x1: %.2f' % regr.coef_[0])
    print('x2: %.2f' % regr.coef_[1])
    print('b: %.2f' % regr.intercept_)

    plt.plot(range(4500), np.dot(range(4500), regr.coef_[0]) + np.dot(range(4500), regr.coef_[1]) + regr.intercept_, color='red')
    plt.show()

if __name__ == "__main__":
    data_train = pd.read_csv("data-sets/housing-prices-dataset/train.csv", usecols=[43,44,80])
    LRGradDesc(data_train.iloc[:,:2], data_train.iloc[:,2])