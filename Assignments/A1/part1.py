import sys
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

def TestModel(x1, x2, b):

    print("\n### Testing Model ###\n")

    data_file = pd.read_csv("data-sets/housing-prices-dataset/test.csv", usecols=[43,44])
    data = data_file.iloc[:,:2]

    for i in range(100):
        prediction = data.iloc[i,0] * x1 + data.iloc[i,1] * x2 + b
        print("Sale Price: " + str(prediction))

def LRGradDesc(data, target, init_x1=0, init_x2=0, init_b=0,
    learning_rate_w=0.001, learning_rate_b=0.1, max_iter=10000):

    sns.set(style="whitegrid",rc={'figure.figsize':(11.7,8.27)})
    X = data.iloc[:,0] + data.iloc[:, 1]
    print(X)
    plotting_data = pd.DataFrame({
        'X': X,
        'Y': target})

    print(plotting_data)
    sns.scatterplot(x="X", y="Y", data=plotting_data)

    print("\n### Training Model ###\n")

    init_cost = 0
    N = len(data.iloc[:, 0])
    current_x1 = init_x1
    current_x2 = init_x2
    current_b = init_b
    for i in range(max_iter):
        prediction = np.dot(data.iloc[:,0], current_x1) + np.dot(data.iloc[:,1], current_x2) + current_b

        diff = np.divide(prediction - target, target)
        cost = np.sum(diff ** 2) / (2. * N)
        if i == 0:
            init_cost = cost

        x1_grad = np.dot((learning_rate_w/N), np.sum(np.dot(data.iloc[:,0], diff)))
        x2_grad = np.dot((learning_rate_w/N), np.sum(np.dot(data.iloc[:,1], diff)))
        b_grad = np.dot((learning_rate_b/N), np.sum(diff))
    
        current_x1 = current_x1 - x1_grad
        current_x2 = current_x2 - x2_grad
        current_b = current_b - b_grad

        if i % 1000 == 0:
            print("[Iteration " + str(i) + "] x1: " + str(current_x1) + " x2: " + str(current_x2) + " b: " + str(current_b) + " cost: " + str(cost))

        if i == max_iter - 1:
            print("\n### Final Results ###\n")
            print("Cost Improvement: %.2f%%" % ((1 - (cost / init_cost)) * 100))
            print("[Iteration " + str(i) + "] x1: " + str(current_x1) + " x2: " + str(current_x2) + " b: " + str(current_b) + " cost: " + str(cost))
    
    plt.plot(range(4500), np.dot(range(4500), current_x1) + np.dot(range(4500), current_x2) + current_b, color='red')
    plt.show()
    return current_x1, current_x2, current_b

if __name__ == "__main__":
    data_file = pd.read_csv("data-sets/housing-prices-dataset/train.csv", usecols=[43,44,80])
    x1, x2, b = LRGradDesc(data_file.iloc[:,:2], data_file.iloc[:,2])
    TestModel(x1, x2, b)

    