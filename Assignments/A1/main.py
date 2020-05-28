import sys
import pandas as pd
import numpy as np

def LRGradDesc(data, target, init_x1=0, init_x2=0, init_b=0,
    learning_rate_w=0.001, learning_rate_b=0.5, max_iter=10000):
    
    init_cost = 0
    #print(data)
    #print(target)
    N = len(data.iloc[:, 0])
    current_x1 = init_x1
    current_x2 = init_x2
    current_b = init_b
    for i in range(max_iter):
        prediction = np.dot(data.iloc[:,0], current_x1) + np.dot(data.iloc[:,1], current_x2) + current_b
        #print(prediction)

        diff = np.divide(prediction - target, target)
        cost = np.sum(diff ** 2) / (2. * N)
        if i == 0:
            init_cost = cost

        x1_grad = np.dot((learning_rate_w/N), np.sum(np.dot(data.iloc[:,0], diff)))
        x2_grad = np.dot((learning_rate_w/N), np.sum(np.dot(data.iloc[:,1], diff)))
        b_grad = np.dot((learning_rate_b/N), np.sum(diff))
        #print(x1_grad)
    
        current_x1 = current_x1 - x1_grad
        current_x2 = current_x2 - x2_grad
        current_b = current_b - b_grad

        if i % 1000 == 0:
            print("[Iteration " + str(i) + "] x1: " + str(current_x1) + " x2: " + str(current_x2) + " b: " + str(current_b) + " cost: " + str(cost))

        if i == max_iter - 1:
            print("Cost Improvement: %.2f%%" % ((1 - (cost / init_cost)) * 100))

    print("### Final Results ###")
    print("[Iteration " + str(i) + "] x1: " + str(current_x1) + " x2: " + str(current_x2) + " b: " + str(current_b) + " cost: " + str(cost))
    return current_x1, current_x2, current_b

if __name__ == "__main__":
   
    data_file = None

    if(len(sys.argv) > 1 and sys.argv[1] == "--test"):
        print('test')
        data_file = pd.read_csv("housing-prices-dataset/test.csv", header=None)
        print(data_file)
    else:
        print("train")
        data_file = pd.read_csv("housing-prices-dataset/train.csv", usecols=[43,44,80])
        LRGradDesc(data_file.iloc[:,:2], data_file.iloc[:,2])

    