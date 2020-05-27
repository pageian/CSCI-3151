import sys
import pandas as pd
import numpy as np

def LRGradDesc(data, target, init_x1=0, init_x2=0, init_b=0,
    learning_rate_w=0.0001, learning_rate_b=0.0001, max_iter=100):
    
    #print(data)
    #print(target)
    N = len(data.iloc[:, 0])
    current_x1 = init_x1
    current_x2 = init_x2
    current_b = init_b
    for i in range(max_iter):
        prediction = np.dot(data.iloc[:,0], current_x1) + np.dot(data.iloc[:,1], current_x2) + current_b
        #print(target)
        cost = np.sum((prediction - target) ** 2) / (2. * N)

        x1_grad = np.dot((learning_rate_w/N), np.sum(np.dot(data.iloc[:,0], (prediction - target))))
        x2_grad = np.dot((learning_rate_w/N), np.sum(np.dot(data.iloc[:,1], (prediction - target))))
        b_grad = np.dot((learning_rate_b/N), np.sum(prediction - target))
        print(x1_grad)
    
        current_x1 = current_x1 - x1_grad
        current_x2 = current_x2 - x2_grad
        current_b = current_b - b_grad

        #print("[Iteration " + str(i) + "] x1: " + str(current_x1) + " x2: " + str(current_x2) + " b: " + str(current_b) + " cost: " + str(cost))

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

    