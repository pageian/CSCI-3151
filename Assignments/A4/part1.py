import numpy as np
import pandas as pd 
from sklearn import datasets

import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

print('test')

(X, y) = datasets.load_wine(return_X_y=True)

clrs = ['red', 'green', 'blue', 'yellow', 'purple']
for k in range(2, 6):
    km = KMeans(
        n_clusters=k, init='random',
        n_init=10, max_iter=300, 
        tol=1e-04, random_state=0
    )

    y_km = km.fit_predict(X)

    for i in range(0, k):
        plt.scatter(
            X[y_km == i, 0], X[y_km == i, 1],
            s=50, c=clrs[i],
            marker='s', edgecolor='black',
            label='cluster 1'
        )

    plt.legend(scatterpoints=1)
    plt.grid()

    plt.show()