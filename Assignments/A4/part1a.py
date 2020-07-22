import numpy as np
import pandas as pd 
from sklearn import datasets
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_samples, silhouette_score
import matplotlib.cm as cm
from sklearn.metrics.cluster import adjusted_rand_score

(X, y) = datasets.load_wine(return_X_y=True)

k_min = 2
k_max = 6
clrs = ['red', 'green', 'blue', 'yellow']
clr_i = 0
inits = ['kmeans++', 'random']
silhouette_avgs = []
adjusted_rand = []
for i in inits:
    for k in range(k_min, k_max):
        
        km = KMeans(
            n_clusters=k, init='random'
        )

        y_km = km.fit_predict(X)

        # computing sillouette avg score
        silhouette_avg = silhouette_score(X, y_km)
        silhouette_avgs.append(silhouette_avg)
        adjusted_rand.append(adjusted_rand_score(y, y_km))

    sil_title = i + ": silhouette score"
    adj_title = i + ": adjust rand index"
    sil_clr = clrs[clr_i]
    adj_clr = clrs[clr_i + 1]
    plt.plot(range(k_min, k_max), silhouette_avgs, c=sil_clr, label=sil_title)
    plt.plot(range(k_min, k_max), adjusted_rand, c=adj_clr, label=adj_title)
    silhouette_avgs = []
    adjusted_rand = []
    clr_i += 2


plt.legend()
plt.xlabel('K')
plt.ylabel('Score Metric')
plt.suptitle("Scoring Metrics for KMeans Over K")
plt.show()