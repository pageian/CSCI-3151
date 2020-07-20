import numpy as np
import pandas as pd 
from sklearn import datasets

import matplotlib.pyplot as plt
from sklearn.metrics import silhouette_samples, silhouette_score
import matplotlib.cm as cm
from sklearn.metrics.cluster import adjusted_rand_score

from sklearn.cluster import DBSCAN

(X, y) = datasets.load_wine(return_X_y=True)

eps_min = 3
eps_max = 101
samples_min = 2
samples_max = 40
clrs = ['red', 'green', 'blue', 'yellow', 'purple']

eps_silhouette_avgs = []
eps_adjusted_rand = []
samples_silhouette_avgs = []
samples_adjusted_rand = []

for k in range(eps_min, eps_max):
    
    db = DBSCAN(eps=k, min_samples=2)
    labels = db.fit_predict(X)
    silhouette_avg = silhouette_score(X, labels)
    eps_silhouette_avgs.append(silhouette_avg)

    eps_adjusted_rand.append(adjusted_rand_score(y, labels))

for k in range(samples_min, samples_max):
    
    db = DBSCAN(eps=60, min_samples=k)
    labels = db.fit_predict(X)
    silhouette_avg = silhouette_score(X, labels)
    samples_silhouette_avgs.append(silhouette_avg)

    samples_adjusted_rand.append(adjusted_rand_score(y, labels))

plt.clf()
plt.plot(range(samples_min, samples_max), samples_adjusted_rand, c='blue', label='adjust rand index samples')
plt.plot(range(eps_min, eps_max), eps_adjusted_rand, c='red', label='adjust rand index eps')
plt.legend()
plt.xlabel('n clusters')
plt.ylabel('Score Metric')
plt.show()

plt.clf()
plt.plot(range(samples_min, samples_max), samples_silhouette_avgs, c='blue', label='silhouette score samples')
plt.plot(range(eps_min, eps_max), eps_silhouette_avgs, c='red', label='silhouette score eps')
plt.legend()
plt.xlabel('n clusters')
plt.ylabel('Score Metric')
plt.show()

best_sample_silhouette_score = max(samples_silhouette_avgs)
best_samples_adjust_rand_score = max(samples_adjusted_rand)
best_eps_silhouette_score = max(eps_silhouette_avgs)
best_eps_adjust_rand_score = max(eps_adjusted_rand)

best_sample_silhouette = samples_silhouette_avgs.index(best_sample_silhouette_score) + samples_min
best_samples_adjust_rand = samples_adjusted_rand.index(best_samples_adjust_rand_score) + samples_min
best_eps_silhouette = eps_silhouette_avgs.index(best_eps_silhouette_score) + eps_min
best_eps_adjust_rand = eps_adjusted_rand.index(best_eps_adjust_rand_score) + eps_min


print("\nMin Samples Results ###")
print("Silhouette Score: " + str(best_sample_silhouette_score) + " Suggested Param: " + str(best_sample_silhouette))
print("Adjusted Rand Score: " + str(best_samples_adjust_rand_score) + " Suggested Param: " + str(best_samples_adjust_rand))

print("\nEPS Results ###")
print("Silhouette Score: " + str(best_eps_silhouette_score) + " Suggested Param: " + str(best_eps_silhouette))
print("Adjusted Rand Score: " + str(best_eps_adjust_rand_score) + " Suggested Param: " + str(best_eps_adjust_rand))