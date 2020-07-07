'''
    Gaussian Mixture Model for alien races
'''
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.mixture import GaussianMixture as GMM
from sklearn.model_selection import train_test_split


if __name__ == "__main__":

    # get file data
    data_file = pd.read_csv("data-sets/aliens.csv")
    shuffled_data = data_file.sample(frac=1)
    X = shuffled_data # Features

    models = []
    for n in range(2,8):
        gmm = GMM(n_components=n).fit(X)
        labels = gmm.predict(X)
        models.append(gmm)

    BIC_scores = [m.bic(X) for m in models]
    plt.plot(range(2,8), BIC_scores, label='BIC')
    plt.legend(loc='best')
    plt.xlabel('n_components')
    plt.ylabel('BIC Score')
    plt.show()
    print(BIC_scores.index(min(BIC_scores)) + 2)
    optimal_n = BIC_scores.index(min(BIC_scores)) + 2

    gmm = GMM(n_components=optimal_n).fit(X)
    labels = gmm.predict(X)
    plt.scatter(X.iloc[:, 0], X.iloc[:, 1], c=labels, s=40, cmap='viridis')
    plt.xlabel('Alien Height In Meters')
    plt.ylabel('Alien Weight In Kilograms')
    title = '# Components: ' + str(optimal_n)
    plt.title(title)
    plt.show()
    means = gmm.means_
    weights = gmm.weights_
    cov = gmm.covariances_
    stds = [ np.sqrt(  np.trace(cov[i])/optimal_n) for i in range(0,optimal_n) ]

    # print(means)
    # print(cov)
    for i in range(0, optimal_n):
        print('**********')
        print('Alien Race: ' + str(i))
        print('Mean Height: ' + str(means[i][0]))
        print('Mean Weight: ' + str(means[i][1]))
        print('Stan Dev: ' + str(stds[i]))