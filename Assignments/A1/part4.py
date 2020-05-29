import numpy as np
import pandas as pd

# Remove these after testing 
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import itertools

def plot_confusion_matrix(cm, classes,
                        normalize=False,
                        title='Confusion matrix',
                        cmap=plt.cm.Blues):

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                    horizontalalignment="center",
                    color="white" if cm[i, j] > thresh else "black")

    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.tight_layout()

def KNN(X_train, y_train, X_test, y_test, k=3, p=0):
    print("KNN")

    #create object of the lassifier
    neigh = KNeighborsClassifier(n_neighbors=k)

    #Train the algorithm
    neigh.fit(X_train, y_train.values.ravel())

    # predict the response
    pred = neigh.predict(X_test)
    
    # evaluate accuracy
    print ("KNeighbors accuracy score : ",accuracy_score(y_test, pred))

    cnf_matrix = confusion_matrix(y_test, pred)

    plt.figure()
    plot_confusion_matrix(cnf_matrix, classes=["1", "2", "3", "4", "5", "6", "7", "8", "9", "10"], title='Confusion matrix')
    plt.show()
    plot_confusion_matrix(cnf_matrix, classes=["1", "2", "3", "4", "5", "6", "7", "8", "9", "10"], normalize=True, title='Confusion matrix, Normalized')
    plt.show()

if __name__ == '__main__':
    X_train = pd.read_csv("data-sets/reducedMNIST/train.csv")
    y_train = pd.read_csv("data-sets/reducedMNIST/train_labels.csv")
    X_test = pd.read_csv("data-sets/reducedMNIST/test.csv")
    y_test = pd.read_csv("data-sets/reducedMNIST/test_labels.csv")

    KNN(X_train, y_train, X_test, y_test)