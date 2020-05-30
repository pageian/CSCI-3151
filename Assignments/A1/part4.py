import numpy as np
import pandas as pd

# Remove these after testing 
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import itertools
import operator

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

def KNN_canned(X_train, y_train, X_test, y_test, k=3, p=0):
    print("KNN Canned")

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

def euclidean_distance(vector1, vector2):
    return np.sqrt(np.sum(np.power(vector1-vector2, 2)))
def absolute_distance(vector1, vector2):
    return np.sum(np.absolute(vector1-vector2))

def get_neighbours(X_train, X_test_instance, k=3, p=0):
    distances = []
    neighbors = []
    for i in range(0, X_train.shape[0]):
        #print("(" + str(i) + "/" + str(X_train.shape[0]))
        dist = absolute_distance(X_train[i], X_test_instance)
        distances.append((i, dist))
    distances.sort(key=operator.itemgetter(1))
    for x in range(k):
        #print distances[x]
        neighbors.append(distances[x][0])
    return neighbors

def predictkNNClass(output, y_train):
    classVotes = {}
    for i in range(len(output)):
#         print output[i], y_train[output[i]]
        if y_train.values[output[i], 0] in classVotes:
            classVotes[y_train.values[output[i], 0]] += 1
        else:
            classVotes[y_train.values[output[i], 0]] = 1
    sortedVotes = sorted(classVotes.items(), key=operator.itemgetter(1), reverse=True)
    #print sortedVotes
    return sortedVotes[0][0]

def kNN_test(X_train, X_test, Y_train, Y_test, k):
    output_classes = []
    print("Getting Neighbours")
    for i in range(0, (X_test.shape[0])):
        if(i % 10 == 0):
            print("%.2f%%" % ((i / X_test.shape[0]) * 100))

        output = get_neighbours(X_train.values, X_test.iloc[i,:].values, k)
        predictedClass = predictkNNClass(output, Y_train)
        output_classes.append(predictedClass)
    return output_classes

def prediction_accuracy(predicted_labels, original_labels):
    count = 0
    for i in range(len(predicted_labels)):
        if predicted_labels[i] == original_labels.values[i, 0]:
            count += 1
    print( count,  " ", len(predicted_labels))
    return float(count)/len(predicted_labels)

if __name__ == '__main__':
    X_train = pd.read_csv("data-sets/reducedMNIST/train.csv")
    y_train = pd.read_csv("data-sets/reducedMNIST/train_labels.csv")
    X_test = pd.read_csv("data-sets/reducedMNIST/test.csv")
    y_test = pd.read_csv("data-sets/reducedMNIST/test_labels.csv")

    # X_train = X_train.iloc[:100,:]
    # y_train = y_train.iloc[:100,:]
    # X_test = X_test.iloc[:100,:]
    # y_test = y_test.iloc[:100,:]

    predicted_classes = {}
    final_accuracies = {}

    #KNN_canned(X_train, y_train, X_test, y_test)
    for k in range(1, 21):
        print("K = ", k)
        predicted_classes[k] = kNN_test(X_train, X_test, y_train, y_test, k)
        final_accuracies[k] = prediction_accuracy(predicted_classes[k], y_test)

    plt.figure(figsize=(15, 6))
    plt.plot(list(final_accuracies.keys()),list(final_accuracies.values()))
    plt.xticks(list(final_accuracies.keys()))
    plt.xlabel("k")
    plt.ylabel("Accuracy")
    plt.title("Plot of the prediction accuracy of KNN Classifier as a function of k (Number of Neighbours)")
    plt.show()