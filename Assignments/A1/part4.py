import numpy as np
import pandas as pd
import numpy as np
import pandas as pd

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

def minkowski_distance(x, y, p_value):  
    return np.divide(np.sum(np.power((np.abs(x-y)), p_value)),(1 / p_value))

def get_neighbours(X_train, X_test_instance, k, p):
    distances = []
    neighbors = []
    for i in range(0, X_train.shape[0]):
        dist = minkowski_distance(X_train[i], X_test_instance, p)
        distances.append((i, dist))
    distances.sort(key=operator.itemgetter(1))
    for x in range(k):
        neighbors.append(distances[x][0])
    return neighbors

def predictKNNClass(output, y_train):
    classVotes = {}
    for i in range(len(output)):
        if y_train.values[output[i], 0] in classVotes:
            classVotes[y_train.values[output[i], 0]] += 1
        else:
            classVotes[y_train.values[output[i], 0]] = 1
    sortedVotes = sorted(classVotes.items(), key=operator.itemgetter(1), reverse=True)
    return sortedVotes[0][0]

def KNN(X_train, X_test, Y_train, Y_test, k, p):
    output_classes = []
    for i in range(0, (X_test.shape[0])):
        output = get_neighbours(X_train.values, X_test.iloc[i,:].values, k, p)
        predictedClass = predictKNNClass(output, Y_train)
        output_classes.append(predictedClass)
    return output_classes

def prediction_accuracy(predicted_labels, original_labels):
    count = 0
    for i in range(len(predicted_labels)):
        if predicted_labels[i] == original_labels.values[i, 0]:
            count += 1
    return float(count)/len(predicted_labels)

if __name__ == '__main__':
    X_train = pd.read_csv("data-sets/reducedMNIST/train.csv")
    y_train = pd.read_csv("data-sets/reducedMNIST/train_labels.csv")
    X_test = pd.read_csv("data-sets/reducedMNIST/test.csv")
    y_test = pd.read_csv("data-sets/reducedMNIST/test_labels.csv")
    
    final_accuracies = []

    for p in range(1, 6):
        predicted_classes = {}
        accuracies = {}
        for k in range(1, 6):
            predicted_classes[k] = KNN(X_train, X_test, y_train, y_test, k, p)
            accuracies[k] = prediction_accuracy(predicted_classes[k], y_test)
            prediction_accuracy(predicted_classes[k], y_test)
            
            cnf_matrix = confusion_matrix(y_test, predicted_classes[k])

            title = 'Confusion matrix, Normalized(k=' + str(k) + ', p=' + str(p) + ')'
            plt.figure()
            plot_confusion_matrix(cnf_matrix, classes=["1", "2", "3", "4", "5", "6", "7", "8", "9", "10"], normalize=True, title=title)
            plt.show()

        final_accuracies.append(accuracies)
        
    plt.figure(figsize=(15, 6))
    p = 1
    for accuracy in final_accuracies:
        label = 'p=' + str(p)
        plt.plot(list(accuracy.keys()),list(accuracy.values()), label=label)
        p += 1
    
    plt.legend(loc="upper right")
    plt.xlabel("k")
    plt.ylabel("Accuracy")
    plt.title("Plot of the prediction accuracy of KNN Classifier as a function of k (Number of Neighbours)")
    plt.show()