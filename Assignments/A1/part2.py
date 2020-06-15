'''
    SVM Classifier
'''
import numpy as np
import pandas as pd

import sklearn.svm as svm
from sklearn.model_selection import KFold
from sklearn import metrics
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import itertools
from sklearn.model_selection import train_test_split

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

if __name__ == "__main__":
    col_names = [
        'Pregnancies', 
        'Glucose', 
        'BllodPressure', 
        'SkinThickness', 
        'Insulin', 
        'BMI',
        'DiabetesPedigreeFunction',
        'Age','Outcome'
    ]

    feature_cols = [
        'Pregnancies',
        'Glucose',
        'BllodPressure',
        'SkinThickness',
        'Insulin',
        'BMI',
        'DiabetesPedigreeFunction',
        'Age'
    ]
    
    data_file = pd.read_csv("data-sets/diabetes.csv")

    print("\n### 5-fold Validation Results ###")
    X = data_file.iloc[1:,:8]
    y = data_file.iloc[1:,8]
    fold_accuracies = []
    kf = KFold(n_splits=5)
    kf.get_n_splits(X)
    KFold(n_splits=5, random_state=None, shuffle=False)

    i = 1
    for train_index, test_index in kf.split(X):
        X_train, X_test = X.iloc[train_index], X.iloc[test_index]
        y_train, y_test = y.iloc[train_index], y.iloc[test_index]

        classifier = svm.SVC(kernel = 'linear', random_state = 0)
        classifier.fit(X_train, y_train)

        prediction = classifier.predict(X_test)
        print("Accuracy:",metrics.accuracy_score(y_test, prediction))

        cnf_matrix = confusion_matrix(y_test, prediction)

        plt.figure()
        plot_confusion_matrix(cnf_matrix, classes=["0", "1"], normalize=True, title='Confusion matrix, Normalized (Fold: ' + str(i) + ")")
        plt.show()
        i += 1

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1)

    classifier = svm.SVC(kernel = 'linear', random_state = 0)

    classifier = classifier.fit(X_train,y_train)
    y_pred = classifier.predict(X_test)

    print("\n### Overall Results ###")
    print("Accuracy:",metrics.accuracy_score(y_test, y_pred))

    cnf_matrix = confusion_matrix(y_test, y_pred)

    plt.figure()
    plot_confusion_matrix(cnf_matrix, classes=["0", "1"], normalize=True, title='Overall Confusion matrix, Normalized')
    plt.show()
    