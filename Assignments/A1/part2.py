'''
    Descision Tree Classifier w/ confusion matrix plotting
'''
import itertools
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier # Import Decision Tree Classifier
from sklearn.model_selection import train_test_split # Import train_test_split function
from sklearn import metrics
from sklearn.tree import export_graphviz
import io  
from IPython.display import Image  
import pydotplus
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import KFold
from sklearn.linear_model import Ridge, RidgeCV, ElasticNet, LassoCV, LassoLarsCV
from sklearn.model_selection import cross_val_score

def plot_confusion_matrix(cm, classes,
                        normalize=False,
                        title='Confusion matrix',
                        cmap=plt.cm.Blues):

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        #print("Normalized confusion matrix")
    #else:
        #print('Confusion matrix, without normalization')

    #print(cm)

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
    col_names = ['Pregnancies', 'Glucose', 'BllodPressure', 'SkinThickness', 'Insulin', 'BMI',
        'DiabetesPedigreeFunction', 'Age', 'Outcome']
    data_file = pd.read_csv("data-sets/diabetes.csv", header=None)
   
    feature_cols = ['Pregnancies', 'Glucose', 'BllodPressure', 'SkinThickness', 'Insulin', 'BMI',
        'DiabetesPedigreeFunction', 'Age']
    
    X = data_file.iloc[1:,:8] # Features
    y = data_file.iloc[1:,8] # Target variable
    fold_accuracies = []
    kf = KFold(n_splits=5)
    kf.get_n_splits(X)
    KFold(n_splits=5, random_state=None, shuffle=False)
    print("\n### 5-fold Validation Results ###\n")
    for train_index, test_index in kf.split(X):
        # print("TRAIN:", train_index, "TEST:", test_index, "\n")
        X_train, X_test = X.iloc[train_index], X.iloc[test_index]
        y_train, y_test = y.iloc[train_index], y.iloc[test_index]
        # print("TEST")
        clf = DecisionTreeClassifier()
        clf = clf.fit(X_train,y_train)
        y_pred = clf.predict(X_test)
        print("Accuracy:",metrics.accuracy_score(y_test, y_pred))
        fold_accuracies.append(metrics.accuracy_score(y_test, y_pred))
        cnf_matrix = confusion_matrix(y_test, y_pred)
        plt.figure()
        plot_confusion_matrix(cnf_matrix, classes=["0", "1"], title='Confusion matrix, without normalization')
        #plt.show()

    print("\nStandard Dev: ", np.std(fold_accuracies))
    # print(X)
    # print(y)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1)

    clf = DecisionTreeClassifier()

    # Train Decision Tree Classifer
    clf = clf.fit(X_train,y_train)

    #Predict the response for test dataset
    y_pred = clf.predict(X_test)

    print("\n### Overall Results ###\n")
    print("Accuracy:",metrics.accuracy_score(y_test, y_pred))

    dot_data = io.StringIO()  
    export_graphviz(clf, out_file=dot_data, filled=True, rounded=True,
        special_characters=True,feature_names = feature_cols,class_names=['0','1'])
    graph = pydotplus.graph_from_dot_data(dot_data.getvalue())  
    graph.write_png('diabetes.png')
    Image(graph.create_png())

cnf_matrix = confusion_matrix(y_test, y_pred)

plt.figure()
plot_confusion_matrix(cnf_matrix, classes=["0", "1"], title='Confusion matrix, without normalization')
plt.show()