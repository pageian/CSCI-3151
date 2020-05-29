'''
    SVM Classifier
'''
import numpy as np
import pandas as pd

import sklearn.svm as svm
from sklearn.pipeline import Pipeline
from sklearn.model_selection import KFold
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer

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

    print("\n[5-fold Validation Results ###]")
    X = data_file.iloc[1:,:8] # Features
    y = data_file.iloc[1:,8] # Target variable
    #y = y.reshape((len(y),1))
    fold_accuracies = []
    kf = KFold(n_splits=5)
    kf.get_n_splits(X)
    KFold(n_splits=5, random_state=None, shuffle=False)

    for train_index, test_index in kf.split(X):
        X_train, X_test = X.iloc[train_index], X.iloc[test_index]
        y_train, y_test = y.iloc[train_index], y.iloc[test_index]
        # y_train = y_train.reshape(len(y_train), 1)

        # Using pipeline for a quick implementation https://scikit-learn.org/stable/modules/generated/sklearn.pipeline.Pipeline.html
        text_clf_svm = Pipeline([('vect', CountVectorizer(lowercase=False)),('tf-idf', TfidfTransformer()),('clf-svm', svm.SVC()),])

        X_train.fillna('0')
        y_train.fillna('0')
        print(X_train.dtypes)
        print(y_train.dtypes)
        
        # X_train = X_train.values.astype('unicode')
        # y_train = y_train.values.astype('unicode')
        # y_train =y_train.reshape((train_index,1))
        # y_train =y_train.reshape((train_index,1))
        # print(y_train.shape)
        
        # print(y_train)


        text_clf_svm.fit(X_train.to_string(), y_train.to_string())

        predicted_svm = text_clf_svm.predict(X_test)
        np.mean(predicted_svm == y_test)

    