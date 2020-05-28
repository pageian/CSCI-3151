import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeClassifier # Import Decision Tree Classifier
from sklearn.model_selection import train_test_split # Import train_test_split function
from sklearn import metrics
from sklearn.tree import export_graphviz
import io  
from IPython.display import Image  
import pydotplus

if __name__ == "__main__":
    col_names = ['Pregnancies', 'Glucose', 'BllodPressure', 'SkinThickness', 'Insulin', 'BMI',
        'DiabetesPedigreeFunction', 'Age', 'Outcome']
    data_file = pd.read_csv("diabetes.csv", header=None)
   
    feature_cols = ['Pregnancies', 'Glucose', 'BllodPressure', 'SkinThickness', 'Insulin', 'BMI',
        'DiabetesPedigreeFunction', 'Age']
    
    X = data_file.iloc[1:,:8] # Features
    y = data_file.iloc[1:,8] # Target variable
    # print(X)
    # print(y)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1)

    clf = DecisionTreeClassifier()

    # Train Decision Tree Classifer
    clf = clf.fit(X_train,y_train)

    #Predict the response for test dataset
    y_pred = clf.predict(X_test)

    print("Accuracy:",metrics.accuracy_score(y_test, y_pred))

    dot_data = io.StringIO()  
    export_graphviz(clf, out_file=dot_data, filled=True, rounded=True,
        special_characters=True,feature_names = feature_cols,class_names=['0','1'])
    graph = pydotplus.graph_from_dot_data(dot_data.getvalue())  
    graph.write_png('diabetes.png')
    Image(graph.create_png())