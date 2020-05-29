import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split 
from sklearn.linear_model import LinearRegression
from sklearn import preprocessing
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import mean_squared_error, r2_score

# list of columns that require pre-processing
string_val_cols = [
    'MSZoning',
    'Street',
    'Alley',
    'LotShape',
    'LandContour',
    'Utilities',
    'LotConfig',
    'LandSlope',
    'Neighborhood',
    'Condition1',
    'Condition2',
    'BldgType',
    'HouseStyle',
    'RoofStyle',
    'RoofMatl',
    'Exterior1st',
    'Exterior2nd',
    'MasVnrType',
    'ExterQual',
    'ExterCond',
    'Foundation',
    'BsmtQual',
    'BsmtCond',
    'BsmtExposure',
    'BsmtFinType1',
    'BsmtFinType2',
    'Heating',
    'HeatingQC',
    'CentralAir',
    'Electrical',
    'KitchenQual',
    'Functional',
    'FireplaceQu',
    'GarageType',
    'GarageFinish',
    'GarageQual',
    'GarageCond',
    'PavedDrive',
    'PoolQC',
    'Fence',
    'MiscFeature',
    'SaleType',
    'SaleCondition']

# create the Labelencoder object
le = preprocessing.LabelEncoder()

# Importing the dataset
dataset = pd.read_csv("data-sets/housing-prices-dataset/train.csv")
X = dataset.iloc[:,1:len(dataset.loc[0,:])-1]
y = dataset.iloc[:, len(dataset.loc[0,:])-1]

print("processing data")
for col in string_val_cols:
    #print("Processing " + col)
    X[col] = le.fit_transform(X[col].astype(str))
X = np.nan_to_num(X)
y = np.nan_to_num(y)

for i in range(1,5):
    print("### Degree = " + str(i) + " ###")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

    # Visualizing the Linear Regression results
    poly_reg = PolynomialFeatures(degree=i)
    print("Fitting model to polyreg")
    X_poly = poly_reg.fit_transform(X_train)

    pol_reg = LinearRegression()
    print("Fitting Model to linreg")
    pol_reg.fit(X_poly, y_train)

    print("Making prediction")
    prediction = pol_reg.predict(X_poly)

    rmse = np.sqrt(mean_squared_error(y_train,prediction))
    r2 = r2_score(y_train,prediction)
    print(rmse)
    print(r2)

    # Visualizing the Polymonial Regression results
    def viz_polymonial():
        plt.scatter(range(len(X_test)), y_test, color='red')
        plt.plot(range(len(X_test)), pol_reg.predict(poly_reg.fit_transform(X_test)), color='blue')
        title = 'House Sale Price Estimation (Degree = ' + str(i) + ')'
        plt.title(title)
        plt.xlabel('House')
        plt.ylabel('Sale Price')
        plt.show()
        return
    viz_polymonial()