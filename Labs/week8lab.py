import numpy as np
import pandas as pd
import patsy as pt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn import tree
import graphviz 
from sklearn import metrics
from sklearn.metrics import confusion_matrix
from sklearn import datasets
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import train_test_split

#step 1 : get the data in
boston_data = pd.read_csv('https://raw.githubusercontent.com/dsnair/ISLR/master/data/csv/Boston.csv')
boston_data.head()

# step 2 : inspecting the data
# look for any column that's not numeric
boston_data.dtypes
# if a column isn't numeric you have change it
example_data = pd.DataFrame(['Yes','No','No','No'], columns = ['col1'])
example_data
what_i_want = [1,0,0,0]
# write a function that replaces yes with 1 and no with 0
example_data.col1 == 'Yes'
# look for any missing values in any of the columns

# step 3 : split data into training and testing
boston_data.shape
# features : X variables
# target : Y variables

# convert to numpy
target = boston_data['medv'].values
target

features = boston_data.drop(columns = ['medv']).values
features

features_train, features_test, target_train, target_test = train_test_split(features, target, test_size = 0.20, random_state = 42)

# step 4 : fit the model
clf = RandomForestRegressor(max_features = 4, random_state = 0, n_estimators = 100)
mod = clf.fit(features_train,target_train)
target_pred = clf.predict(features_test)
mse = metrics.mean_squared_error(target_test,target_pred)


results = []
n_features = [2,4,6,8,10]

for i in n_features:
    clf = RandomForestRegressor(max_features = i, random_state = 0, n_estimators = 100)
    mod = clf.fit(features_train,target_train)
    target_pred = clf.predict(features_test)
    mse = metrics.mean_squared_error(target_test,target_pred)
    rmse = np.sqrt(mse)
    results.append([i,rmse])

results2 = []
n_estimators = [50,100,150,200,250]

for i in n_features:
    for j in n_estimators:
        clf = RandomForestRegressor(max_features = i, random_state = 0, n_estimators = j)
        mod = clf.fit(features_train,target_train)
        target_pred = clf.predict(features_test)
        mse = metrics.mean_squared_error(target_test,target_pred)
        rmse = np.sqrt(mse)
        results2.append([i,j,rmse])


plot_df = pd.DataFrame(results2, columns = ['mtry','ntrees','rmse'])
sns.lineplot(x = 'mtry', y = 'rmse', hue = 'ntrees', data = plot_df)
plt.show()