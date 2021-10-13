import numpy as np
from numpy.core.fromnumeric import mean
import pandas as pd
import patsy as pt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn import tree 
from sklearn import metrics
from sklearn.metrics import confusion_matrix
from sklearn import datasets
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import train_test_split

cars_df = pd.read_csv('https://raw.githubusercontent.com/dsnair/ISLR/master/data/csv/Carseats.csv')
cars_df.head()

# the shape of the data set
cars_df.shape
# the types of variables - fix ShelveLoc, Urban and US
cars_df.dtypes
# any missing variables?
cars_df.isnull().sum()

# a look at ShelveLoc
cars_df['ShelveLoc']
# factorize()
cars_df['ShelveLoc'].factorize()[0]

def factorize_col (data, columns):
  data[columns] = data[columns].factorize()[0]
  return data

factorize_col(cars_df,"US")
factorize_col(cars_df,"ShelveLoc")
factorize_col(cars_df,"Urban")

# write a small for loop to automate this
cars_df.dtypes == "object"

cars_df.dtypes

# define what our features (X) are and what our target or Y is
# checked that feature and target dimensions are the same - 400 rows
target = cars_df['Sales'].values
target
target.shape

features = cars_df.drop(columns = ['Sales']).values
features
features.shape

# split into training and testing
features_train, features_test, target_train, target_test = train_test_split(features, target, test_size = 0.30, random_state = 42)

# create an instance of decision tree
clf = tree.DecisionTreeRegressor(max_depth = 5)
# we will use our training data on this tree
mod = clf.fit(features_train, target_train)
# we are going to predict on our test data
target_pred = mod.predict(features_test)
# find out the error associated with the model
mse = metrics.mean_squared_error(target_test, target_pred)
print(np.sqrt(mse))

# using cross-validation to find out optimal depth of tree
# how CV works in sklear
clf = tree.DecisionTreeRegressor(max_depth = 5)
scores = cross_val_score(clf, features_train, target_train, 
                         cv = 5, scoring = 'neg_mean_squared_error')
print(-scores)
print(np.mean(-scores))

# iterate over values for max depth
max_depth = np.arange(1,10, 2)

results = []

for m in max_depth:
    regr = tree.DecisionTreeRegressor(max_depth = m)
    scores = cross_val_score(regr, features_train, target_train, cv = 5, scoring = 'neg_mean_squared_error')
    rmses = np.sqrt(np.absolute(scores))
    rmse = np.mean(rmses)
    conf_int = np.std(rmses) *2
    results.append([m, rmse, rmse+conf_int, rmse-conf_int])

print(results[0])

plot_df = pd.DataFrame(results, columns = ['maxdepth','rmse','ucl','lcl'])
sns.lineplot(x = 'maxdepth', y = 'rmse',  data = plot_df)
plt.show();

# build and predict a regression random forest
regr = RandomForestRegressor(max_features = 10, random_state = 0, n_estimators = 100)
mod = regr.fit(features_train, target_train)
target_pred = mod.predict(features_test)
mse = metrics.mean_squared_error(target_test, target_pred)
print(np.sqrt(mse))

# created feature importances
mod.feature_importances_
colname = list(cars_df.drop(columns = ['Sales']).columns)
feature_imp_dict = {'feature': colname, 'importance': mod.feature_importances_}
plot_df = pd.DataFrame(feature_imp_dict)

# plot feature importance
sns.barplot(x = 'importance', y = 'feature', data = plot_df.sort_values('importance', ascending=False),
            color = 'b')
plt.xticks(rotation=90);
plt.show();



# iterate over values for max depth
max_features = np.arange(1,10, 2)

results2 = []

for m in max_features:
    regr = RandomForestRegressor(max_features = m, random_state = 0, n_estimators = 100)
    mod = regr.fit(features_train, target_train)
    target_pred = mod.predict(features_test)
    mse = metrics.mean_squared_error(target_test, target_pred)
    rmse = np.sqrt(mse)
    results2.append([m, rmse])

print(results2[2])



plot_df = pd.DataFrame(results2, columns = ['maxfeatures','rmse'])
sns.lineplot(x = 'maxfeatures', y = 'rmse',  data = plot_df)
plt.show();

