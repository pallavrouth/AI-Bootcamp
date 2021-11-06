import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import statsmodels.api as sm
import statsmodels.formula.api as smf
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
from sklearn.svm import SVC

filepath = "https://raw.githubusercontent.com/pallavrouth/AI-Bootcamp/main/Data/Smarket.csv"
smarket = pd.read_csv(filepath, index_col = "Unnamed: 0")
smarket.head()

# data characteristics
smarket.isnull().sum()
smarket.describe()
smarket.shape
smarket.corr()

# change DV type
smarket["Direction"].value_counts()

def label(x):
    if x == "Up": return 1
    else: return 0

smarket['direction'] = smarket['Direction'].apply(label)

# check
smarket.dtypes

# Stats model api
response = smarket[['direction']]
predictors = sm.add_constant(smarket.drop(columns = ['Direction','direction','Year','Today']))
logit_mod = sm.Logit(response, predictors).fit()
print(logit_mod.summary())
print(logit_mod.params)

# predictions - probabilities
logit_probs = pd.DataFrame(logit_mod.predict(), columns = ["predicted_probability"])

def prob_to_label(probability, threshold):
    if probability >= threshold: return 1
    else: return 0

logit_probs['predicted_direction'] = logit_probs.apply(lambda df: prob_to_label(df['predicted_probability'],0.5), axis = 1)

# Sklearn api

response = smarket['direction'].values
response.shape
predictors = smarket.drop(columns = ['Direction','direction','Year','Today']).values
predictors.shape

logistic_mod = LogisticRegression(penalty = "l1", solver = "liblinear")
logistic_mod_fit = logistic_mod.fit(predictors, response)

# predictions - probabilities
logistic_reg_probs = pd.DataFrame(logistic_mod_fit.predict_proba(predictors),
                                  columns = ['Up','Down'])
logistic_reg_probs

# predictions - labels
logistic_reg_labels = pd.DataFrame(logistic_mod_fit.predict(predictors),
                                   columns = ['labels'])

# confusion matrix
pd.DataFrame(confusion_matrix(response, logistic_mod_fit.predict(predictors)),
                              columns = ['PredDown', 'PredUp'],
                              index = ['ActualDown', 'ActualUp'])


# Repeat the same with a train test split
train_data = smarket.loc[smarket['Year'] < 2005,:]
test_data = smarket.loc[smarket['Year'] >= 2005,:]

response_train = train_data['direction'].values
predictors_train = train_data.drop(columns = ['Direction','direction','Year','Today']).values

response_test = test_data['direction'].values
predictors_test = test_data.drop(columns = ['Direction','direction','Year','Today']).values

logistic_mod = LogisticRegression(penalty = "l1", solver = "liblinear")
logistic_mod_fit = logistic_mod.fit(predictors_train, response_train)

pd.DataFrame(confusion_matrix(response_test, logistic_mod_fit.predict(predictors_test)),
                              columns = ['PredDown', 'PredUp'],
                              index = ['ActualDown', 'ActualUp'])

# repeat for SVM

svm_mod = SVC()
svm_mod_fit = svm_mod.fit(predictors_train, response_train)

pd.DataFrame(confusion_matrix(response_test, svm_mod_fit.predict(predictors_test)),
                              columns = ['PredDown', 'PredUp'],
                              index = ['ActualDown', 'ActualUp'])
