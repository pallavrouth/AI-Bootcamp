
#----------- Monday
# Data science pipeline

#----------- Friday

import tensorflow as tf
import numpy as np
import pandas as pd

# apply the same set of steps to another data
cars_df = pd.read_csv('https://raw.githubusercontent.com/dsnair/ISLR/master/data/csv/Carseats.csv')
cars_df.head()

cars_df.dtypes
cars_df.columns

# split the dataset
from sklearn.model_selection import train_test_split
train_data, test_data = train_test_split(cars_df, test_size=0.2, random_state=42)

# get training and testing
def get_feats_and_labels(data, label):
  """ Take data and label as inputs, return features and labels separated """
  # drop the label column and save it in data_feats variable -> essentially getting everything other than the label column.
  data_feats = data.drop(label, axis=1)
  # labels of the dataset = label column in the dataframe.
  data_label = data[label]
  return data_feats, data_label

train_feats, train_label = get_feats_and_labels(train_data, 'Sales')

# preprocess the data
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OrdinalEncoder

scaler = StandardScaler() # normalization engine
encoder = OrdinalEncoder() # encoder engine

# The column transformer requires lists of features
train_feats.columns
num_feats = ['CompPrice','Income','Advertising','Population','Price','Age','Education']
cat_feats = ['ShelveLoc','Urban','US']

final_pipe = ColumnTransformer([           
   ('num', scaler, num_feats),  
   ('cat', encoder, cat_feats)])

# apply transformation
training_data_prepared = final_pipe.fit_transform(train_feats)
input_shape = training_data_prepared.shape[1:]
input_shape

# build model
model = tf.keras.Sequential([
    tf.keras.layers.Dense(128, activation = 'relu', input_shape = (10,)),
    tf.keras.layers.Dense(128, activation = 'relu'),
    tf.keras.layers.Dense(1, activation = 'relu')])

# Now we compile the model
model.compile(loss = 'mean_squared_error', optimizer = 'adam')
model.summary()

# fit the model
model.fit(x = training_data_prepared,y = train_label,epochs = 10)

# apply transformation to test data
test_feats, test_label = get_feats_and_labels(test_data, 'Sales')
test_data_prepared = final_pipe.transform(test_feats)

# evaluate on test data
model.evaluate(test_data_prepared, test_label) # OVERFITTING -> DROPOUT.