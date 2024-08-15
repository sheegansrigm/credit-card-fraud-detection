import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
new_dataset = pd.read_csv('DataSet/new_dataset.csv')
print(new_dataset.head())
print(new_dataset.tail())
print(new_dataset['Class'].value_counts())
print(new_dataset.groupby('Class').mean())
X = new_dataset.drop(columns='Class', axis=1)
Y = new_dataset['Class']
print(X)
print(Y)
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, stratify=Y, random_state=2)
print(X.shape, X_train.shape, X_test.shape)
# Convert X_train, X_test, Y_train, Y_test to DataFrame and save as CSV
X_train.to_csv('TrainDataSet/X_train.csv', index=False)
X_test.to_csv('TestDataSet/X_test.csv', index=False)
Y_train.to_csv('TrainDataSet/Y_train.csv', index=False)
Y_test.to_csv('TestDataSet/Y_test.csv', index=False)
