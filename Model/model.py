import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import joblib

X_train = pd.read_csv('TrainDataSet/X_train.csv')
Y_train = pd.read_csv('TrainDataSet/Y_train.csv')

model = LogisticRegression()
model.fit(X_train, Y_train)

X_train_prediction = model.predict(X_train)
training_data_accuracy = accuracy_score(X_train_prediction, Y_train)

print('Accuracy on Training data : ', training_data_accuracy)

joblib.dump(model, 'Model/logistic_regression_model.pkl')

print('Model saved successfully!')
