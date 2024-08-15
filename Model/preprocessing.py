import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
credit_card_data = pd.read_csv('DataSet/creditcard.csv')
print(credit_card_data.head())
print(credit_card_data.tail())
print(credit_card_data.info())
print(credit_card_data.isnull().sum())
print(credit_card_data['Class'].value_counts())
legit = credit_card_data[credit_card_data.Class == 0]
fraud = credit_card_data[credit_card_data.Class == 1]
print(legit.shape)
print(fraud.shape)
print(legit.Amount.describe())
print(credit_card_data.groupby('Class').mean())
legit_sample = legit.sample(n=492)
new_dataset = pd.concat([legit_sample, fraud], axis=0)
new_dataset.to_csv('DataSet/new_dataset.csv', index=False)
