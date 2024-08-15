import requests
import pandas as pd

# Load the test datasets
X_test = pd.read_csv('TestDataSet/X_test.csv')
Y_test = pd.read_csv('TestDataSet/Y_test.csv')

X_test_json = X_test.values.tolist()  # Convert DataFrame to list of lists
Y_test_json = Y_test.iloc[:, 0].tolist()  # Adjust this to reference the correct column or index

# Construct the payload
payload = {
    'X_test': X_test_json,
    'Y_test': Y_test_json
}

# Send the POST request to the Flask API
response = requests.post('http://127.0.0.1:5000/predict', json=payload)

# Print the response (which includes predictions and accuracy)
print(response.json())