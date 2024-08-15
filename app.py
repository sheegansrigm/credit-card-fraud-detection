from flask import Flask, request, jsonify, send_file
import joblib
import pandas as pd
from sklearn.metrics import accuracy_score

app = Flask(__name__)

# Load the saved model
model = joblib.load('Model/logistic_regression_model.pkl')

# Load X_test and Y_test for download purposes
X_test = pd.read_csv('TestDataSet/X_test.csv')
Y_test = pd.read_csv('TestDataSet/Y_test.csv')

# Endpoint to download the test datasets
@app.route('/download_test_data', methods=['GET'])
def download_test_data():
    # Send the X_test and Y_test files for download
    return jsonify({"message": "Use /download_x_test and /download_y_test to download the datasets."})

@app.route('/download_x_test', methods=['GET'])
def download_x_test():
    return send_file('TestDataSet/X_test.csv', as_attachment=True)

@app.route('/download_y_test', methods=['GET'])
def download_y_test():
    return send_file('TestDataSet/Y_test.csv', as_attachment=True)

# Endpoint to predict using the saved model and uploaded test data
@app.route('/predict', methods=['POST'])
def predict():
    # Expecting JSON with the test data
    data = request.get_json(force=True)
    
    # Parse X_test and Y_test from the uploaded data
    X_test_uploaded = pd.DataFrame(data['X_test'])
    Y_test_uploaded = pd.Series(data['Y_test'])
    
    # Make predictions using the saved model
    X_test_prediction = model.predict(X_test_uploaded)
    
    # Calculate accuracy
    test_data_accuracy = accuracy_score(Y_test_uploaded, X_test_prediction)
    
    # Return the predictions and accuracy
    return jsonify({
        'predictions': X_test_prediction.tolist(),
        'accuracy': test_data_accuracy
    })

if __name__ == '__main__':
    app.run(debug=True)
