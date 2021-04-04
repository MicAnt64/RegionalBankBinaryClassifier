import os
from flask import Flask, request, jsonify, url_for, render_template
import numpy as np
import joblib
import pandas as pd   
from sklearn import preprocessing
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from feature_engine.categorical_encoders import OneHotCategoricalEncoder
from sklearn.ensemble import GradientBoostingClassifier

preprocessor_file = 'models/preprocessor.pkl'
preprocessor = joblib.load(os.path.join(os.getcwd(), preprocessor_file))

gradBoost_file = 'models/gradientBoostinModel.pkl'
gradBoost = joblib.load(os.path.join(os.getcwd(), gradBoost_file))

numerical_features = ['ModifiedCreditScore',
 'ModifiedBankruptcyScore',
 'EmployedMonths',
 'PrevEmployedMonths',
 'PrimeMonthlyIncome',
 'PrimeMonthlyLiability',
 'PrimeMonthlyRent',
 'TotalMonthlyDebtBeforeLoan',
 'VehicleMileage',
 'TotalVehicleValue',
 'AmountRequested',
 'DownPayment',
 'Loanterm',
 'OccupancyDuration',
 'EstimatedMonthlyPayment',
 'NumberOfOpenRevolvingAccounts',
 'LTV',
 'DTI']

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/about')
def about():
    return render_template('about.html')

@app.route('/apply')
def apply():
    return render_template('application.html')

@app.route('/apply', methods=['POST'])
def loan_approval():
    # 1) GET DATA FROM FROM
    form_dictionary = request.form.to_dict()
    print("Dictionary: ", form_dictionary)
    form_dataFrame = pd.DataFrame([form_dictionary])
    print("Proc. Dict.: ", form_dataFrame)
    
    # Convert values from string to float
    for field in numerical_features:
        form_dataFrame[field] = pd.to_numeric(form_dataFrame[field])

    print("Updated form: ", form_dataFrame)
    
    # 2) PREPROCESS DATA USING TRAINED PREPROCESSORS
    X = preprocessor.transform(form_dataFrame)
    
    # 3) PASS DATA INTO GRADIENT BOOSTING MODEL
    y_pred = gradBoost.predict(X)
    y_pred_prob = gradBoost.predict_proba(X)
    
    print("Approved Loan: ", y_pred)
    print("With a probability of: ", y_pred_prob)
    
    if y_pred == 1:
        loanStatus='Approved'
    elif y_pred == 0:
        loanStatus = 'Declined'
        
    probability = round(y_pred_prob[0][y_pred][0] * 100, 2)
    
    response = {"loanStatus": loanStatus, "probability":probability}
            
    return render_template('results.html', outcome = response)

if __name__ == '__main__':
    app.run(debug=True, use_reloader=False)