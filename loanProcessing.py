import os
from flask import Flask, request, render_template
import numpy as np
import joblib
import pandas as pd   
from sklearn import preprocessing
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from feature_engine.categorical_encoders import OneHotCategoricalEncoder
from sklearn.ensemble import GradientBoostingClassifier
from flask_wtf import FlaskForm
from wtforms import SubmitField, SelectField, DecimalField, IntegerField
from wtforms.validators import Length, InputRequired, AnyOf, NumberRange

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


class CreateApplicationForm(FlaskForm):
    #Categorical Variables
    Source = SelectField(label=('Source'), validators=[AnyOf(values=['CONSUMER','GATEWAY','LENDER'], message="Source status must be selected.")], choices=[('', 'Choose...'),('CONSUMER','CONSUMER'),('GATEWAY','GATEWAY'),('LENDER','LENDER')])
    EmploymentStatus = SelectField(label=('Employment Status'), validators=[AnyOf(values=['Employed','Retired','Unemployed','Others'], message="Employment Status status must be selected.")], choices=[('', 'Choose...'),('Employed','Employed'),('Retired','Retired'),('Unemployed','Unemployed'),('Others','Others')])
    
    isNewVehicle = SelectField(label=('New Vehicle'), validators=[AnyOf(values=['Y', 'N'], message="New Vehicle status must be selected.")], choices=[('', 'Choose...'),('Y','Y'),('N','N')])
    OccupancyStatus = SelectField(label=('Occupancy Status'), validators=[AnyOf(values=['RENT','OWN','BUYING','GOVQUARTERS','LIVEWITHPARENTS','OTHER'], message="Occupancy Status must be selected.")], choices=[('', 'Choose...'),('RENT','RENT'),('OWN','OWN'),('BUYING','BUYING'),('GOVQUARTERS','GOVQUARTERS'),('LIVEWITHPARENTS','LIVEWITHPARENTS'),('OTHER','OTHER')])
    
    RequestType     = SelectField(label=('Request Type'), validators=[AnyOf(values=['DEALER PURCHASE','INDIRECT','CAR SALE','PRIVATE PARTY','REFINANCE','TITLE LOAN','LEASE BUYOUT','A CAR SALE PREAPPROVAL','REFINANCE-PROMO','VEHICLE - CROSS SELL'], message="Request Type must be selected.")] , choices=[('', 'Choose...'),('DEALER PURCHASE','DEALER PURCHASE'),('INDIRECT','INDIRECT'),('CAR SALE','CAR SALE'),('PRIVATE PARTY','PRIVATE PARTY'),('REFINANCE','REFINANCE'),('TITLE LOAN','TITLE LOAN'),('LEASE BUYOUT','LEASE BUYOUT'),('A CAR SALE PREAPPROVAL','A CAR SALE PREAPPROVAL'),('REFINANCE-PROMO','REFINANCE-PROMO'),('VEHICLE - CROSS SELL','VEHICLE - CROSS SELL')])
    MemberIndicator = SelectField(label=('Member'), validators=[AnyOf(values=['Y', 'N'], message="Member Status must be selected.")], choices=[('', 'Choose...'),('Y','Y'),('N','N')])
    
    #Numerical Variables
    ModifiedCreditScore = DecimalField(label=('Credit Score'), validators=[InputRequired(message="Credit Score needs a value."), NumberRange(min=300, max=850, message='Credit Score must be between %(min)s and %(max)s.')])
    ModifiedBankruptcyScore = DecimalField(label=('Bankruptcy Score'), validators=[InputRequired(message="Bankruptcy Score needs a value."), NumberRange(min=0, message='Bankruptcy Score must be equal to or larger than %(min)s.')])
    
    EmployedMonths = IntegerField(label=('Employed Months'), validators=[InputRequired(message="Employed Months needs a value."), NumberRange(min=0, message='Employed Months must be equal to or larger than %(min)s.')])
    PrevEmployedMonths = IntegerField(label=('Previous Employed Months'), validators=[InputRequired(message="Previous Employed Months needs a value."), NumberRange(min=0, message='Previous Employed Months must be equal to or larger than %(min)s.')])
    
    PrimeMonthlyIncome = DecimalField(label=('Monthly Income'), validators=[InputRequired(message="Monthly Income needs a value."), NumberRange(min=0, message='Monthly Income must be equal to or larger than %(min)s.')])
    PrimeMonthlyLiability = DecimalField(label=('Monthly Liability'), validators=[InputRequired(message="Monthly Liability needs a value."), NumberRange(min=0, message='Monthly Liability must be equal to or larger than %(min)s.')])
    
    PrimeMonthlyRent = DecimalField(label=('Monthly Rent'), validators=[InputRequired(message="Monthly Rent needs a value."), NumberRange(min=0, message='Monthly Rent must be equal to or larger than %(min)s.')])
    TotalMonthlyDebtBeforeLoan = DecimalField(label=('Monthly Debt Before Loan'), validators=[InputRequired(message="Monthly Debt Before Loan needs a value."), NumberRange(min=0, message='Monthly Debt Before Loan must be equal to or larger than %(min)s.')])
    
    VehicleMileage = DecimalField(label=('Vehicle Mileage'), validators=[InputRequired(message="Vehicle Mileage needs a value."), NumberRange(min=0, message='Vehicle mileage must be equal to or larger than %(min)s.')])
    TotalVehicleValue = DecimalField(label=('Vehicle Value'), validators=[InputRequired(message="Vehicle Value needs a value."), NumberRange(min=0, message='Vehicle value must be equal to or larger than %(min)s.')])
    
    AmountRequested = DecimalField(label=('Amount Requested'), validators=[InputRequired(message="Amount Requested needs a value."), NumberRange(min=0, message='Amount requested must be larger than %(min)s.')])
    DownPayment = DecimalField(label=('Down Payment'), validators=[InputRequired(message="Down Payment needs a value."), NumberRange(min=0, message='Down Payment must be equal to or larger than %(min)s.')])
    
    Loanterm = IntegerField(label=('Loan Term'), validators=[InputRequired(message="Loan Term needs a value."), NumberRange(min=0, message='Loan Term must be equal to or larger than %(min)s.')])
    OccupancyDuration = IntegerField(label=('Occupancy Durantion'), validators=[InputRequired(message="Occupancy Duration needs a value."), NumberRange(min=0, message='Occupancy Duration must be equal to or larger than %(min)s.')])
    
    EstimatedMonthlyPayment = DecimalField(label=('Estimated Monthly Payment'), validators=[InputRequired(message="Estimated Monthly Payment needs a value."), NumberRange(min=0, message='Estimated Monthly Payment must be equal to or larger than %(min)s.')])
    NumberOfOpenRevolvingAccounts = IntegerField(label=('Number of Open Revolving Accounts'), validators=[InputRequired(message="Number of Open Revolving Accounts needs a value."), NumberRange(min=0, message='Number of Open Revolving Accounts must be equal to or larger than %(min)s.')])
    
    LTV = DecimalField(label=('Loan to Vehicle Value Ratio'), validators=[InputRequired(message="Loan to Vehicle Value Ratio needs a value."), NumberRange(min=0.1, max=3.0, message='Loan to Vehicle Value Ratio must be between %(min)s and %(max)s.')])
    DTI = DecimalField(label=('Debt to Income Ratio'), validators=[InputRequired(message="Debt to Income Ratio needs a value."), NumberRange(min=0, message='Debt to Income Ratio must be equal to or larger than %(min)s.')])
    
    #Submit Button
    submit = SubmitField(label=('Submit'))

app = Flask(__name__)
app.config['SECRET_KEY']='as#JNnfsjv39@*$*FB$*B@'

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/about')
def about():
    return render_template('about.html')

@app.route('/apply')
def apply():
    form = CreateApplicationForm()
    return render_template('application.html', form=form)

@app.route('/apply', methods=['POST'])
def loan_approval():
    
    form = CreateApplicationForm()
    
    if form.validate_on_submit():
        form_dictionary = request.form.to_dict()
        form_dataFrame = pd.DataFrame([form_dictionary])

        # Convert values from string to float

        for field in numerical_features:
            form_dataFrame[field] = pd.to_numeric(form_dataFrame[field])

        # 2) PREPROCESS DATA USING TRAINED PREPROCESSORS
        X = preprocessor.transform(form_dataFrame)

        # 3) PASS DATA INTO GRADIENT BOOSTING MODEL
        y_pred = gradBoost.predict(X)
        y_pred_prob = gradBoost.predict_proba(X)

        #print("Approved Loan: ", y_pred)
        #print("With a probability of: ", y_pred_prob)

        if y_pred == 1:
            loanStatus='Approved'
        elif y_pred == 0:
            loanStatus = 'Declined'

        probability = round(y_pred_prob[0][y_pred][0] * 100, 2)
        response = {"loanStatus": loanStatus, "probability":probability}

        #print("Loan is " + loanStatus + " with booleadn: " + str(y_pred[0]) + " and prob of : " + str(probability) + '%')
        #return jsonify(response)
        return render_template('results.html', outcome = response)
        
    return render_template('application.html', form=form)


if __name__ == '__main__':
    app.run(debug=True, use_reloader=False)