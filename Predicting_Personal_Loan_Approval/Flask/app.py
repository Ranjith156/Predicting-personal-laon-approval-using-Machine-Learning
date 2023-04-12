import pandas as pd
import os
import numpy as np
import pickle
from flask import Flask, render_template, request


app=Flask(__name__)
model = pickle.load(open(r'rdf.pkl', 'rb'))
scale = pickle.load(open(r'scale.pkl','rb'))


@app.route('/') #rendering the html template
def home():
    return render_template('predict.html')

@app.route('/predict',methods=["POST", "GET"]) #rendering the html
def predict():
    return render_template('predict.html')


@app.route('/submit', methods=["POST", "GET"])  #rout to show the predictions in a web UI
def submit():
    #redaing the inputs given by the user
    input_feature=[int(x) for x in request.form.values()]
    #input_feature = np.transpose (input_feature)
    input_feature =[np.array(input_feature)]
    print(input_feature)
    names = ['Gender', 'Married', 'Dependents', 'Education', 'Self_Employed', 'ApplicantIncome', 'CoapplicantIncome', 'LoanAmount', 'Loan_Amount_Term', 'Creadit_History', 'Property_Area']
    data = pandas.DataFrame(input_feature, colums=names)
    print(data)
    
    data_scaled = scale.fit_transform(data)
    data = pandas.DataFrame(data,columns=names)
    
    #predictions using the loaded model file
    prediction = model.predict(data)
    print(prediction)
    prediction = int(prediction)
    print(type(prediction))
    
    if (prediction == 0):
        return render_template('predict.html', result="Loan will Not be Approved")
    else:
        return render_template('predict.html', result = "Loan will be Approved")
    
    
    
if __name__ =="__main__":
    app.run(debug=True) #running the app
    