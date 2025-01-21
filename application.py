from flask import Flask,jsonify, render_template, request
import pickle
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

application=Flask(__name__)
app=application

#import our model and standard-scaler
ridge_regressor=pickle.load(open('models/ridge.pkl','rb'))
standard_scaler=pickle.load(open('models/scaler.pkl','rb'))

@app.route('/')
def hello_world():
    return render_template('index.html')

@app.route('/predictData',methods=['GET','POST'])
def predictData():
    if request.method=='POST':
        Temperature=float(request.form.get('Temperature'))
        RH=float(request.form.get('RH'))
        Ws=float(request.form.get('Ws'))
        Rain=float(request.form.get('Rain'))
        FFMC=float(request.form.get('FFMC'))
        DMC=float(request.form.get('DMC'))
        ISI=float(request.form.get('ISI'))
        Classes=float(request.form.get('Classes'))
        Region=float(request.form.get('Region'))

        new_scaled_data=standard_scaler.transform([[Temperature,RH,Ws,Rain,FFMC,DMC,ISI,Classes,Region]])
        result=ridge_regressor.predict(new_scaled_data)

        return render_template('home.html',result=result)

    else:
        return render_template('home.html')

if __name__=='__main__':
    app.run(host='0.0.0.0',port=8000,debug=True)