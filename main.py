import json
import pickle
import joblib
from flask import Flask,request,app,render_template
import numpy as np
import pandas as pd
app=Flask(__name__)
import xgboost as xgb

model=pickle.load(open('model.sav','rb'))
# model = joblib.load("model2.pkl")
@app.route('/')
def home():
    return render_template('home.html')

@app.route('/predict',methods=['POST'])
def predict():
    data=[float(x) for x in request.form.values()]
    print(data)
    final_input=model.predict(np.array(data).reshape(1,-1))[0]
    print(final_input)

    return render_template("home.html",prediction_text="The House price is {}".format(final_input))


if __name__=="__main__":
    app.run(debug=True)