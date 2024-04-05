from flask import Flask, request, jsonify
import pickle
import numpy as np
import pandas as pd
import pickle
from scipy.stats import mode
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
import dill

app = Flask(__name__)

# Load the pickled predictDisease function

with open('predictDisease.pkl', 'rb') as f:
	loaded_function = dill.load(f)
     
#print("lets print" + loaded_function("Itching,Skin Rash,Nodal Skin Eruptions"))

 
@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    symptoms = data['symptoms']
    prediction = loaded_function(symptoms)
    return jsonify({"prediction": prediction})
    
@app.route('/', methods=['GET'])
def test():
      return jsonify({"text": "hello"})

if __name__ == '__main__':
    app.run(debug=True)