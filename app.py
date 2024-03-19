#!/usr/bin/env python
# coding: utf-8

# In[2]:


from flask import Flask, request, render_template
import numpy as np
import pickle

app = Flask(__name__)

# Load your trained model
model = pickle.load(open('../../../../../fish_classifier/Deployment/fish_classifier.pkl', 'rb'))

# Load the scaler
scaler = pickle.load(open('../../../../../fish_classifier/Deployment/sc.pkl', 'rb'))

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    # Extract features from form
    features = [float(x) for x in request.form.values()]
    # Scale the features using the loaded scaler
    final_features = scaler.transform([features])
    # Make prediction
    prediction = model.predict(final_features)
    
    output = prediction[0]

    # Return result
    return render_template('index.html', prediction_text=f'Species Predicted: {output}')

if __name__ == "__main__":
    app.run(debug=True)


# In[ ]:




