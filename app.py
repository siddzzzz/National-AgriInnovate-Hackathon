from flask import Flask, request, jsonify, render_template 
import numpy as np
import joblib
import tensorflow as tf
import pandas as pd

app = Flask(__name__)

crop_pred = tf.keras.models.load_model("crop_recommender.h5")

crops_dict = {0: 'apple', 1: 'banana', 2: 'blackgram', 3: 'chickpea',
             4: 'coconut', 5: 'coffee', 6: 'cotton', 7: 'grapes', 8: 'jute',
             9: 'kidneybeans', 10: 'lentil', 11: 'maize', 12: 'mango',
             13: 'mothbeans', 14: 'mungbean', 15: 'muskmelon', 16: 'orange', 17: 'papaya',
             18: 'pigeonpeas', 19: 'pomegranate', 20: 'rice', 21: 'watermelon'}

data_limit = {0: 235, 1: 414, 2: 270, 3: 416, 4: 318, 5: 26, 6: 106, 7: 419, 8: 5,
             9: 418, 10: 267, 11: 298, 12: 285, 13: 418, 14: 418, 15: 247, 16: 165, 17: 135,
             18: 418, 19: 417, 20: 734, 21: 247}

state_data = pd.read_csv("state_data.csv")

##N,P,K,temp,humidity,ph,rainfall

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    feature = [x for x in request.form.values()]

    state = feature[4]

    temp = state_data[state_data['State'] == state]['temp'].tolist() 

    humid = state_data[state_data['State'] == state]['humidity'].tolist()
    
    rainfall = state_data[state_data['State'] == state]['rainfall'].tolist()

    
    final_features = np.array([[float(feature[0]), float(feature[1]), float(feature[2]), float(temp[0]), float(humid[0]), float(feature[3]), float(rainfall[0])]])

    print(final_features)

    predicted_crop_index = np.argmax(crop_pred.predict(final_features), axis=-1)[0]
    
    crop = crops_dict[predicted_crop_index]
    future_days = data_limit[predicted_crop_index]
    
    price_model = joblib.load(f"price_models/{crop}.pkl")
    
    future_data = np.array([[future_days + 120]])
    
    prediction = price_model.predict(future_data)

    output = round(prediction[0], 2)
    
    return render_template('index.html', prediction_text=f'Crop = {crop.capitalize()} and Predicted price after 4 months: {output}')

if __name__ == "__main__":
    app.run(host="127.0.0.9", port=8080, debug=True)
