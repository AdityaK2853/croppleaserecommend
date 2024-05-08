from multiprocessing import reduction
from typing import List
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from fastapi import FastAPI, Form, Request
from fastapi.responses import HTMLResponse
import numpy as np
import pandas as pd
import sklearn
import pickle

model = pickle.load(open('NBClassifier.pkl', 'rb')) 
app = FastAPI()

@app.get("/", response_class=HTMLResponse)
async def index():
    content = """
    <html>
        <head>
            <title>Crop Prediction</title>
                <style>
        body {
            font-family: Arial, sans-serif;
            margin: 0;
            padding: 0;
            background-color: #f2f2f2;
            background-repeat: no-repeat;
            background-attachment: fixed;
            background-size: cover;
            background-position: center;
            background-image : url('/Users/adityakkoundinya/Desktop/Crop/download.jpeg');
        }
        
        .container {
            max-width: 500px;
            margin: 50px auto;
            background-color: #fff;
            padding: 20px;
            border-radius: 10px;
            box-shadow: 0 0 10px rgba(0, 0, 0, 0.1);
        }
        h1 {
            text-align: center;
        }
        label {
            font-weight: bold;
        }
        input[type="text"] {
            width: calc(100% - 20px);
            padding: 8px;
            margin-bottom: 10px;
            border: 1px solid #ccc;
            border-radius: 5px;
        }
        button {
            width: 100%;
            padding: 10px;
            background-color: #4CAF50;
            color: white;
            border: none;
            border-radius: 5px;
            cursor: pointer;
        }
        button:hover {
            background-color: #45a049;
        }
    </style>
</head>
<body>
<div class="background">
</div>
    <div class="container">
        <h1>Welcome to Crop Prediction</h1>
        <form id="predictionForm" method="post" action="/predict">
            <label for="Nitrogen">Nitrogen:</label>
            <input type="text" id="Nitrogen" name="Nitrogen"><br><br>
            <label for="Phosporus">Phosporus:</label>
            <input type="text" id="Phosporus" name="Phosporus"><br><br>
            <label for="Potassium">Potassium:</label>
            <input type="text" id="Potassium" name="Potassium"><br><br>
            <label for="Temperature">Temperature:</label>
            <input type="text" id="Temperature" name="Temperature"><br><br>
            <label for="Humidity">Humidity:</label>
            <input type="text" id="Humidity" name="Humidity"><br><br>
            <label for="pH">pH:</label>
            <input type="text" id="pH" name="pH"><br><br>
            <label for="Rainfall">Rainfall:</label>
            <input type="text" id="Rainfall" name="Rainfall"><br><br>
            <button type="button" onclick="submitForm()">Predict</button>
        </form>

        <div id="result"></div>
    </div>


        

            <script>
                function submitForm() {
                    var form = document.getElementById("predictionForm");
                    var formData = new FormData(form);
                    fetch("/predict", {
                        method: "POST",
                        body: formData
                    }).then(response => response.text())
                    .then(data => {
                        document.getElementById("result").innerHTML = data;
                    });
                }
            </script>
        </body>
    </html>
    """
    return HTMLResponse(content=content)

@app.post("/predict", response_class=HTMLResponse)
async def predict(request: Request):
    form = await request.form()
    feature_list = [int(form["Nitrogen"]), int(form["Phosporus"]), int(form["Potassium"]), 
                    float(form["Temperature"]), float(form["Humidity"]), float(form["pH"]), 
                    float(form["Rainfall"])]
    single_pred = np.array(feature_list).reshape(1, -1)

    prediction = model.predict(single_pred)
    
    crop_array = [
        'rice', 'maize', 'jute', 'cotton', 'coconut', 'papaya', 'orange', 'apple',
        'muskmelon', 'watermelon', 'grapes', 'mango', 'banana', 'pomegranate', 
        'lentil', 'blackgram', 'mungbean', 'mothbeans', 'pigeonpeas', 
        'kidneybeans', 'chickpea', 'coffee'
    ]

    predicted_crop_name = prediction[0]  # Convert prediction to integer

    if predicted_crop_name in crop_array:
        result = f"{predicted_crop_name} is the best crop to be cultivated"
    else:
        result = "No crop is predicted"

    return HTMLResponse(content=f"<p>{result}</p>")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port= 8000)
