from flask import Flask, request, render_template
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from src.pipeline.predict_pipeline import Customdata, PredictPipeline  # Ensure these exist

app = Flask(__name__)

@app.route('/')  # Home page
def home():
    return render_template('home.html')

@app.route('/predictdata', methods=['POST'])
def predict():
    # Extract data from form
    data = Customdata(
        gender=request.form.get('gender'),
        race_ethnicity=request.form.get('race_ethnicity'),
        parental_level_of_education=request.form.get('parental_level_of_education'),
        lunch=request.form.get('lunch'),
        test_preparation_course=request.form.get('test_preparation_course'),
        reading_score=int(request.form.get('reading_score')),
        writing_score=int(request.form.get('writing_score'))
    )
    
    # Convert to DataFrame
    pred_df = data.get_data_as_dataframe()
    print(pred_df)
    
    # Load prediction pipeline
    predict_pipeline = PredictPipeline()  # Ensure this class is implemented
    preds = predict_pipeline.predict(pred_df)
    
    return render_template('home.html', results=preds[0])

if __name__ == '__main__':
    app.run(host="0.0.0.0", debug=True)
