from flask import Flask, request, render_template
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from src.pipeline.predict_pipeline import CustomData, PredictionPipeline






application = Flask(__name__, template_folder="templates")

app=application

@app.route("/", methods=["GET", "POST"])
def home():
    if request.method == 'POST':
        return predict()
    return render_template("index.html")

@app.route('/predict', methods=['POST'])
def predict():
    data = CustomData(
        gender=request.form.get('gender'),
        race_ethnicity=request.form.get('race_ethnicity'),
        parental_level_of_education=request.form.get('parental_level_of_education'),
        lunch=request.form.get('lunch'),
        test_preparation_course=request.form.get('test_preparation_course'),
        reading_score=int(request.form.get('reading_score')),
        writing_score=int(request.form.get('writing_score'))
    )
    
    pred_df = data.get_data_as_dataframe()
    predict_pipeline = PredictionPipeline()
    preds = predict_pipeline.predict(pred_df)
    
    return render_template('index.html', results=preds[0])


def test_prediction():
    
    test_data = CustomData(
        gender="male",
        race_ethnicity="group A",
        parental_level_of_education="bachelor's degree",
        lunch="standard",
        test_preparation_course="completed",
        reading_score=70,
        writing_score=75
    )
    pred_df = test_data.get_data_as_dataframe()
    predict_pipeline = PredictionPipeline()
    preds = predict_pipeline.predict(pred_df)
    return True if isinstance(preds[0], (int, float)) else False

if __name__ == '__main__':
    
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == '--test':
        success = test_prediction()
        sys.exit(0 if success else 1)
    
   
    app.run(host="0.0.0.0")
