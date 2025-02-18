import sys
import pandas as pd
from src.exception import Customexception
from src.utils import load_object

class PredictionPipeline:
    def __init__(self):
        try:
            self.model_path = "artifacts/model_trainer.pkl"
            self.preprocessor_path = "artifacts/preprocessor.pkl"
            self.model = load_object(self.model_path)
            self.preprocessor = load_object(self.preprocessor_path)
        except Exception as e:
            raise Customexception(sys, e)

    def predict(self, features):
        try:
            data_scaled = self.preprocessor.transform(features)
            preds = self.model.predict(data_scaled)
            return preds
        except Exception as e:
            raise Customexception(sys, e)


class CustomData:
    def __init__(self,
                 gender: str,
                 race_ethnicity: str,
                 parental_level_of_education,
                 lunch: str,
                 test_preparation_course: str,
                 reading_score: int,
                 writing_score: int):
        
        self.gender = gender
        self.race_ethnicity = race_ethnicity
        self.parental_level_of_education = parental_level_of_education
        self.lunch = lunch
        self.test_preparation_course = test_preparation_course
        self.reading_score = reading_score
        self.writing_score = writing_score

    def get_data_as_dataframe(self):
        try:
            data_dict = {
                "gender": [self.gender],
                "race_ethnicity": [self.race_ethnicity],
                "parental_level_of_education": [self.parental_level_of_education],
                "lunch": [self.lunch],
                "test_preparation_course": [self.test_preparation_course],
                "reading_score": [self.reading_score],
                "writing_score": [self.writing_score]
            }
            return pd.DataFrame(data_dict)
        except Exception as e:
            raise Customexception(sys, e)
