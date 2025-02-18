import os
import sys
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, AdaBoostRegressor, GradientBoostingRegressor
from dataclasses import dataclass
from catboost import CatBoostRegressor
from xgboost import XGBRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import r2_score
from src.utils import evaluate_model, save_object
from src.exception import Customexception
from src.logger import logging

@dataclass
class ModelTrainerConfig:
    model_trainer_file_path = os.path.join('artifacts', 'model_trainer.pkl')

class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()

    def initiate_model_trainer(self, train_array, test_array):
        try:
            logging.info("Splitting the data into features and target")
            x_train, y_train, x_test, y_test = (
                train_array[:, :-1], train_array[:, -1], test_array[:, :-1], test_array[:, -1]
            )
            
            models = {
                "Random Forest": RandomForestRegressor(),
                "Decision Tree": DecisionTreeRegressor(),
                "Gradient Boosting": GradientBoostingRegressor(),
                "Linear Regression": LinearRegression(),
                "XGBRegressor": XGBRegressor(),
                "CatBoosting Regressor": CatBoostRegressor(verbose=False),
                "AdaBoost Regressor": AdaBoostRegressor(),
            }
            
            models = {
                "Random Forest": RandomForestRegressor(),
                "Decision Tree": DecisionTreeRegressor(),
                "Gradient Boosting": GradientBoostingRegressor(),
                "Linear Regression": LinearRegression(),
                "XGBRegressor": XGBRegressor(),
                "CatBoosting Regressor": CatBoostRegressor(verbose=False),
                "AdaBoost Regressor": AdaBoostRegressor(),
            }
            

            model_report = evaluate_model(x_train, y_train, x_test, y_test, models)

            best_model_name = max(model_report, key=lambda x: model_report[x]['test_model_r2'])
            best_model_score = model_report[best_model_name]['test_model_r2']

            if best_model_score < 0.6:
                raise Customexception("No suitable model found with r2_score >= 0.6")

            logging.info(f"Best model found: {best_model_name} with R² = {best_model_score}")

            best_model = models[best_model_name]

            save_object(
                file_path=self.model_trainer_config.model_trainer_file_path,
                obj=best_model
            )

            predicted = best_model.predict(x_test)
            r2_square = r2_score(y_test, predicted)

            return r2_square

        except Exception as e:
            logging.error(f"Error in model training: {str(e)}")
            raise Customexception(e, sys)
