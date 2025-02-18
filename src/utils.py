import os
import sys
import dill
import numpy as np
import pandas as pd
from src.exception import Customexception
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import r2_score    

def save_object(file_path, obj):
    try:
        dir_path = os.path.dirname(file_path)
        os.makedirs(dir_path, exist_ok=True)

        with open(file_path, 'wb') as file:
            dill.dump(obj, file)

    except Exception as e:
        raise Customexception(e, sys)

def evaluate_model(x_train, y_train, x_test, y_test, models):
    try:
        report = {}

        for name, model in models.items():
            
            model.fit(x_train, y_train)
            y_pred = model.predict(x_test)
            
            train_model_r2 = r2_score(y_train, model.predict(x_train))
            test_model_r2 = r2_score(y_test, y_pred)

            report[name] = {'train_model_r2': train_model_r2, 'test_model_r2': test_model_r2}

        return report

    except Exception as e:
        raise Customexception(e, sys)
    
    
    
    
def load_object(file_path):
     try:
        with open(file_path, 'rb') as file:
            return dill.load(file)  
        
     except Exception as e:
        raise Customexception(e, sys)
