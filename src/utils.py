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

def evaluate_model(x_train, y_train, x_test, y_test, models, params):
    try:
        report = {}

        for name, model in models.items():
            param_grid = params.get(name, {})
            
            if param_grid:
                gs = GridSearchCV(model, param_grid, cv=3, n_jobs=-1)
                gs.fit(x_train, y_train)
                model.set_params(**gs.best_params_)
            
            model.fit(x_train, y_train)
            y_pred = model.predict(x_test)
            
            train_model_r2 = r2_score(y_train, model.predict(x_train))
            test_model_r2 = r2_score(y_test, y_pred)

            report[name] = {'train_model_r2': train_model_r2, 'test_model_r2': test_model_r2}

        return report

    except Exception as e:
        raise Customexception(e, sys)
