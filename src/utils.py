import os
import sys
import dill
import numpy as np
import pandas as pd
from src.exception import Customexception
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import r2_score   
import yaml
from src.logging import logger
from ensure import ensure_annotations
from box import ConfigBox
from pathlib import Path
from src.exception import Customexception

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



import os
import yaml
from src.logging import logger
from ensure import ensure_annotations
from box import ConfigBox
from pathlib import Path
import sys
from src.exception import Customexception


@ensure_annotations
def read_yaml(path_to_yaml: Path) -> ConfigBox:
    """reads yaml file and returns . type key value pairs

    Args:
        path_to_yaml (str): path like input

    Raises:
        ValueError: if yaml file is empty
        e: empty file

    Returns:
        ConfigBox: ConfigBox type
    """
    try:
        with open(path_to_yaml) as yaml_file:
            content = yaml.safe_load(yaml_file)
            logger.info(f"yaml file: {path_to_yaml} loaded successfully")
            return ConfigBox(content)
    
    except Exception as e:
        raise Customexception(e,sys)
    
@ensure_annotations
def create_directories(path_to_directories: list, verbose=True):
    """create list of directories

    Args:
        path_to_directories (list): list of path of directories
        ignore_log (bool, optional): ignore if multiple dirs is to be created. Defaults to False.
    """
    for path in path_to_directories:
        os.makedirs(path, exist_ok=True)
        if verbose:
            logger.info(f"created directory at: {path}")


