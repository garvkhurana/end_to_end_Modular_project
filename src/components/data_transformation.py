import os
import sys
from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline

from src.exception import Customexception
from src.logger import logging


@dataclass
class DataTransformationConfig:
    preprocessor_obj_file_path = os.path.join('artifacts', "preprocessor.pkl")


class DataTransformation:
    def __init__(self):
        self.data_transformation_config = DataTransformationConfig()

    def get_data_transformer_object(self):
        try:
            num_columns = ["writing score", "reading score", "math score"]
            categorical_columns = ["gender", "race_ethnicity", "parental level of education", "lunch", "test preparation course"]

            # Numerical pipeline
            num_pipeline = Pipeline(steps=[
                ('imputer', SimpleImputer(strategy='median')),
                ('scaler', StandardScaler())
            ])

            logging.info("Numerical columns transformation completed.")

            # Categorical pipeline
            cat_pipeline = Pipeline(steps=[
                ('imputer', SimpleImputer(strategy='most_frequent')),
                ('onehot', OneHotEncoder(handle_unknown='ignore'))
            ])

            logging.info("Categorical columns encoding completed.")

            # Column Transformer
            preprocessor = ColumnTransformer(transformers=[
                ('num', num_pipeline, num_columns),
                ('cat', cat_pipeline, categorical_columns)
            ])

            return preprocessor

        except Exception as e:
            raise Customexception(e, sys)
            logging.error("Error occurred while transforming the data.")