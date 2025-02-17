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
from src.utils import save_object


@dataclass
class DataTransformationConfig:
    preprocessor_obj_file_path = os.path.join('artifacts', "preprocessor.pkl")


class DataTransformation:
    def __init__(self):
        self.data_transformation_config = DataTransformationConfig()

    def get_data_transformer_object(self):
        try:
            num_columns = ["writing_score", "reading_score"]
            categorical_columns = ["gender", "race_ethnicity", "parental_level_of_education", "lunch", "test_preparation_course"]

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
            logging.error(f"Error occurred while creating transformer object: {str(e)}")
            raise Customexception(e, sys)

    def initiate_data_transformation(self, train_path, test_path):
        try:
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)
            logging.info("Data loaded successfully.")

            logging.info(f"Train Columns: {train_df.columns.tolist()}")
            logging.info(f"Test Columns: {test_df.columns.tolist()}")

            target_column_name = "math_score"
            num_columns = ["writing_score", "reading_score"]
            categorical_columns = ["gender", "race_ethnicity", "parental_level_of_education", "lunch", "test_preparation_course"]

            if target_column_name not in train_df.columns:
                raise ValueError(f"Target column '{target_column_name}' not found in dataset.")

            missing_cols = [col for col in num_columns + categorical_columns if col not in train_df.columns]
            if missing_cols:
                raise ValueError(f"Missing columns in dataset: {missing_cols}")

            
            X_train = train_df.drop(columns=[target_column_name], axis=1)
            X_test = test_df.drop(columns=[target_column_name], axis=1)
            y_train = train_df[target_column_name]
            y_test = test_df[target_column_name]

           
            processing_obj = self.get_data_transformer_object()

           
            training_arr = processing_obj.fit_transform(X_train)
            test_arr = processing_obj.transform(X_test)

            
            train_arr = np.c_[training_arr, np.array(y_train)]
            test_arr = np.c_[test_arr, np.array(y_test)]

            
            save_object(
                file_path=self.data_transformation_config.preprocessor_obj_file_path,
                obj=processing_obj
            )

            logging.info("Data transformation completed successfully.")
            return train_arr, test_arr, processing_obj

        except Exception as e:
            logging.error(f"Error during data transformation: {str(e)}")
            raise Customexception(e, sys)
