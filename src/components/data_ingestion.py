import os
import sys
import pandas as pd
from sklearn.model_selection import train_test_split
from dataclasses import dataclass
from src.exception import Customexception  
from src.logger import logging
from data_transformation import DataTransformation  
from model_training import ModelTrainer
from src.utils import read_yaml, create_directories
from src.constants import CONFIG_FILE_PATH

@dataclass
class DataIngestionConfig:
    root_dir: str
    raw_data_path: str
    train_data_path: str
    test_data_path: str
    test_size: float
    random_state: int

class DataIngestion:
    def __init__(self, config_path=CONFIG_FILE_PATH):
        try:
            self.config = read_yaml(config_path)  
            ingestion_config = self.config["data_ingestion"]

            self.ingestion_config = DataIngestionConfig(
                root_dir=ingestion_config["root_dir"],
                raw_data_path=ingestion_config["raw_data_path"],
                train_data_path=ingestion_config["train_data_path"],
                test_data_path=ingestion_config["test_data_path"],
                test_size=ingestion_config["test_size"],
                random_state=ingestion_config["random_state"]
            )

           
            create_directories([self.ingestion_config.root_dir])

        except Exception as e:
            raise Customexception(e, sys)

    def initiate_data_ingestion(self):
        logging.info('Entered the data ingestion method or component')
        try:
            df = pd.read_csv(r'C:\Users\Garv Khurana\OneDrive\Desktop\end_to_end 2\artifacts\data.csv')  
            logging.info('Data read successfully')

            os.makedirs(os.path.dirname(self.ingestion_config.train_data_path), exist_ok=True)

            df.to_csv(self.ingestion_config.raw_data_path, index=False, header=True)
            logging.info('Raw data saved successfully')

            train_set, test_set = train_test_split(df, test_size=0.2, random_state=42)
            logging.info('Train-test split done successfully')

            train_set.to_csv(self.ingestion_config.train_data_path, index=False, header=True)
            logging.info('Train data saved successfully')

            test_set.to_csv(self.ingestion_config.test_data_path, index=False, header=True)
            logging.info('Test data saved successfully')

            return (
                self.ingestion_config.train_data_path,
                self.ingestion_config.test_data_path,
            )
        except Exception as e:
            logging.error('Error occurred while ingesting the data')
            raise Customexception(e, sys)

if __name__ == '__main__':
    data_ingestion = DataIngestion()
    train_data,test_data=data_ingestion.initiate_data_ingestion()
    
    data_transformation=DataTransformation()
    train_array1,test_array1,_=data_transformation.initiate_data_transformation(train_data,test_data)
    
    model_trainer=ModelTrainer()
    print(model_trainer.initiate_model_trainer(train_array1,test_array1))
    