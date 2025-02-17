import os
import sys
import pandas as pd
from sklearn.model_selection import train_test_split
from dataclasses import dataclass
from src.exception import Customexception  
from src.logger import logging  

@dataclass
class DataIngestionConfig:
    train_data_path: str = os.path.join('artifacts', 'train.csv')
    test_data_path: str = os.path.join('artifacts', 'test.csv')
    raw_data_path: str = os.path.join('artifacts', 'data.csv')

class DataIngestion:
    def __init__(self):
        self.ingestion_config = DataIngestionConfig()

    def initiate_data_ingestion(self):
        logging.info('Entered the data ingestion method or component')
        try:
            df = pd.read_csv(r'C:\Users\Garv Khurana\OneDrive\Desktop\end_to_end 2\data.csv')  
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
    data_ingestion.initiate_data_ingestion()
