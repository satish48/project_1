import os
import sys
from src.exception import CustomException
from src.logger import logging
import pandas as pd

from sklearn.model_selection import train_test_split
from dataclasses import dataclass


@dataclass
class DataIngestionconfig:
    train_data_path: str = os.path.join('artifacts', "train.csv")
    test_data_path: str = os.path.join('artifacts', "test.csv")
    raw_data_path: str = os.path.join('artifacts', "raw.csv")


class DataIngestion:

    def __init__(self):
        self.ingestion_config = DataIngestionconfig()

    def begin_data_ingestion(self):
        logging.info("Initiated the data ingestion component")
        try:
            df = pd.read_csv("data/teen_phone_addiction_dataset.csv")
            logging.info("Read the dataset as df")

            os.makedirs(os.path.dirname(DataIngestionconfig.train_data_path), exist_ok=True) #saving the output

            df.to_csv(self.ingestion_config.raw_data_path,index=False, header=True)

            logging.info("Train Test Spilt started ")
            train_set, test_set=train_test_split(df,test_size=0.2,random_state=42)

            train_set.to_csv(self.ingestion_config.train_data_path,index = False, header = True)

            test_set.to_csv(self.ingestion_config.test_data_path, index= False, header = True)

            logging.info("Ingestion of the data is completed")

            return(
                self.ingestion_config.train_data_path,
                self.ingestion_config.test_data_path,
                
            )
        
        except Exception as e:
            raise CustomException(e, sys)


if __name__ == "__main__":
    obj = DataIngestion()
    obj.begin_data_ingestion()