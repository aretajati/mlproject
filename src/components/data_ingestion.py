import os
import sys
from src.exception import CustomException
from src.logger import logging
import pandas as pd

from sklearn.model_selection import train_test_split
from dataclasses import dataclass


## Determine the output path of this data ingestion
@dataclass
class DataIngestionConfig:
        train_data_path: str=os.path.join('artifacts','train.csv')
        test_data_path: str=os.path.join('artifacts','test.csv')
        raw_data_path: str=os.path.join('artifacts','data.csv')

class DataIngestion:
    def __init__(self):
        self.ingestion_config=DataIngestionConfig()
    
    def initiate_data_ingestion(self):
        logging.info("Entered the data ingestion method or component")
        try:
            # change the code below with the source of data (API, MangoDB, etc)
            df=pd.read_csv('notebook\data\stud.csv') 
            logging.info("Read the dataset as dataframe")

            # create the 'artifacts' folder
            os.makedirs(os.path.dirname(self.ingestion_config.train_data_path),exist_ok=True)

            # save the input data into raw data
            df.to_csv(self.ingestion_config.raw_data_path,index=False,header=True)

            logging.info("Train test split initiated")
            train_set, test_set=train_test_split(df,test_size=0.2,random_state=32)

            # save the train data into csv
            train_set.to_csv(self.ingestion_config.train_data_path,index=False,header=True)
            
            # save the test data into csv
            test_set.to_csv(self.ingestion_config.test_data_path,index=False,header=True)

            logging.info("Ingestion of the data is completed")
            
            # return the file location of train and test data,
            # bcs the transformation data will use /read the train&test data from here.
            return(
                self.ingestion_config.train_data_path,
                self.ingestion_config.test_data_path
            )
        except Exception as e:
            logging.error(f"An error occurred: {e}")
            print(f"An error occurred: {e}")  # Immediate feedback
            raise CustomException(e,sys)


if __name__=="__main__":
    obj=DataIngestion()
    obj.initiate_data_ingestion()

### WRITE CODE BELOW TO RUN THIS DATA IN THE TERMINAL
### python -m src.components.data_ingestion