import os
import sys
import pandas as pd 
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.datasets import fetch_openml

from src.handwritten.logger import logging
from src.handwritten.exception import CustomException
from src.handwritten.entity.config_entity import DataIngestionConfig
#from src.handwritten.config.configuration import ConfugarationManager



class DataIngestion:
    def __init__(self,config_:DataIngestionConfig):
        self.config_ = config_

    def data_ingestion_initialization(self):
        logging.info('Entered data ingestion method and components')
        try:
            mnist = fetch_openml('mnist_784', as_frame=False,cache=False)
            #feature extraction                
            logging.info(f'Data extraction from mnist_784')
            dependet_feature    = pd.DataFrame(data=mnist.data.astype('int64'),columns=mnist.feature_names)
            independent_feature = pd.DataFrame(data=mnist.target.astype('int64'),columns=mnist.target_names)
            data                = pd.concat([dependet_feature,independent_feature],axis=1)

            logging.info(f'Saving data into {self.config_.raw_data_path}')
            data.to_csv(self.config_.raw_data_path,index=False,header=True)
            logging.info(f'Saving data completed -- {self.config_.raw_data_path}')


            logging.info("train test split initiated")
            train_set,test_set = train_test_split(data,test_size=0.2,random_state=42,shuffle=True)
            test_set,val_set   = train_test_split(test_set,test_size=0.1,random_state=42,shuffle=True)

            train_set.to_csv(self.config_.train_data_path,index=False,header=True)
            test_set.to_csv (self.config_.test_data_path,index=False,header=True)
            val_set.to_csv  (self.config_.val_data_path,index=False,header=True)
            logging.info('Ingestion of data is completed')
            return (
                self.config_.train_data_path,
                self.config_.test_data_path,
                self.config_.val_data_path
            )



        except Exception as e:
            raise CustomException (e,sys)
    
    
        
if __name__ == "__main__":
    #config                      = ConfugarationManager()
    data_ingestion_config       = config.get_data_ingestion_config()
    split_data                  = DataIngestion(config_=data_ingestion_config)
    dataframe                   = split_data.data_ingestion_initialization()