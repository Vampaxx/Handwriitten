import os
import pandas as pd
from pathlib import Path
import tensorflow as tf
from src.handwritten.logger import logging
from src.handwritten.exception import CustomException
from src.handwritten.utils.common import (read_yaml,
                                          create_directories)

from src.handwritten.constants import *
from src.handwritten.entity.config_entity import (DataIngestionConfig,
                                                  PrepareBaseModelConfig)



class ConfugarationManager:

    def __init__(self,
                 config_file_path=CONFIG_FILE_PATH,
                 params_file_path=PARAMS_FILE_PATH):
        
        self.config_ = read_yaml(config_file_path)
        self.params_ = read_yaml(params_file_path)
        create_directories([self.config_.artifacts_root])


    def get_data_ingestion_config(self) -> DataIngestionConfig:

        config_ = self.config_.data_ingestion 
        create_directories([config_.root_dir])

        self.data_ingestion_config = DataIngestionConfig(
            train_data_path = Path(config_.train_data_path),
            test_data_path  = Path(config_.test_data_path),
            val_data_path   = Path(config_.val_data_path),
            raw_data_path   = Path(config_.raw_data_path),)
            
        return self.data_ingestion_config
    
    def get_prepare_base_model_config(self) -> PrepareBaseModelConfig:

        config_ = self.config_.prepare_base_model 
        create_directories([config_.root_dir])
        prepare_base_model_Config   = PrepareBaseModelConfig(
            root_dir                = Path (config_.root_dir),
            base_model_path         = Path (config_.base_model_path), 
            updated_base_model_path = Path (config_.updated_base_model_path),
            params_image_size       = list (self.params_.IMAGE_SIZE),
            params_learning_rate    = float(self.params_.LEARNING_RATE))
                
        return prepare_base_model_Config


if __name__ == "__main__":
    obj = ConfugarationManager()
    