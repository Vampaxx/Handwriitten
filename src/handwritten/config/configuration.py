import os
import pandas as pd
from pathlib import Path
import tensorflow as tf
from src.handwritten.logger import logging
from src.handwritten.exception import CustomException
from src.handwritten.utils.common import (read_yaml,
                                          create_directories)

from src.handwritten.constants import *
from src.handwritten.components.data_procesing import DataProcessing
from src.handwritten.entity.config_entity import (DataIngestionConfig,
                                                  PrepareBaseModelConfig,
                                                  PreprocessingConfig,
                                                  TrainigConfig)



class ConfugarationManager:

    def __init__(self,
                 config_file_path=CONFIG_FILE_PATH,
                 params_file_path=PARAMS_FILE_PATH):
        
        self.config_ = read_yaml(config_file_path)
        self.params_ = read_yaml(params_file_path)
        create_directories([self.config_.artifacts_root])
        #load Datapreprocessing 
        self.data_processing = DataProcessing(config=self.get_data_processing_config())

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


    def get_data_processing_config(self) -> PreprocessingConfig:
        config_                     = self.config_.data_ingestion
        self.data_processing_config = PreprocessingConfig(
            train_data_path = Path(config_.train_data_path),
            test_data_path  = Path(config_.test_data_path), 
            val_data_path   = Path(config_.val_data_path),
            image_size      = list(self.params_.IMAGE_SIZE),
            buffer_size     = int(self.params_.BUFFER_SIZE),
            batch_size      = int(self.params_.BATCH_SIZE))
        return self.data_processing_config
    

    def get_training_config(self,dataset_type: str) -> TrainigConfig:
        self.dataset_type               = dataset_type
        training                        = self.config_.training 
        
        prepare_base_model              = self.config_.prepare_base_model
        params                          = self.params_
        training_x_data, training_y_data= self.data_processing.get_processing_data_path(self.dataset_type)
        create_directories([Path(training.root_dir)])
        
        training_config                 = TrainigConfig(
            root_dir                        = Path(training.root_dir),
            trained_model_path              = Path(training.trained_model_path),
            updated_base_model_path         = Path(prepare_base_model.updated_base_model_path),
            data_for_pipeline               = self.data_processing.get_processing_pipeline(),
            params_epochs                   = params.EPOCHS,
            params_batch_size               = params.BATCH_SIZE,
            params_image_size               = params.IMAGE_SIZE,)
        
        return training_config

if __name__ == "__main__":
    obj = ConfugarationManager()
    obj.get_training_config('test')
    