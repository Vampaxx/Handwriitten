import os
import sys
import tensorflow as tf
from pathlib import Path
from tensorflow.keras.models import load_model

from src.handwritten.logger import logging
from src.handwritten.exception import CustomException
from src.handwritten.utils.model import model_
from src.handwritten.utils.common import save_model

from src.handwritten.entity.config_entity import PrepareBaseModelConfig
from src.handwritten.config.configuration import ConfugarationManager

class PrepareBaseModel:
    try:
        def __init__(self, config: PrepareBaseModelConfig):
            self.config = config

        def get_base_model_and_updated_model(self):
            logging.info('Base model building initialized')
            self.model = model_()

            save_model(path=self.config.base_model_path, model=self.model) 
            logging.info(f'Base model saved on {self.config.base_model_path}')

            #load the model
            self.loaded_model = load_model(self.config.base_model_path)
            logging.info('compiling started')
            self.loaded_model.compile(optimizer='adam',
                                    metrics=['accuracy'],
                                    loss='sparse_categorical_crossentropy')
            save_model(path=self.config.updated_base_model_path, model=self.loaded_model) 
            logging.info(f'updated base model saved on {self.config.updated_base_model_path}')   
    except Exception as e:
        raise CustomException(e,sys)
    

if __name__ == "__main__":
    try:
        config = ConfugarationManager()  # Corrected the class name
        path = config.get_prepare_base_model_config()
        obj = PrepareBaseModel(config=path)
        obj.get_base_model_and_updated_model()
    except Exception as e:
        raise CustomException(e,sys)
