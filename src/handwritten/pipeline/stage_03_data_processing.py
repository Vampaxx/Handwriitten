import sys
from src.handwritten.logger import logging
from src.handwritten.exception import CustomException
from src.handwritten.config.configuration import ConfugarationManager
from src.handwritten.components.data_procesing import DataProcessing


STAGE_NAME = "Data processing"

class PrepareDataProcessingPipeline:

    def __init__(self) -> None:
        pass

    def main(self,data_split:str):
        config                      = ConfugarationManager()
        data_ingestion_config       = config.get_data_processing_config()
        data_processing             = DataProcessing(config=data_ingestion_config)
        data_                       = data_processing.get_processing_data_path('train')
        data_1                      = data_processing.get_processing_pipeline()
        return data_1 

if __name__ == "__main__":
    try:
        logging.info('***************************************')
        logging.info(f">>>>>>>> stage {STAGE_NAME} started <<<<<<<<<")
        obj         = PrepareDataProcessingPipeline()
        obj.main(data_split='train')
        logging.info(f">>>>>>>> stage {STAGE_NAME} completed <<<<<<<<")
        logging.info('***************************************')
    except Exception as e:
        raise CustomException(e,sys)
        