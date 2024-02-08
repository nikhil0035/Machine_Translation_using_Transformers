
from Translate.config.configuration import ConfigurationManager
from Translate.components.data_injestion import DataInjestion
from Translate.components.dataset import BilingualDataset
from Translate import logger

STAGE_NAME = "Data Ingestion stage"



class DataIngestionTrainingPipeline:
    
    def __init__(self):
        pass

    def main(self):
        config_instance = ConfigurationManager()
        config_obj = config_instance.get_config()
        param_obj = config_instance.get_training_config()
        datainjestion = DataInjestion(config=config_obj,param=param_obj,data_class=BilingualDataset)
        train_dataloader, val_dataloader, tokenizer_src, tokenizer_tgt= datainjestion.get_ds()
        return train_dataloader, val_dataloader, tokenizer_src, tokenizer_tgt

if __name__ == '__main__':
    try:
        logger.info(f">>>>>> stage {STAGE_NAME} started <<<<<<")
        obj = DataIngestionTrainingPipeline()
        obj.main()
        logger.info(f">>>>>> stage {STAGE_NAME} completed <<<<<<\n\nx==========x")
    except Exception as e:
        logger.exception(e)
        raise e