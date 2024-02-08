from Translate.constants import *
from Translate.utils.common import *
from Translate.config.configuration import ConfigurationManager
from Translate.components.model_train import train_model
from Translate import logger

STAGE_NAME = "Training Pipeline"



class TrainingPipeline:
    
    def __init__(self,train_dataloader, val_dataloader, tokenizer_src, tokenizer_tgt,model):
        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader
        self.tokenizer_src = tokenizer_src
        self.tokenizer_tgt = tokenizer_tgt
        self.model = model
    
    def main(self):
        config_instance = ConfigurationManager()
        config_obj = config_instance.get_config()
        param_obj = config_instance.get_training_config()
        train_obj = train_model(config_obj,param_obj,self.train_dataloader, self.val_dataloader, self.tokenizer_src, self.tokenizer_tgt,self.model)
        train_obj.train()


if __name__ == '__main__':
    try:
        logger.info(f">>>>>> stage {STAGE_NAME} started <<<<<<")
        obj = TrainingPipeline(train_dataloader, val_dataloader, tokenizer_src, tokenizer_tgt,model)
        obj.main()
        # print(model)

        logger.info(f">>>>>> stage {STAGE_NAME} completed <<<<<<\n\nx==========x")
    except Exception as e:
        logger.exception(e)
        raise e
       