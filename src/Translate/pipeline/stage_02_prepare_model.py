from Translate.components.build_model import Prepare_model
from Translate import logger
from Translate.entity.config_entity import Config_Data
from Translate.config.configuration import ConfigurationManager
from torchsummary import summary

STAGE_NAME = "Prepare model stage"

class PrepareModelPipeline:
    
    def __init__(self):
        pass

    def main(self,src_vocab_size,tgt_vocab_size):
        config_instance = ConfigurationManager()
        config_obj = config_instance.get_config()
        param_obj  = config_instance.get_training_config()
        prepare_obj=Prepare_model(config=config_obj,param=param_obj)
        model = prepare_obj.get_model(src_vocab_size,tgt_vocab_size)
        return model
        

if __name__ == '__main__':
    try:
        logger.info(f">>>>>> stage {STAGE_NAME} started <<<<<<")
        obj = PrepareModelPipeline()
        model = obj.main(2000,2000)
        # print(model)

        logger.info(f">>>>>> stage {STAGE_NAME} completed <<<<<<\n\nx==========x")
    except Exception as e:
        logger.exception(e)
        raise e