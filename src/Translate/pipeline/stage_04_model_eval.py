from Translate.constants import *
from Translate.utils.common import *
from Translate.config.configuration import ConfigurationManager
from Translate.components.model_eval_mlflow import Evaluation
from Translate import logger
import torch

STAGE_NAME = "Evaluation Pipeline"

class EvaluationPipeline:
    
    def __init__(self, val_dataloader, tokenizer_src, tokenizer_tgt,model):
        self.val_dataloader = val_dataloader
        self.tokenizer_src = tokenizer_src
        self.tokenizer_tgt = tokenizer_tgt
        self.model = model
    
    def main(self):
        os.environ["MLFLOW_TRACKING_URI"] = 'https://dagshub.com/nikhil0035/Machine_Translation_using_Transformers.mlflow'
        os.environ["MLFLOW_TRACKING_USERNAME"] = 'nikhil0035'
        os.environ["MLFLOW_TRACKING_PASSWORD"] = '8d6d437e619e46ffbd80ee64bee271c7f105427c'
        config_instance = ConfigurationManager()
        config_obj = config_instance.get_config()
        param_obj = config_instance.get_training_config()
        eval_obj = config_instance.get_evaluation_config()
        
        model_filename = latest_weights_file_path(config_obj)
        state = torch.load(model_filename)
        self.model.load_state_dict(state['model_state_dict'])

        Evaluation_obj =Evaluation(config=config_obj,params=param_obj,eval=eval_obj)
        Evaluation_obj.run_validation(self.model,self.val_dataloader, self.tokenizer_src, self.tokenizer_tgt)
        Evaluation_obj.log_into_mlflow()

if __name__ == '__main__':
    try:
        logger.info(f">>>>>> stage {STAGE_NAME} started <<<<<<")
        obj = EvaluationPipeline(val_dataloader, tokenizer_src, tokenizer_tgt,model)
        obj.main()
        # print(model)

        logger.info(f">>>>>> stage {STAGE_NAME} completed <<<<<<\n\nx==========x")
    except Exception as e:
        logger.exception(e)
        raise e
       