from Translate.constants import *
import os
from Translate.utils.common import *
from Translate.entity.config_entity import Config_Data,TrainingConfig,EvaluationConfig

class ConfigurationManager:
    def __init__(
        self,
        config_filepath = CONFIG_FILE_PATH,
        params_filepath = PARAMS_FILE_PATH):

        self.config = read_yaml(config_filepath)
        self.params = read_yaml(params_filepath)

        create_directories([self.config.artifacts_root])
    
    def get_training_config(self)->TrainingConfig:

        params = self.params

        training_config = TrainingConfig(

            batch_size = params.batch_size,
            num_epochs = params.num_epochs,
            lr = params.lr,
            seq_len =  params.seq_len,
            d_model = params.d_model,
        )

        return training_config

    
    def get_config(self) -> Config_Data:
        config = self.config.config_data


        data_ingestion_config = Config_Data(
            
            batch_size = config.batch_size,
            num_epochs = config.num_epochs,
            lr = config.lr,
            seq_len =  config.seq_len,
            d_model = config.d_model,
            datasource = config.datasource,
            lang_src = config.lang_src,
            lang_tgt = config.lang_tgt,
            model_folder = config.model_folder,
            model_basename = config.model_basename,
            preload = config.preload,
            tokenizer_file = config.tokenizer_file,
            experiment_name = config.experiment_name,
        )

        return data_ingestion_config
    
      
    def get_evaluation_config(self) -> EvaluationConfig:
        eval_config = EvaluationConfig(
            mlflow_uri="https://dagshub.com/nikhil0035/Machine_Translation_using_Transformers.mlflow",
            all_params=self.params
           
        )
        return eval_config

