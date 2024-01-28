from Translate.constants import *
import os
from Translate.utils.common import *
from Translate.entity.config_entity import Config_Data


class ConfigurationManager:
    def __init__(
        self,
        config_filepath = CONFIG_FILE_PATH,
        params_filepath = PARAMS_FILE_PATH):

        self.config = read_yaml(config_filepath)
        self.params = read_yaml(params_filepath)

        create_directories([self.config.artifacts_root])
    

    def get_config(self) -> Config_Data:
        config = self.config.config_data

        # create_directories([config.root_dir])

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

