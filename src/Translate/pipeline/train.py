
from Translate.config.configuration import ConfigurationManager
from Translate.components.data_injestion import DataInjestion
from Translate.components.dataset import BilingualDataset


try:
    config_instance = ConfigurationManager()
    config_obj = config_instance.get_config()
    datainjestion = DataInjestion(config=config_obj,data_class=BilingualDataset)
    train_dataloader, val_dataloader, tokenizer_src, tokenizer_tgt= datainjestion.get_ds()
#     batch = next(iter(train_dataloader))

# # Print the batch or inspect its structure
#     print(batch['src_text'])
#     print(batch['tgt_text'])        
except Exception as E:
    pass