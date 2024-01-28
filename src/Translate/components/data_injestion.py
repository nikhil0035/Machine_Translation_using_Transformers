import os
from Translate import logger
from Translate.entity.config_entity import Config_Data
from Translate.components.dataset import BilingualDataset

from datasets import load_dataset
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.trainers import WordLevelTrainer
from tokenizers.pre_tokenizers import Whitespace

from torch.utils.data import Dataset, DataLoader, random_split


import warnings
from tqdm import tqdm
from pathlib import Path

class DataInjestion():
    def __init__(self,config: Config_Data,data_class):
        self.config = config
        self.BilingualDataset = data_class
    
    @staticmethod
    def get_all_sentences(ds, lang):
        for item in ds:
            yield item['translation'][lang]
    
    @staticmethod
    def get_or_build_tokenizer(config,ds, lang):
        tokenizer_path = Path(config.tokenizer_file.format(lang))
        if not Path.exists(tokenizer_path):

            tokenizer = Tokenizer(WordLevel(unk_token="[UNK]"))
            tokenizer.pre_tokenizer = Whitespace()
            trainer = WordLevelTrainer(special_tokens=["[UNK]", "[PAD]", "[SOS]", "[EOS]"], min_frequency=2)
            tokenizer.train_from_iterator(DataInjestion.get_all_sentences(ds, lang), trainer=trainer)
            tokenizer.save(str(tokenizer_path))
        
        else:
            tokenizer = Tokenizer.from_file(str(tokenizer_path))
        return tokenizer
    
    
    def get_ds(self):
        ds_raw = load_dataset(f"{self.config.datasource}", f"{self.config.lang_src}-{self.config.lang_tgt}", split='train')

        tokenizer_src = self.get_or_build_tokenizer(self.config, ds_raw, self.config.lang_src)
        tokenizer_tgt = self.get_or_build_tokenizer(self.config, ds_raw, self.config.lang_tgt)

        # tokenizer_src = self.get_or_build_tokenizer(ds_raw, self.config.lang_src)
        # tokenizer_tgt = self.get_or_build_tokenizer(ds_raw, self.config.lang_tgt)

        train_ds_size = int(0.9 * len(ds_raw))
        val_ds_size = len(ds_raw) - train_ds_size
        train_ds_raw, val_ds_raw = random_split(ds_raw, [train_ds_size, val_ds_size])

        train_ds = BilingualDataset(train_ds_raw, tokenizer_src, tokenizer_tgt, self.config.lang_src, self.config.lang_tgt, self.config.seq_len)
        val_ds = BilingualDataset(val_ds_raw, tokenizer_src, tokenizer_tgt,  self.config.lang_src, self.config.lang_tgt, self.config.seq_len)

        max_len_src = 0
        max_len_tgt = 0

        for item in ds_raw:
            src_ids = tokenizer_src.encode(item['translation'][self.config.lang_src]).ids
            tgt_ids = tokenizer_tgt.encode(item['translation'][self.config.lang_tgt]).ids
            max_len_src = max(max_len_src, len(src_ids))
            max_len_tgt = max(max_len_tgt, len(tgt_ids))

        # print(f'Max length of source sentence: {max_len_src}')
        # print(f'Max length of target sentence: {max_len_tgt}')
        logger.info(f'Max length of source sentence: {max_len_src}')
        logger.info(f'Max length of target sentence: {max_len_tgt}')

        train_dataloader = DataLoader(train_ds, batch_size=self.config.batch_size, shuffle=True)
        val_dataloader = DataLoader(val_ds, batch_size=1, shuffle=True)

        return train_dataloader, val_dataloader, tokenizer_src, tokenizer_tgt