from src.Translate.components.dataset import BilingualDataset,causal_mask
from src.Translate.components.build_model import Prepare_model
from src.Translate.utils.common import *
from src.Translate.entity.config_entity import *

import torchtext.datasets as datasets
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split
from torch.optim.lr_scheduler import LambdaLR
import mlflow

from urllib.parse import urlparse
import torchmetrics

import warnings
from tqdm import tqdm
import os
from pathlib import Path

class Evaluation:
    def __init__(self,config:Config_Data,params:TrainingConfig,eval:EvaluationConfig) -> None:
        self.config = config
        self.params = params
        self.eval = eval
    
    @staticmethod
    def greedy_decode(model, source, source_mask, tokenizer_src, tokenizer_tgt, max_len, device):
        sos_idx = tokenizer_tgt.token_to_id('[SOS]')
        eos_idx = tokenizer_tgt.token_to_id('[EOS]')

        # Precompute the encoder output and reuse it for every step
        encoder_output = model.encode(source, source_mask)
        # Initialize the decoder input with the sos token
        decoder_input = torch.empty(1, 1).fill_(sos_idx).type_as(source).to(device)
        while True:
            if decoder_input.size(1) == max_len:
                break

            # build mask for target
            decoder_mask = causal_mask(decoder_input.size(1)).type_as(source_mask).to(device)

            # calculate output
            out = model.decode(encoder_output, source_mask, decoder_input, decoder_mask)

            # get next token
            prob = model.project(out[:, -1])
            _, next_word = torch.max(prob, dim=1)
            decoder_input = torch.cat(
                [decoder_input, torch.empty(1, 1).type_as(source).fill_(next_word.item()).to(device)], dim=1
            )

            if next_word == eos_idx:
                break

        return decoder_input.squeeze(0)
    
    def run_validation(self, model, validation_ds, tokenizer_src, tokenizer_tgt):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print("Using device:", device)
        self.model = model
        model = self.model.to(device)
        model.eval()
        count = 0
        max_len = self.config.seq_len
        source_texts = []
        expected = []
        predicted = []


        with torch.no_grad():
            batch_iterator = tqdm(validation_ds)
            for batch in batch_iterator:
                count += 1
                encoder_input = batch["encoder_input"].to(device) # (b, seq_len)
                encoder_mask = batch["encoder_mask"].to(device) # (b, 1, 1, seq_len)

                # check that the batch size is 1
                assert encoder_input.size(
                    0) == 1, "Batch size must be 1 for validation"

                model_out = self.greedy_decode(model, encoder_input, encoder_mask, tokenizer_src, tokenizer_tgt, max_len, device)

                source_text = batch["src_text"][0]
                target_text = batch["tgt_text"][0]
                model_out_text = tokenizer_tgt.decode(model_out.detach().cpu().numpy())

                source_texts.append(source_text)
                expected.append(target_text)
                predicted.append(model_out_text)

                if count ==5:
                    break

        metric = torchmetrics.CharErrorRate()
        self.cer = metric(predicted, expected)
       

        # Compute the word error rate
        metric = torchmetrics.WordErrorRate()
        self.wer = metric(predicted, expected)
      

        # Compute the BLEU metric
        metric = torchmetrics.BLEUScore()
        self.bleu = metric(predicted, expected)

    def log_into_mlflow(self):
        mlflow.set_registry_uri(self.eval.mlflow_uri)
        tracking_url_type_store = urlparse(mlflow.get_tracking_uri()).scheme

        with mlflow.start_run():
            mlflow.log_params(self.eval.all_params)

            mlflow.log_metrics(
                {"CER": self.cer, "WER": self.wer , "Bleu":self.bleu}
            )

            if tracking_url_type_store != "file":

                # Register the model
                # There are other ways to use the Model Registry, which depends on the use case,
                # please refer to the doc for more information:
                # https://mlflow.org/docs/latest/model-registry.html#api-workflow
                mlflow.pytorch.log_model(self.model, "model", registered_model_name="NMT_Transformers")
            else:
                mlflow.pytorch.log_model(self.model, "model")