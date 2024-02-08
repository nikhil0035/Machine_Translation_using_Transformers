import torchtext.datasets as datasets
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split
from torch.optim.lr_scheduler import LambdaLR

import os
from pathlib import Path

from Translate.constants import *
from Translate.utils.common import *
from Translate.entity.config_entity import Config_Data


class train_model():
    def __init__(self,config,param,train_dataloader, val_dataloader, tokenizer_src, tokenizer_tgt, model):
        self.param=param
        self.config=config
        self.train_dataloader=train_dataloader
        self.val_dataloader=val_dataloader
        self.tokenizer_src=tokenizer_src
        self.tokenizer_tgt=tokenizer_tgt
        self.model = model
    
    def train(self):
        device = "cuda" if torch.cuda.is_available() else "mps" if torch.has_mps or torch.backends.mps.is_available() else "cpu"
        print("Using device:", device)
        if (device == 'cuda'):
            print(f"Device name: {torch.cuda.get_device_name(device.index)}")
            print(f"Device memory: {torch.cuda.get_device_properties(device.index).total_memory / 1024 ** 3} GB")
        elif (device == 'mps'):
            print(f"Device name: <mps>")
        else:
            print("NOTE: If you have a GPU, consider using it for training.")
            print("      On a Windows machine with NVidia GPU, check this video: https://www.youtube.com/watch?v=GMSjDTU8Zlc")
            print("      On a Mac machine, run: pip3 install --pre torch torchvision torchaudio torchtext --index-url https://download.pytorch.org/whl/nightly/cpu")
        device = torch.device(device)

        Path(f"{self.config.datasource}_{self.config.model_folder}").mkdir(parents=True, exist_ok=True)

        model = self.model.to(device)
        
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001, eps=1e-9)
        
        initial_epoch = 0
        global_step = 0
        preload = self.config.preload

        model_filename = latest_weights_file_path(self.config) if preload == 'latest' else get_weights_file_path(self.config, preload) if preload else None

        if model_filename:
            print(f'Preloading model {model_filename}')
            state = torch.load(model_filename)
            model.load_state_dict(state['model_state_dict'])
            initial_epoch = state['epoch'] + 1
            optimizer.load_state_dict(state['optimizer_state_dict'])
            global_step = state['global_step']
        else:
            print('No model to preload, starting from scratch')
        
        loss_fn = nn.CrossEntropyLoss(ignore_index=tokenizer_src.token_to_id('[PAD]'), label_smoothing=0.1).to(device)

        for epoch in range(initial_epoch, param.num_epochs):
            torch.cuda.empty_cache()
            model.train()
            batch_iterator = tqdm(train_dataloader, desc=f"Processing Epoch {epoch:02d}")

            for batch in batch_iterator:

                encoder_input = batch['encoder_input'].to(device) # (b, seq_len)
                decoder_input = batch['decoder_input'].to(device) # (B, seq_len)
                encoder_mask = batch['encoder_mask'].to(device) # (B, 1, 1, seq_len)
                decoder_mask = batch['decoder_mask'].to(device) # (B, 1, seq_len, seq_len)

                # Run the tensors through the encoder, decoder and the projection layer
                encoder_output = model.encode(encoder_input, encoder_mask) # (B, seq_len, d_model)
                decoder_output = model.decode(encoder_output, encoder_mask, decoder_input, decoder_mask) # (B, seq_len, d_model)
                proj_output = model.project(decoder_output) # (B, seq_len, vocab_size)

                # Compare the output with the label
                label = batch['label'].to(device) # (B, seq_len)

                # Compute the loss using a simple cross entropy
                loss = loss_fn(proj_output.view(-1, tokenizer_tgt.get_vocab_size()), label.view(-1))
                batch_iterator.set_postfix({"loss": f"{loss.item():6.3f}"})

                # # Log the loss
                # writer.add_scalar('train loss', loss.item(), global_step)
                # writer.flush()

                # Backpropagate the loss
                loss.backward()

                # Update the weights
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)

                global_step += 1

            model_filename = get_weights_file_path(config, f"{epoch:02d}")
            torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'global_step': global_step
            }, model_filename)