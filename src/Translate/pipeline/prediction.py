import torch
from tokenizers import Tokenizer
from Translate.utils.common import *
from Translate.components.build_model import Prepare_model
from Translate.config.configuration import ConfigurationManager

class PredictionPipeline:
    def __init__(self) -> None:
        pass
    def translate(self,sentence: str):
    
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print("Using device:", device)
       
        config_instance = ConfigurationManager()
        config = config_instance.get_config()
        param  = config_instance.get_training_config()

        tokenizer_src = Tokenizer.from_file(str(Path(config.tokenizer_file.format(config.lang_src))))
        tokenizer_tgt = Tokenizer.from_file(str(Path(config.tokenizer_file.format(config.lang_tgt))))
        
        prepare_obj=Prepare_model(config=config,param=param)
        model = prepare_obj.get_model(tokenizer_src.get_vocab_size(), tokenizer_tgt.get_vocab_size())
        
        # Load the pretrained weights
        model_filename = latest_weights_file_path(config)
        state = torch.load(model_filename)
        model.load_state_dict(state['model_state_dict'])

        model = model.to(device)
    
        # translate the sentence
        model.eval()
        with torch.no_grad():
            # Precompute the encoder output and reuse it for every generation step
            source = tokenizer_src.encode(sentence)
            source = torch.cat([
                torch.tensor([tokenizer_src.token_to_id('[SOS]')], dtype=torch.int64), 
                torch.tensor(source.ids, dtype=torch.int64),
                torch.tensor([tokenizer_src.token_to_id('[EOS]')], dtype=torch.int64),
                torch.tensor([tokenizer_src.token_to_id('[PAD]')] * (param.seq_len - len(source.ids) - 2), dtype=torch.int64)
            ], dim=0).to(device)
            source_mask = (source != tokenizer_src.token_to_id('[PAD]')).unsqueeze(0).unsqueeze(0).int().to(device)
            encoder_output = model.encode(source, source_mask)

            # Initialize the decoder input with the sos token
            decoder_input = torch.empty(1, 1).fill_(tokenizer_tgt.token_to_id('[SOS]')).type_as(source).to(device)

            # Generate the translation word by word
            while decoder_input.size(1) < param.seq_len:
                # build mask for target and calculate output
                decoder_mask = torch.triu(torch.ones((1, decoder_input.size(1), decoder_input.size(1))), diagonal=1).type(torch.int).type_as(source_mask).to(device)
                out = model.decode(encoder_output, source_mask, decoder_input, decoder_mask)

                # project next token
                prob = model.project(out[:, -1])
                _, next_word = torch.max(prob, dim=1)
                decoder_input = torch.cat([decoder_input, torch.empty(1, 1).type_as(source).fill_(next_word.item()).to(device)], dim=1)

                # print the translated word
                # print(f"{tokenizer_tgt.decode([next_word.item()])}", end=' ')

                # break if we predict the end of sentence token
                if next_word == tokenizer_tgt.token_to_id('[EOS]'):
                    break
        
        # print(tokenizer_tgt.decode(decoder_input[0].tolist()))
        # convert ids to tokens
        return tokenizer_tgt.decode(decoder_input[0].tolist())


if __name__ == '__main__':
    try:
       
        obj = PredictionPipeline()
        obj.translate('What is happening?')
        

    
    except Exception as e:
        logger.exception(e)
        raise e
       