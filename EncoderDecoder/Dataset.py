
# In this file we get the data in the Dataset format i.e. is a grouping of tensors that our model will use -> models needs training data in tensor format

import torch
import torch.nn as nn
from torch.utils.data import Dataset

class BilingualDataset(Dataset)

    def __init__(self, raw_dataset, src_tokenizer, tgt_tokenizer, src_language, tgt_language, seq_len) -> None:
        self().__init__()

        self.raw_dataset = raw_dataset
        self.src_tokenizer = src_tokenizer
        self.tgt_tokenizer = tgt_tokenizer
        self.src_language = src_language
        self.tgt_language = tgt_language
        self.seq_len = seq_len
 
#        Tensor with one element but has to be tensor because we're catting it below into a tensor 
        self.sos_token = torch.Tensor([tokenizer_src.token_to_id(['[SOS]'])], dtype=torch.int64) # 32-bit may not be enough for the input_id range
        self.eos_token = torch.Tensor([tokenizer_src.token_to_id(['[EOS]'])], dtype=torch.int64) # 32-bit may not be enough for the input_id range
        self.pad_token = torch.Tensor([tokenizer_src.token_to_id(['[PAD]'])], dtype=torch.int64) # 32-bit may not be enough for the input_id range
        
    def __len__(self):

        return len(self.raw_dataset)

#    __getitem__() must return one element we can train on, and then the data loader is going to call this to create training batches
    def __getitem__(self, index: Any) -> Any:

        src_target_pair = self.raw_dataset[index]
        src_text = src_target_pair['translation'][self.src_language]
        tgt_text = src_target_pair['translation'][self.tgt_language]

        encoder_input_ids = self.src_tokenizer.encode(src_text).ids # first part splits it into word-tokens, then .ids converts it to the integers
        decoder_input_ids = self.tgt_tokenizer.encode(tgt_text).ids 

        encoder_num_padding_tokens = seq_len - len(encoder_input_ids) - 2 # -2 because we'll also add [SOS] and [EOS]
        decoder_num_padding_tokens = seq_len - len(decoder_input_ids) - 1 # for the decoder we only add [SOS] (decoder label set otoh only has [EOS] )

        if encoder_num_padding_tokens < 0 or decoder_num_padding_tokens < 0:
            raise ValueError("seq_len is too short")

#        Now we need three tensors: one with input for the encoder, and two for input of the decoder (training and label)

        encoder_input_tensor = torch.cat(
            [
                self.sos_token,
                torch.tensor(encoder_input_ids, dtype=torch.int64),
                self.eos_token,                
                torch.tensor([self.pad_token]) * encoder_num_padding_tokens, dtype=torch.int64)
            ]
        )

        decoder_input_tensor = torch.cat(
            [
                self.sos_token,
                torch.tensor(decoder_input_ids, dtype=torch.int64),
                torch.tensor([self.pad_token]) * decoder_num_padding_tokens, dtype=torch.int64)
            ]
        )

        decoder_label_tensor = torch.cat(
            [
            toch.tensor(decoder_input_ids, dtype=torch.int64),
            self.eos_token,
            torch.tensor([self.pad_token]) * decoder_num_padding_tokens, dtype=torch.int64)
            ]
        )

        assert encoder_input_tensor.size(0) == self.seq_len
        assert decoder_input_tensor.size(0) == self.seq_len
        assert decoder_label_tensor.size(0) == self.seq_len

# Return as dictionary

        return {
            "encoder_input_tensor": encoder_input_tensor, # dimension is seq_len
            "decoder_input_tensor": decoder_input_tensor, # dimension is seq_len
            "encoder_input_tensor_padding_mask": (encoder_input_tensor != self.pad_token).unsqueeze(0).unsqueeze(0).int() # dimension is (1,1,seq_len)  
            "decoder_input_tensor_causal_mask": (decoder_input_tensor != self.pad_token).unsqueeze(0).unsqueeze(0).int() & causal_mask(decoder_input_tensor.size(0)), # dimensions are (1,1,seq_len) and (1,seq_len,seq_len) respectively
            "decoder_label_tensor": decoder_label_tensor, # dimension is seq_len
            "src_txt": src_txt,
            "tgt_txt": tgt_txt
        }


def causal_mask(dim):
  
    mask = torch.triu(torch.ones(1, dim, dim), diagonal=1).type(torch.int) # This one has zero diag and zeroes below, we need opposite hence:
    return mask == 0

 
