import torch
import torch.nn as nn
from torch.utils.data import Dataset

class BilingualDataset(Dataset):

    # This class is very specific to how the raw sequences are made available.
    # Via __get_item__(), this class should return tensors for the src and tgt sequences, and then also for the label squence (shifted tgt sequence).
    # These tensors should include input ids.
    # We also return: 
    # - The tokenizers that we used (received) to do all this, and also the src and tgt sequences in text format.
    # - The src and tgr sentences in raw text format
    # - The needed masks: a padding mask for the encoder, and a combined padding/causal mask for the decoder.
    # Note: batching is not done here, that's done in train.py via a data loader that we're wrapping around this class (object).

    def __init__(self, raw_dataset, src_tokenizer, tgt_tokenizer, src_language, tgt_language, seq_len):
        #self().__init__() # This is an abstract class.

        # seq_len here is the same for src/tgt - does not have to be the case, transformer can handle different lens.

        self.raw_dataset = raw_dataset
        self.src_tokenizer = src_tokenizer
        self.tgt_tokenizer = tgt_tokenizer
        self.src_language = src_language
        self.tgt_language = tgt_language
        self.seq_len = seq_len
 
        # Tensors with one element that we're going to cat to other tensors later on. We could use the tgt tokenizer for this as well.
        #self.sos_token = torch.Tensor([src_tokenizer.token_to_id(['[SOS]'])], dtype=torch.int64) # 32-bit may not be enough for the input_id range.
        #self.eos_token = torch.Tensor([src_tokenizer.token_to_id(['[EOS]'])], dtype=torch.int64) # 32-bit may not be enough for the input_id range.
        #self.pad_token = torch.Tensor([src_tokenizer.token_to_id(['[PAD]'])], dtype=torch.int64) # 32-bit may not be enough for the input_id range.
        self.sos_token = torch.tensor([src_tokenizer.token_to_id('[SOS]')], dtype=torch.int64) # 32-bit may not be enough for the input_id range.
        self.eos_token = torch.tensor([src_tokenizer.token_to_id('[EOS]')], dtype=torch.int64) # 32-bit may not be enough for the input_id range.
        self.pad_token = torch.tensor([src_tokenizer.token_to_id('[PAD]')], dtype=torch.int64) # 32-bit may not be enough for the input_id range.
        
    def __len__(self):

        return len(self.raw_dataset)

#    __getitem__() must return one element we can train on, and then the data loader is going to call this to create training batches
    def __getitem__(self, index):

        src_target_pair = self.raw_dataset[index]
        #src_text = src_target_pair['translation'][self.src_language]
        #tgt_text = src_target_pair['translation'][self.tgt_language]
        src_text = src_target_pair[self.src_language]
        tgt_text = src_target_pair[self.tgt_language]

        encoder_input_ids = self.src_tokenizer.encode(src_text).ids # First part splits text into tokens, .ids converts these to the integers.
        decoder_input_ids = self.tgt_tokenizer.encode(tgt_text).ids 

        encoder_num_padding_tokens = self.seq_len - len(encoder_input_ids) - 2 # -2 because we'll also add [SOS] and [EOS].
        decoder_num_padding_tokens = self.seq_len - len(decoder_input_ids) - 1 # For the decoder we only add [SOS] (decoder label only has [EOS] (shifted)).

        if encoder_num_padding_tokens < 0 or decoder_num_padding_tokens < 0:
            raise ValueError("seq_len is too short")

        encoder_input_tensor = torch.cat(
            [
                self.sos_token,
                torch.tensor(encoder_input_ids, dtype=torch.int64),
                self.eos_token,                
                torch.tensor([self.pad_token] * encoder_num_padding_tokens, dtype=torch.int64)
            ]
        )

        decoder_input_tensor = torch.cat(
            [
                self.sos_token,
                torch.tensor(decoder_input_ids, dtype=torch.int64),
                torch.tensor([self.pad_token] * decoder_num_padding_tokens, dtype=torch.int64)
            ]
        )

        decoder_label_tensor = torch.cat(
            [
            torch.tensor(decoder_input_ids, dtype=torch.int64),
            self.eos_token,
            torch.tensor([self.pad_token] * decoder_num_padding_tokens, dtype=torch.int64)
            ]
        )

        assert encoder_input_tensor.size(0) == self.seq_len, "Encoder input tensor does not have size seq_len."
        assert decoder_input_tensor.size(0) == self.seq_len, "Decoder input tensor does not have size seq_len."
        assert decoder_label_tensor.size(0) == self.seq_len, "Decoder label tensor does not have size seq_len."

        return {
            "encoder_input_tensor": encoder_input_tensor, # dimension is seq_len
            "decoder_input_tensor": decoder_input_tensor, # dimension is seq_len
            # Padding mask is the same for each row in the final matrix of the dot product.
            "encoder_input_tensor_mask": (encoder_input_tensor != self.pad_token).unsqueeze(0).unsqueeze(0).int(), # dimension is (1,1,seq_len)  
            # Causal mask is diagonal. We also want to ignore the paddings in the decoder though.
            # Dimensions are (1,1,seq_len) and (1,seq_len,seq_len) respectively.
            # 
            # Example of how this works: 
            # Let's say decoder_input_tensor is (-100 is padding id) [2,2,2,-100],
            # that means the first part becomes [[[1,1,1,0]]], 
            # then causal_mask will produce 1,0,0,0 which the '&' turns into 1,0,0,0
            #                               1,1,0,0                          1,1,0,0
            #                               1,1,1,0                          1,1,1,0
            #                               1,1,1,1                          1,1,1,0 <<< column 4 is set to zeros as these are padding tokens
            #
            "decoder_input_tensor_mask": (decoder_input_tensor != self.pad_token).unsqueeze(0).unsqueeze(0).int() & causal_mask(decoder_input_tensor.size(0)), 
            "decoder_label_tensor": decoder_label_tensor, # dimension is seq_len
            "src_text": src_text,
            "tgt_text": tgt_text
        }


def causal_mask(dim):
  
    mask = torch.triu(torch.ones(1, dim, dim), diagonal=1).type(torch.int) # This one has zero diag and zeroes below, we need opposite hence:
    return (mask == 0).int()

 
