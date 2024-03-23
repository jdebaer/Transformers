import torch
import torch.nn as nn

from torch.utils.data import Dataset, DataLoader, random_split

from datasets import load_dataset
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.trainers import WordLevelTrainer
from tokenizers.pre_tokenizers import Whitespace
from tqdm import tqdm
import warnings

from dataset import BilingualDataset, causal_mask
from model import build_transformer
from config import get_weights_file_path, get_config

from torch.utils.tensorboard import SummaryWriter

from pathlib import Path


def greedy_decode(model, encoder_input_tensor_batch, encoder_input_tensor_padding_mask_batch, src_tokenizer, tgt_tokenizer, max_len, device): # Fix these to be more representative 

    sos_id = src_tokenizer.token_to_id('[SOS]') # Either tokenizer can be used for this
    eos_id = src_tokenizer.token_to_id('[EOS]') # Either tokenizer can be used for this

    # Now get the encoder output and we're going to use it for every token we predict with the decoder, for the cross attention
    encoder_output_tensor_batch = model.encode(encoder_input_tensor_batch, encoder_input_tensor_padding_mask_batch) # dim (batch_size, seq_len, embed_size)

    # Now get the decoder started with just the [SOS] token
    decoder_input_tensor_batch = torch.empty(1,1).fill_(sos_id).type_as(encoder_input_tensor_batch).to(device) # dim (b_s, seq_len) but 1 and still 1 here since 1 token
    
    # Now we're going keep predicting the next token until we get [EOS] or until we reach max_len
    while True:
        if decoder_input_tensor_batch.size(1) == max_len: # dim 0 is the batch
            break
     
        # At each inference we need to provide an inference-specific causal mask, which shorten on each iteration
        decoder_input_tensor_causal_mask_batch = causal_mask(decoder_input_tensor_batch.size(1)).type_as(encoder_input_tensor_padding_mask_batch).to(device)
        
        # Do inference using the decoder, providing it the cross attention
        decoder_output_tensor_batch = model.decode(decoder_input_tensor_batch, decoder_input_tensor_causal_mask_batch, encoder_output_tensor_batch, encoder_input_tensor_padding_mask_batch)

        # Get the next token
       2:29:49 
        
        



#def run_validation(model, val_dataset, src_tokenizer, tgt_tokenizer, max_len, device, print_msg, global_state, writer, num_examples=2)
    # Note: val_dataset is not correct here, this has to be a batch_iterator wrapped around a data_loader or at least a data_loader which can provide batches

def run_validation(model, val_dataloader, src_tokenizer, tgt_tokenizer, max_len, device, print_msg, global_state, writer, num_examples=2)

    model.eval()
    count = 0
    source_texts = []
    expected = []
    predicted = []
      
    console_width = 80 # Size of the control window

    with torch.no_grad(): # Disable gradient calculation to make it faster

        for batch in val_dataloader: # Remember that for the validation dataloader we have set a batch size of 1
            count += 1
            
            encoder_input_tensor_batch = batch['encoder_input_tensor'].to(device) # dimension is (batch_size,seq_len)
            encoder_input_tensor_padding_mask_batch = batch['encoder_input_tensor_padding_mask'].to(device) # dimension is (batch_size,1,1,seq_len)

            assert encoder_input_tensor_batch.size(0) == 1, "Batch size for validation must be 1"

            decoder_input_tensor_batch = batch['decoder_input_tensor'].to(device) # dimension is (batch_size_seq_len)
            decoder_input_tensor_causal_mask_batch = batch['decoder_input_tensor_causal_mask'].to(device) # dimension is (batch_size,1,seq_len,seq_len)
        
    
    



def get_all_sentences(dataset, language):
    
    for item in dataset:
        yield item['translation'][language]


def get_or_build_tokenizer(config, dataset, language):

    tokenizer_path = Path(config.tokenizer_file.format(language))
    
    if not Path.exists(tokenizer_path):
        tokenizer = Tokenizer(WordLevel(unk_token='[UNK]'))
        tokenizer.pre_tokenizer = Whitespace()
        trainer = WordLevelTrainer(special_tokens=['UNK', 'PAD', 'SOS', 'EOS'], min_frequency=2)
        tokenizer.train_from_iterator(get_all_sentences(dataset, language), trainer=trainer)
        tokenizer.save(str(tokenizer_path))
    else:
        tokenizer = Tokenizer.from_file(str(tokinizer_path))

    return tokenizer

def get_dataset(config):
    
    dataset_raw = load_dataset('opus_books', f'{config.src_language}-{config.tgt_language}', split='train')

    src_tokenizer = get_or_build_tokenizer(config, dataset_raw, config.src_language)
    tgt_tokenizer = get_or_build_tokenizer(config, dataset_raw, config.tgt_language)

#    Split the training part of data set into a "real" training part and a validation part that you will use after each epoch

    train_dataset_size = int(0.9 * len(dataset_raw))
    val_dataset_size = len(dataset_raw) - train_dataset_size
    train_dataset_raw, val_dataset_raw = random.split(dataset_raw, [train_dataset_size, val_dataset_size])
    

#   Here we're converting strings to something we can actually train with i.e., tensors containing input_ids with padding and other special tokens etc.
    train_dataset = BilingualDataset(train_dataset_raw, src_tokenizer, tgt_tokenizer, config.src_language, config.tgt_language, config.seq_len)
    val_dataset = BilingualDataset(val_dataset_raw, src_tokenizer, tgt_tokenizer, config.src_language, config.tgt_language, config.seq_len)

#   New let's find out what the longest sequence is that we have in the dataset

    max_len_src = 0
    max_len_tgt = 0

    for item in dataset_raw:
        src_ids = src_tokenizer.encode(item['translation'][config.src_language]).ids
        tgt_ids = src_tokenizer.encode(item['translation'][config.tgt_language]).ids

        max_len_src = max(max_len_src, len(src_ids))
        max_len_tgt = max(max_len_tgt, len(tgt_ids))

    print(f'Maximum length of source sentences is {max_len_src}')
    print(f'Maximum length of target sentences is {max_len_tgt}')

#    Now create the data loaders to batch the training samples up

    train_dataloader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=1, shuffle=True) # For validation we're doing one by one, find out why this is

    return train_dataloader, val_dataloader, src_tokenizer, tgt_tokenizer

def get_model(config, src_vocab_size, tgt_vocab_size):

    model = build_transformer(src_vocab_size, tgt_vocab_size, config.seq_len, config.seq_len) # build_transformer can take different seq_lens if needed
    return model

#   If GPU is not big enough for this model then we can reduce number of heads and/or number of layers
        
def train_model(config):

# Define the device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device {device}')

#    It is necessary to have both the model, and the data on the same device, either CPU or GPU, for the model to process data. 

    Path(config['model_folder']).mkdir(parents=True, exist_ok=True)    
  
    train_dataloader, val_dataloader, src_tokenizer, tgt_tokenizer = get_dataset(config) 

    model = get_model(config, src_tokenizer.get_vocab_size(), tgt_tokenizer.get_vocab_size()).to(device) # Here we send the model to the device

# TensorBoard to visualize the loss/charts

    writer = SummaryWriter(config['experiment_name'])    

    optimizer = toch.optim.Adam(model.parameters(), lr=config['lr'], eps=1e-9)

#   We want to be able to resume the training if something crashes, based on model state and optimizer state (latter is optional but helps)

    initial_epoch = 0
    global_step = 0

    if config['preload']:
        model_filename = get_weights_file_path(config, config['preload'])
        print(f'Preloading model {model_filename}'
        state = torch.load(model_filename)
        initial_epoch = state['epoch'] + 1
        optimizer.load_state_dict(state['optimizer_state_dict'])
        global_step = state['global_step']
        
    loss_function = nn.CrossEntropyLoss(ignore_index=src_tokenizer.token_to_id('[PAD]'), label_smoothing=0.1).to(device)    
    # The label smoothing here takes 0.1% of the highest score and distributes it over the others - this makes the model more accurate
    # note: implement your own CEL: https://discuss.pytorch.org/t/cross-entropy-loss-clarification/103830/2

    # About the to(device) on the loss function:
    #The to() move happens to only parameters and buffers. Hence, moving a loss function like CrossEntropy to GPU doesn’t change anything. 
    #In a custom Loss function made subclassing nn.Module, the “.to()” will be inherited and will move any parameters/buffers to the gpu.
    #It depends on the inputs you are passing to this loss function. I.e. if the model output and targets are on the GPU, the computation 
    #will also be performed on the GPU. If your custom loss function has internal states stored as tensors you should move it to the same 
    # device before calculating the loss. If it’s stateless you can just pass the inputs to it to calculate the loss.

    for epoch in range(initial_epoch, config['num_epochs']:

        model.train()
        batch_iterator = tqdm(train_dataloader, desc=f'Processing epoch {epoch:02d}') # tqdm is what draws the nice progress bars, wrap it around DataLoader

        for batch in batch_iterator:

            # Below is how we push the data to the GPU as well (the model is already there)
            # If this stuff doesn't fit on the GPU mem then you get a runtime CUDA out of mem error
            # Note that these are batched returns of our __get_item()__ function in dataset

            encoder_input_tensor_batch = batch['encoder_input_tensor'].to(device) # dimension is (batch_size,seq_len)
            decoder_input_tensor_batch = batch['decoder_input_tensor'].to(device) # dimension is (batch_size_seq_len)
            encoder_input_tensor_padding_mask_batch = batch['encoder_input_tensor_padding_mask'].to(device) # dimension is (batch_size,1,1,seq_len)
            decoder_input_tensor_causal_mask_batch = batch['decoder_input_tensor_causal_mask'].to(device) # dimension is (batch_size,1,seq_len,seq_len)
            # decoder_label_tensor = batch['decoder_label_tensor'].to(device)
            
            # Now push the training data tensors through the transformer (the whole batch)

            encoder_output_tensor_batch = model.encode(encoder_input_tensor_batch, encoder_input_tensor_padding_mask_batch) # dim (batch_size, seq_len, embed_size)
            # The below matches 'decode(self, decoder_input_ids, decoder_mask, encoder_output, encoder_mask):' in transformer.py
            # Dim of the below is also (batch_size, seq_len, embed_size)
            decoder_output_tensor_batch = model.decode(decoder_input_tensor_batch, decoder_input_tensor_causal_mask_batch, encoder_output_tensor_batch, encoder_input_tensor_padding_mask_batch )
            transformer_output_tensor_batch = model.projectdecoder_output_tensor_batch) # dim is (batch_size, seq_len, tgt_vocab_size)
                   
            # Now let's compare these batched predictions with our batched labels
            # The label (below) is essentially the same sequence but the ids are shifted with one position (same amount of padding tokens) 
          
            decoder_label_tensor_batch = batch['decoder_label_tensor'].to(device) # dim (batch, seq_len)

            loss = loss_fn(transformer_output_tensor_batch.view(-1, tgt_tokenizer.get_vocab_size()), label.view(-1))

            # Couple of issues 
            # CrossEntropyLoss seems to apply log_softmax itself, and it seems to be recommended to NOT feed it pre-softmaxed inputs, which is however what we're doing
            # The first view transforms (b_s, seq_len, tgt_vocab_size) to (b_s * seq_len, tgt_vocab_size)
            # However label.view(-1) will do (b_s, seq_len) -> (b_s * seq_len) 
            # So while we have probs in the tgt_vocab_size dimension in the prediction (transformer_output_tensor_batch) we don't have 0/1 probs in the label - find out why

            batch_iterator.set_postfix({f"loss": f"{loss.item():6.3f}"}) # This is for tqdm visualization to show loss on progress bar
            writer.add_scaler('train_loss', loss.item(), global_step) # This is for TensorBoard
            writer.flush()

            # Backpropagate the loss
            loss.backward()

            # Update the weights (this is the job of the optimizer)
            optimizer.step()
            optimizer.zero_grad() # Zero out the gradient

            global_step += 1 # This is only used by TensorBoard to keep track of the loss

        # Save the model after every epoch in case of a crash
        # It's also really needed to save the optimizer as it keeps track of certain statistics, one for each weight, to understand how to optimize each weight independently 

        model_filename = get_weights_file_path(config, f'{epoch:02d}' # Name of the file contains epoch with zeros in front
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(), # This is all the weights of the model
            'optimizer_state_dict': optimizer.state_dict(),
            'global_step': global_step 


        }, model_filename)

if __name__ == '__main__':
    warnings.filterwarnings('ignore')
    config = get_config()
    train_model(config)


