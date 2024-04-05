import torch
import torch.nn as nn
import warnings

from torch.utils.data import Dataset, DataLoader, random_split
from datasets import load_dataset
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.trainers import WordLevelTrainer
from tokenizers.pre_tokenizers import Whitespace
from tqdm import tqdm
from dataset import BilingualDataset, causal_mask
from transformer import build_transformer
from config import get_config
from torch.utils.tensorboard import SummaryWriter
from pathlib import Path

def get_model_file_path(config, epoch: str):
    model_folder = config['model_folder']
    model_basename = config['model_basename']
    model_filename = f'{model_basename}{epoch}.pt'
    return str(Path('.') / model_folder / model_filename)

##########################
####### VALIDATION #######


def greedy_decode(model, encoder_input_tensor_batch, encoder_input_tensor_mask_batch, src_tokenizer, tgt_tokenizer, max_len, device): 

    # Note: would not work on a batch with more than 1 items in it. Need to rework this for validation better than PoC if we really want to batch.

    sos_id = src_tokenizer.token_to_id('[SOS]') 							# Either tokenizer can be used for this.
    eos_id = src_tokenizer.token_to_id('[EOS]') 							# Either tokenizer can be used for this.

    # Generate the encoder output. We'll use it in every decoder invocation where we predict the next id.
    encoder_output_tensor_batch = model.encode(encoder_input_tensor_batch, encoder_input_tensor_mask_batch) # Dim (1, seq_len, embed_size).

    # Now get the decoder started with just the [SOS] token.
    decoder_input_tensor_batch = torch.empty(1,1).fill_(sos_id).type_as(encoder_input_tensor_batch).to(device) # Dim at this point is (1,1).
    
    # Now use the decoder with cross attention to keep predicting the next id until we get the '[EOS]' id or until we reach max_len (which is seq_len).
    while True:
        if decoder_input_tensor_batch.size(1) == max_len: 						# We filled it up with the previous iteration.
            break											# Note no '[EOS]' id in this case.
     
        # You might ask why we need a causal mask, since we start predicting from '[SOS]'.
        # This is because on every iteration, our model predicts the next id for EVERY id that we provide as input, not just for the last id.
        # As we will see, we ignore all these 'previous words' predictions, so an open question is: do we actually need to causal mask.
        # Note that there is for sure no padding going on, so we don't need to mask for padding in the decoder.
        decoder_input_tensor_mask_batch = causal_mask(decoder_input_tensor_batch.size(1)).type_as(encoder_input_tensor_mask_batch).to(device)
        
        # Do inference using the decoder, while providing it with the cross attention of dim (batch_size, seq_len) which here is (1, seq_len).
        # Dim is (batch_size, <how many ids we have so far in the sequence>, embed_size) with batch_size being 1 here.
        decoder_output_tensor_batch = model.decode(decoder_input_tensor_batch, decoder_input_tensor_mask_batch, encoder_output_tensor_batch, encoder_input_tensor_mask_batch) 

        # Now we feed ONLY THE LAST ([:,-1]) context vector in the predicted sequence so far to the projection layer to predict the next id.
        # What we feed in has dim (1, embed_size) so the last context vector which has all the context for the previous ids.
        # Dim of probabilities consequently is (1, tgt_vocab_size) .
        probabilities = model.project(decoder_output_tensor_batch[:,-1]) 			

        # Via the probabilities we select the next id by choosing the one with the highest probability (greedy).
        _, next_id = torch.max(probabilities, dim=1) 

        # Now we need to append the next id to decoder_input_tensor_batch that we created above (in the seq dimension).
        # item() converts tensor with one element to a standard number (not a tensor).
        # On the first run our dim goes from (1,1) to (1,2) and we keep growing in that dimension.
        decoder_input_tensor_batch = torch.cat(
		[decoder_input_tensor_batch, torch.empty(1,1).fill_(next_id.item()).type_as(encoder_input_tensor_batch).to(device)],
	dim=1) # dim is the dimension in which we do the concat
    
        if next_id == eos_id:
            break

    return decoder_input_tensor_batch.squeeze(0) # Remove the batch dimension so we end up with a tensor containing one dimension of ids.

def run_validation(model, valid_dataloader, src_tokenizer, tgt_tokenizer, seq_len, device, print_msg, global_state, writer, num_examples=1):

    print("-------------------------------------------")
    print("Running validation")
    print("-------------------------------------------")

    # Currently this is a PoC validation, where we print out the target sentence as well as the predicted sentence.

    model.eval()											# Put model in eval mode.
    count = 0
    src_sentence_tb = []
    tgt_sentence_tb = []
    prd_sentence_tb = []
      
    console_width = 80 											# Size of the control window.

    with torch.no_grad(): 										# Disable gradient calculation to speed up validation.

        for batch in valid_dataloader: 									# Validation batch size is currently 1.
            count += 1
            
            encoder_input_tensor_batch = batch['encoder_input_tensor'].to(device) 			# Dim is (batch_size,seq_len).
            encoder_input_tensor_mask_batch = batch['encoder_input_tensor_mask'].to(device) 		# Dim is (batch_size,1,1,seq_len).

            assert encoder_input_tensor_batch.size(0) == 1, "Batch size for validation must be 1."

            # We don't need to pass anything on for the decoder, since we're going to start decoding with '[SOS]'.
            # The output is a (batch_size, seq_len) (or shorter than seq_len) where the last element in dim seq_len is '[EOS]'.
            # transformer_infer only has one dimension, the size of which is the amount of generated ids.
            transformer_infer = greedy_decode(model, encoder_input_tensor_batch, encoder_input_tensor_mask_batch, src_tokenizer, tgt_tokenizer, seq_len, device)
            print("*********** Inferred ids ***********")
            print(transformer_infer)
            print("************************************")

            src_sentence = batch['src_text'][0]
            tgt_sentence = batch['tgt_text'][0]

            # The model, it's inputs and the model parameters should all be on the same device, ideally CUDA. The ouputs as well, if they are going to interact with the model again
            # which is typically the case while inferencing a transformer, as we use the output again as input. As soon as we have our final inferencing though, then we have output
            # that no longer will need to interact with the model. This means we can use detach() -> returns a tensor detached from the graph and then cpu() to move it to the CPU.
            # We should always do this ASAP whenever we end up with a tensor that won't need to interact with the model anymore, so save RAM.

            transformer_infer_sentence = tgt_tokenizer.decode(transformer_infer.detach().cpu().numpy()) 

            # Lists are for TensorBoard: to do.
            # src_sentence_tb.append(src_sentence)
            # tgt_sentence_tb.append(tgt_sentence)
            # prd_sentence_tb.append(transformer_infer_sentence)

            # Don't use the regular print function as it will mess up tqdm.
            print_msg('-' * console_width)
            print_msg(f'SOURCE: {src_sentence}')
            print_msg(f'TARGET: {tgt_sentence}')
            print_msg(f'PREDICTED: {transformer_infer_sentence}')

            if count == num_examples:
                break

#    if writer: 	# To do: TensorBoard. 
#        # To do: TorchMetrics add this CharErrorRate, BLEU, WordErrorRate
        
    
####### VALIDATION #######
##########################
    
##########################
######## TRAINING ########

def get_all_sentences(dataset, language):
    
    for item in dataset:
        #yield item['translation'][language]
        yield item[language]

def get_or_build_tokenizer(config, dataset, language):

    tokenizer_path = Path(config['tokenizer_file'].format(language))
    
    if not Path.exists(tokenizer_path):
        print("We create a tokenizer from scratch.")
        tokenizer = Tokenizer(WordLevel(unk_token='[UNK]'))
        tokenizer.pre_tokenizer = Whitespace()
        trainer = WordLevelTrainer(special_tokens=['[UNK]', '[PAD]', '[SOS]', '[EOS]'], min_frequency=2)
        tokenizer.train_from_iterator(get_all_sentences(dataset, language), trainer=trainer)
        tokenizer.save(str(tokenizer_path))
    else:
        tokenizer = Tokenizer.from_file(str(tokenizer_path))

    return tokenizer

def get_dataloader(config):

    # This function eventually wants to set up BilingualDataset objects. Let's first stage all we need for that.
    
    # 1. We need the raw dataset. We use load_dataset from Hugging Face's datasets for that.
    if config['edu']:
        print(config['src_language'])
        print(config['tgt_language'])
    # dataset_raw = load_dataset('opus_books', f"{config['src_language']}-{config['tgt_language']}", split='train', streaming=True)
    # dataset_raw = load_dataset("opus_books", "en-fr", split='train', streaming=True)
    dataset_raw = load_dataset('json', data_files='en-fr.tiny', split='train')

    # 2. We also need tokenizers, one for each raw dataset language subset.
    src_tokenizer = get_or_build_tokenizer(config, dataset_raw, config['src_language'])
    tgt_tokenizer = get_or_build_tokenizer(config, dataset_raw, config['tgt_language'])

    # 3. We split here. Could also do this after we got our BilingualDataset objects.
    train_dataset_size = int(0.9 * len(dataset_raw))
    valid_dataset_size = len(dataset_raw) - train_dataset_size
    train_dataset_raw, valid_dataset_raw = random_split(dataset_raw, [train_dataset_size, valid_dataset_size])
    
    # 4. We need a seq_len for both the encoder and the decoder
    src_seq_len = 0
    tgt_seq_len = 0
    seq_len = 0

    for item in dataset_raw:
        #src_ids = src_tokenizer.encode(item['translation'][config['src_language']]).ids
        #tgt_ids = src_tokenizer.encode(item['translation'][config['tgt_language']]).ids
        src_ids = src_tokenizer.encode(item[config['src_language']]).ids
        tgt_ids = src_tokenizer.encode(item[config['tgt_language']]).ids
        src_seq_len = max(src_seq_len, len(src_ids))
        tgt_seq_len = max(tgt_seq_len, len(tgt_ids))
    #print(f'Maximum sequence length of tokenized source sentences is {src_seq_len}')
    #print(f'Maximum sequence length of tokenized target sentences is {tgt_seq_len}')
    # For now we use the maximum of the two + 2 as our joint seq_len. We do +2 because we're going to add up to 2 special tokens.
    seq_len = max(src_seq_len, tgt_seq_len) + 2
    if config['edu']:
        print(f'We use sequence length {seq_len}')
    

    # Now we can create our BilinguaDataset objects. To do: rewrite so that we can use the right seq_len for encoder and decoder.
    train_dataset = BilingualDataset(train_dataset_raw, src_tokenizer, tgt_tokenizer, config['src_language'], config['tgt_language'], seq_len)
    valid_dataset = BilingualDataset(valid_dataset_raw, src_tokenizer, tgt_tokenizer, config['src_language'], config['tgt_language'], seq_len)

    # Now we wrap these __get_item()__ Datasets in DataLoaders so that we get batches.

    train_dataloader = DataLoader(train_dataset, batch_size=config['batch_size'], shuffle=True)
    valid_dataloader = DataLoader(valid_dataset, batch_size=1, shuffle=True) # For validation we're doing one by one for now.

    return train_dataloader, valid_dataloader, src_tokenizer, tgt_tokenizer, seq_len

def get_model(config, src_vocab_size, tgt_vocab_size, seq_len):

    model = build_transformer(config, src_vocab_size, tgt_vocab_size, seq_len, seq_len) # To do: use the right seq_len for encoder and decoder.
    # Note: if GPU RAM is not big enough we can reduce number of heads and/or number of layers and/or batch size.
    return model

def train_model(config):

    # Define the device.
    # Note: model and data must be put on the same device. Loss funtion as well if it inherits from torch.nn.
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device {device}')

    Path(config['model_folder']).mkdir(parents=True, exist_ok=True)    
  
    # Get our data loaders.
    train_dataloader, valid_dataloader, src_tokenizer, tgt_tokenizer, seq_len = get_dataloader(config) 

    # Get our model.
    model = get_model(config, src_tokenizer.get_vocab_size(), tgt_tokenizer.get_vocab_size(), seq_len).to(device) # Send model to device.

    # To do: set up a TensorBoard writer
    writer = SummaryWriter(config['experiment_name'])    
    
    # Get our optimizer to update the weights.
    optimizer = torch.optim.Adam(model.parameters(), lr=config['lr'], eps=1e-9)

    # Create some resilience so that we don't have to start training from scratch if there is a crash during training.
    initial_epoch = 0
    global_step = 0

    # config['preload'] is the last epoch that we successfully completed training with.
    if config['preload']:
        model_filename = get_model_file_path(config, config['preload'])
        print(f'Preloading model {model_filename}')
        state = torch.load(model_filename)
        initial_epoch = state['epoch'] + 1
        optimizer.load_state_dict(state['optimizer_state_dict']) 		# Don't forget to restore the optimizer state as well.
        global_step = state['global_step'] 					# This is for TensorBoard, not using this yet.
        
    # Notes on loss function:
    # We're subclassing via nn, so we have to put this loss function on the device as well since it has parameters.
    # We can use either tokenizer to find the input id for '[PAD]'.
    # The label smoothing here takes 0.1% of the highest score and distributes it over the others - this makes the model more accurate.
    # To do: implement your own CEL: https://discuss.pytorch.org/t/cross-entropy-loss-clarification/103830/2.
    # Use: https://discuss.huggingface.co/t/token-probabilities-dont-agree-with-the-output-loss/25960 
    loss_fn = nn.CrossEntropyLoss(ignore_index=src_tokenizer.token_to_id('[PAD]'), label_smoothing=0.1).to(device)    

    for epoch in range(initial_epoch, config['num_epochs']):

        # model.train() 							# Put model.train() here if you run validation after each epoch.
        batch_iterator = tqdm(train_dataloader, desc=f'Processing epoch {epoch:02d}') # tqdm is what draws the nice progress bars, wrap it around DataLoader

        for batch in batch_iterator:
 
            model.train() 							# Put model.train() here if you run validation after each batch.

            # The model is on the device, so we also need to put all the data (parameters, buffers) it needs on the device.
            # Note: if this stuff doesn't fit in the GPU mem then you get a runtime CUDA out of mem error.

            encoder_input_tensor_batch = batch['encoder_input_tensor'].to(device) 				# dim is (batch_size,seq_len).
            decoder_input_tensor_batch = batch['decoder_input_tensor'].to(device) 				# dim is (batch_size_seq_len).
            encoder_input_tensor_mask_batch = batch['encoder_input_tensor_mask'].to(device) 			# dim is (batch_size,1,1,seq_len).
            decoder_input_tensor_mask_batch = batch['decoder_input_tensor_mask'].to(device) 			# dim is (batch_size,1,seq_len,seq_len).
            decoder_label_tensor_batch = batch['decoder_label_tensor'].to(device)                               # dim is (batch, seq_len).
            
            # Now push the training data tensors through the transformer in 3 steps: encode, decode, project.
            # 1. Encode.
            # Dim of output is (batch_size, seq_len, embed_size).
            encoder_output_tensor_batch = model.encode(encoder_input_tensor_batch, encoder_input_tensor_mask_batch) 

            # 2. Decode, using the output of the encoder that we got in the previous step.
            # Dim of output below is also (batch_size, seq_len, embed_size).
            decoder_output_tensor_batch = model.decode(decoder_input_tensor_batch, decoder_input_tensor_mask_batch, encoder_output_tensor_batch, encoder_input_tensor_mask_batch )

            # 3. Project.
            # Dim of output below is (batch_size, seq_len, tgt_vocab_size).
            transformer_output_tensor_batch = model.project(decoder_output_tensor_batch) 
                   
            # Now let's compare these batched predictions with our batched labels.
            # First view transforms (batch_size, seq_len, tgt_vocab_size) to (batch_size * seq_len, tgt_vocab_size)
            # label.view(-1) flattens out the complete batch over all seq_lens, so to batch_size * seq_len
            if config['edu']:
                print(" ------- cross entropy -------")
                print("encoder_input_tensor_batch:")
                print(encoder_input_tensor_batch)
                print("decoder_input_tensor_batch:")
                print(decoder_input_tensor_batch)
                print("decoder_label_tensor_batch.view(-1):")
                print(decoder_label_tensor_batch.view(-1))
                print("")
                print("Logits return by projection layer:")
                print(transformer_output_tensor_batch.view(-1, tgt_tokenizer.get_vocab_size()))
                print(" -----------------------------")

            # Notes on how cross entropy loss function works here:
            # When 'edu' is on and when using 'en-fr.tiny', we can nicely inspect what's happening here.

            # encoder_input_tensor_batch: (en)
            # tensor([[ 2,  4, 10,  9,  3]]) -> [SOS] I am Chloe [EOS]
            # decoder_input_tensor_batch: (fr)
            # tensor([[ 2,  4, 12,  7,  1]]) -> [SOS] Je suis Chloe [PAD]
            # decoder_label_tensor_batch.view(-1): (fr)
            # tensor([ 4, 12,  7,  3,  1]) -> Je suis Chloe [EOS] [PAD] (label is same as input but shifted one -> )
            #
            # Logits return by projection layer:
            # tensor([[-3.1357, -2.7117, -2.7288, -2.1729, -2.0720, -2.4078, -3.2586, -2.9551, -2.6074, -2.2920, -2.0198, -2.7384, -3.4311],
            # <there are seq_len elements in this dimension, one for each token to be predicted>]], grad_fn=<ViewBackward0>)
            #
            # The tensor above, returned by the projection layer, contains 12 'log_softmaxed probabilities' per token to be predicted, where
            # 12 corresponds to vocab_size for 'fr'. Ideally this will match [ 4, 12,  7,  3,  1] as in the label, but during training we 
            # need to measure the loss, and that is done via cross entropy:
            # 1. Convert the logits (== whatever output from projection layer) to probabilities via regular softmax.
            # 2. Use decoder_label_tensor_batch.view(-1) to retrieve the index of the correct token (this is the id), so '4' in our example.
            # 3. Use this index to see what probability that class got, so the softmaxed '-2.0720' here.
            # 4. do 'loss -= log(probability)'
            # 5. Do this for all predicted tokens (all 5 in our example) and this will give the total cross entropy loss.

            loss = loss_fn(transformer_output_tensor_batch.view(-1, tgt_tokenizer.get_vocab_size()), decoder_label_tensor_batch.view(-1))

            batch_iterator.set_postfix({f"loss": f"{loss.item():6.3f}"}) 					# Show loss on the tqdm progress bar.
            writer.add_scalar('train_loss', loss.item(), global_step) 						# This is for TensorBoard, still needs work.
            writer.flush()											# This is for TensorBoard, still needs work.	

            # Backpropagate the loss.
            loss.backward()

            # Update the weights.
            optimizer.step()
            optimizer.zero_grad() 										# Zero out the gradient.


            # run_validation(model, valid_dataloader, src_tokenizer, tgt_tokenizer, seq_len, device, lambda msg: batch_iterator.write(msg), global_step, writer)

            global_step += 1 											# Used by TensorBoard to keep track of the loss.

            break # REMOVE THIS - temporary in order to just train with one batch containing one sample.

        # Put run_validate here to run validation after every epoch - move model.train() accordingly.

        # Save the model after every epoch in case of a crash
        model_filename = get_model_file_path(config, f'{epoch:02d}') 
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(), 								# These are all the weights of the model.
            'optimizer_state_dict': optimizer.state_dict(),							# Make sure to save optimizer as well.
            'global_step': global_step 
        }, model_filename)

if __name__ == '__main__':
    warnings.filterwarnings('ignore')
    config = get_config()
    train_model(config)
