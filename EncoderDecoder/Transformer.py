import torch
import torch.nn as nn

# (batch, seq_len, embed_size)

########## HEAD ##########

class EncoderForSequenceClassification(nn.Module):

    def __init__(self, config):
        super().__init__()

        self.encoder = Encoder(config)
        self.dropout = nn.Dropout(config.dropout_prob)
        self.classifier = nn.Linear(config.embed_size, config.num_labels)

    def forward(self,input_ids):

        context_vectorized_embeddings = self.Encoder(input_ids)[:, 0, :]
        context_vectorized_embeddings = self.dropout(context_vectorized_embeddings)
        classification = self.classifier(context_vectorized_embeddings)
        return classification

########## HEAD ##########

########## BODY  ##########

class Embedding(nn.Module):

    def __init__(self, config, vocab_size, seq_len):
        super().__init__()

        self.seq_len = seq_len
        self.embed_size = config.embed_size

        self.token_embeddings = nn.Embeddings(vocab_size, config.embed_size)
#       seq_len - 1 is the highest position starting from position 0
        self.position_embeddings = nn.Embeddings(self.seq_len, config.embed_size) # We use a learned position encoding, can also do static/not learned

#        self.layer_norm = nn.LayerNorm(config.embed_size, eps=1e-12) switching to default eps since other norm layers do this as well
        self.layer_norm = nn.LayerNorm(config.embed_size) 

        self.dropout = nn.Dropout(config.dropout_prob)

    def forward(self, input_ids):

        assert seq_len == input_ids.size(1), "Sequence length in config must be same as size of dimension 1 (not 0) in input_ids"
        
        position_ids = torch.arange(seq_len, dtype=torch.long).unsqueeze(0) # this creates a [1,seq_len] tensor

        token_embeddings = self.token_embeddings(input_ids) # some implementations add: "* math.sqrt(embed_size)"
        position_embeddings = self.position_embeddings(position_ids)
        
        embeddings = token_embeddings + position_embeddings

        embeddings = self.layer_norm(embeddings) 

        embeddings = self.dropout(embeddings) # This avoids overfitting

        return embeddings

class Transformer(nn.Module):

    def __init__(self, config, encoder_vocab_size, decoder_vocab_size, encoder_seq_len, decoder_seq_len):
        super().__init__()
        
        self.encoder = Encoder(config, encoder_vocab_size, encoder_seq_len)
        self.decoder = Decoder(config, decoder_vocab_size, decoder_seq_len)
        self.projection_layer = ProjectionLayer(config, decoder_vocab_size)

    def encode(self, encoder_input_ids, encoder_mask): # this can be batched or not, note down dimensions to do
        
        return self.encoder(encoder_input_ids, encoder_mask)

    def decode(self, decoder_input_ids, decoder_mask, encoder_output, encoder_mask):

        return self.decoder(decoder_input_ids, decoder_mask, encoder_output, encoder_mask)

    def project(self, decoder_output):
       
        return self.projection_layer(decoder_output)




class ProjectionLayer(nn.Module):

    def __init__(self, config, vocab_size):
        super().__init__()
 
        self.proj = nn.Linear(config.embed_size,vocab_size)

    def forward(self, decoder_output):
#       (batch, seq_len, config.embed_size) -> (batch, seq_len, vocab_size) with the next word having the highest probability (if greedy is used)        
        return torch.log_softmax(self.proj(decoder_output), dim = -1)

class Decoder(nn.Module):

    def __init__(self, config, seq_len):
        super().__init__()

        self.embeddings = Embedding(config, seq_len)

        self.layer_norm = nn.LayerNorm(config.embed_size)

        self.DecoderBlocks = nn.ModuleList(
           [DecoderBlock(config) for _ in range(config.num_decoderblocks)] 
        )

    def forward(self, input_ids, decoder_mask, encoder_output, encoder_mask):
        
        context_vectorized_embeddings = self.embeddings(input_ids) # technically, at this point they are not context_vectorized yet
        for decoder_block in self.DecoderBlocks:
            context_vectorized_embeddings = decoder_block(context_vectorized_embeddings, decoder_mask, encoder_output, encoder_mask)

#       Since we are doing pre-layer normalization, we need to do one final normalization after all the DecoderBlocks have run, as we want to Decoder itself to output something normalized
        norm_context_vectorized_embeddings = self.layer_norm(context_vectorized_embeddings)

        return norm_context_vectorized_embeddings # now context information is added, by one or more DecoderBlocks

class DecoderBlock(nn.Module):

    def __int__(self,config):
        super().__init__()

        self.layer_norm_1 == nn.LayerNorm(config.embed_size)
        self.layer_norm_2 == nn.LayerNorm(config.embed_size)
        self.layer_norm_3 == nn.LayerNorm(config.embed_size)
    
        self.self_multi_head_attention = MultiHeadAttention(config)
        self.cross_multi_head_attention = MultiHeadAttention(config)

        self.feed_forward = FeedForward(config)

    def forward(self,embedding, decoder_mask, encoder_output, encoder_mask):
        
        # embedding is what you add to the upcoming skip connection
   
        norm_embedding = self.layer_norm_1(embedding)
    
        self_multihead_context_vector_skip = embedding + self.self_multi_head_attention(norm_embedding, decoder_mask)

        # self_multihead_context_vector_skip is what you add to the upcoming skip connection

        norm_self_multihead_context_vector_skip = self.layer_norm_2(self_multihead_context_vector_skip)

        cross_multihead_context_vector_skip = self_multihead_context_vector_skip + self.cross_multi_head_attention(norm_self_multihead_context_vector_skip, None, encoder_output) 

        # cross_multihead_context_vector_skip is what you add to the upcoming skip connection

        norm_cross_multihead_context_vector_skip = self.layer_norm_3(cross_multihead_context_vector_skip)

        encoder_block_cv = cross_multihead_context_vector_skip + self.feed_forward(norm_cross_multihead_context_vector_skip)

        return encoder_block_cv
        
class Encoder(nn.Module):

    def __init__(self, config, seq_len):
        super().__init__()

        self.embeddings = Embedding(confiq, seq_len)

        self.layer_norm = nn.LayerNorm(config.embed_size)

        self.EncoderBlocks = nn.ModuleList(
           [EncoderBlock(config) for _ in range(config.num_encoderblocks)] 
        )

    def forward(self, input_ids, mask):
        
        context_vectorized_embeddings = self.embeddings(input_ids) # technically, at this point they are not context_vectorized yet
        for encoder_block in self.EncoderBlocks:
            context_vectorized_embeddings = encoder_block(context_vectorized_embeddings, mask)

#       Since we are doing pre-layer normalization, we need to do one final normalization after all the EncoderBlocks have run, as we want to Encoder itself to output something normalized
        norm_context_vectorized_embeddings = self.layer_norm(context_vectorized_embeddings)

        return norm_context_vectorized_embeddings # now context information is added, by one or more EncoderBlocks




class EncoderBlock(nn.Module):
    
    def __init__(self,config):
        super().__init__()

        self.layer_norm_1 == nn.LayerNorm(config.embed_size)
        self.layer_norm_2 == nn.LayerNorm(config.embed_size)
    
        self.multi_head_attention = MultiHeadAttention(config)

        self.feed_forward = FeedForward(config)

    def forward(self,embedding, mask):
  
        norm_embedding = self.layer_norm_1(embedding)
    
        multihead_context_vector_skip = embedding + self.multi_head_attention(norm_embedding, mask)

        norm_multihead_context_vector_skip = self.layer_norm_2(multihead_context_vector_skip)

        encoder_block_cv = multihead_context_vector_skip + self.feed_forward(norm_multihead_context_vector_skip)

        return encoder_block_cv


class FeedForward(nn.Module):

    def __init__(self,config):
        super().__init__()

        self.ff1 = nn.Linear(config.embed_size, config.ff_intermediate_size) # bias is True by default
        self.ff2 = nn.Linear(config.ff_intermediate_size, config.embed_size) # bias is True by default
        self.gelu = nn.GELU()
        self.dropout = nn.Dropout(config.dropout_prob)

    def forward(self,x):
        x = self.ff1(x)
        x = self.gelu(x)
        x = self.ff2(x)
        x = self.dropout(x) # some implementations put the dropout before ff2 (but after gelu/relu)
        return x
        

class MultiHeadAttention(nn.Module):
    
    def __init__(self, config):
        super().__init__()

        embed_dim = config.embed_dim
        num_attention_heads = config.num_attention_heads

#       If you want to support the case where you divide each embedding by the number of heads then you also need to provide code to
#       set the attention_head_input_dim accordingly. Here we just set it to embed_dim.
#       However, attention_head_output_dim is ALWAYS embed_dim // num_attention_heads (although that's also by choice).

#       Alternative approach (Umar Jamil implementation) is that you have one Wq, Wk and Wv per multi-head attention
#       and the first thing you do is Q x Wq => Q', K * Wk => K' and V * Wv => V' which means that Wq, Wk and Wv are all dimension (embed_size,embed_size)
#       so square matrices (technically Wv can be (embed_size, something else) then 
#       AFTER doing this, you then split the ' matrices in #heads parts (say 4) and then you do :
#       Q'(1) x K'(1) == dotproduct -> /sqrt(remaining size of key/query) -> softmax -> multiply by V'(1) ==> head(1) attention
#       Then when you have all 4 you reaassemble in matrix which you multiply x with Wo to deliver the mutihead_context_vector
#       Wo will be (embed_size, embed_size) unless you gave V' another dimension in which case first dim will be the lend of the sum of all
#       head_attentions (i.e., the concat which needs to be brought back to embed_dim by Wo)
#       This may be a simpler implementation TBD 

        attention_head_input_dim = embed_dim
        attention_head_output_dim = embed_dim // num_attention_heads

        assert embed_dim % num_attention_heads == 0, "Embedding size must be divisible by number of heads" 

        self.heads = nn.ModuleList(
            
            [AttentionHead(attention_head_input_dim, attention_head_output_dim) for _ in range(num_attention_heads)]

        )

        self.Wo = nn.Linear(embed_dim, embed_dim)

    def forward(self, embedding, mask, encoder_output=None):
    
#       This is the point where you can decide to turn embedding into embedding_div_by_nr_of_heads and feed every attention head only
#       a portion of each embedding (each attention head gets the whole sequence, but only a portion of the embedding per token/id
#       In this case we are passing on the full embedding to attention_head()

        if encoder_output is None:
            concatenated_head_context_vectors = torch.cat(
               [attention_head(embedding, mask) for attention_head in self.heads], dim=-1 
            )
        else:
            concatenated_head_context_vectors = torch.cat(
               [attention_head(embedding, mask, encoder_output, encoder_output) for attention_head in self.heads], dim=-1 
            )

        assert concatenated_head_context_vectors.size(-1) == embedding.size(-1), "Concatenated head context vectors size must match embedding size"        

        multihead_context_vector = self.Wo(concatenated_head_context_vectors)

        assert multihead_context_vector.size(-1) == embedding.size(-1), "Output of Wo size must match embedding size"

        return multihead_context_vector

class AttentionHead(nn.Module):
    
    def __init__(self, attn_head_input_dim, attn_head_output_dim):
        super().__init__()
        
        self.Wq = nn.Linear(attn_head_input_dim, attn_head_output_dim, bias=False)
        self.Wk = nn.Linear(attn_head_input_dim, attn_head_output_dim, bias=False)

#       Note: technically Wv can have an output_dim that is different from the output_dim of Wq and Wk
#       Wq and Wk must have the same ouput_dim since we're doing dot product with the outputs

        self.Wv = nn.Linear(attn_head_input_dim, attn_head_output_dim, bias=False)

        self.dropout = nn.Dropout(config.dropout_prob)

#   What is fed in into an attention head can be the whole embedding, or a part of the embedding if the case where
#   each attention head gets a portion of the embedding. In this case each head still gets the whole sequence, but only
#   a portion of the original embedding -> input_size == embedding / nr_of_heads

    def forward(self, embedding_or_embedding_div_by_nr_of_heads, mask, encoder_output_as_key=None, encoder_output_as_value=None): # Last two are for Decoder cross attention
        
        if key is not None and value is not None:
            head_context_vector = scaled_dot_product_attention(
                self.Wq(embedding_or_embedding_div_by_nr_of_heads),
                encoder_output_as_key,
                encoder_output_as_value,
                mask
            )
        elif key is None and value is None:
            head_context_vector = scaled_dot_product_attention(
                self.Wq(embedding_or_embedding_div_by_nr_of_heads),
                self.Wk(embedding_or_embedding_div_by_nr_of_heads),
                self.Wv(embedding_or_embedding_div_by_nr_of_heads),
                mask
            )
        else:
            print("We need both they keys and the vaues from the Encoder and we only got one of them")
            sys.exit()

        return head_context_vector

#     The q/k/v inputs are expected to be tensors with 3 dimensions: [batch_number, sequence_in_the_batch, embedding_in_the_sequence]
#     Example is [1, 5, 768] for sequences of length 5 with embeddings of size 768 -> normally this 768 is reduced by Wq or Wk, Wq and Wk need to have same size 
#     because we need to make a dot product from them, Wv can be different size
#     Transpose (1,2) makes the key in format [1, 768, 5] so that when we do Q * K the result is [1, 5, 5] as we multiply the "deepest" matrix
#     That last 5 here is correct: you get one attention score per input_id/token in the sequence 
#     When then multiply each softmaxed score/attention weight with the corresponding value, [1, 5, 5] * [1, 5, <val size>] to go to [1, 5, <val size>] (returned)
#     Here val size is 768 as well (normally reduced by Wv) but it does not have to be the same as Query/Key (reduced by Wq/Wk)

#    def scaled_dot_product_attention(query, key, value, mask=None, dropout: nn.Dropout):
    def scaled_dot_product_attention(query, key, value, mask=None):
        dimension_of_key = key.size(-1)
        attention_scores = torch.bmm(query, key.transpose(1,2)) / sqrt(dimension_of_key)
        if mask is not None:
            attention_scores = scores.masked_fill(mask == 0, float("-inf"))
#        attention_weights = F.softmax(scores, dim=-1) # Using log softmax provides more training stability
        attention_weights = torch.log_softmax(scores, dim=-1)
#       if dropout is not None:
#           attention_weights = dropout(attention_weights)
        return attention_weights.bmm(value)

def build_transformer(encoder_vocab_size, decoder_vocab_size, encoder_seq_len, decoder_seq_len) -> Transformer:

    transformer = Transformer(config, encoder_vocab_size, decoder_vocab_size, encoder_seq_len, decoder_seq_len)

#    Parameter initialization

    for p in transformer.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform(p)

    return transformer

        

    
    
    
    





