import torch
import torch.nn as nn

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

    # An Embedding maps each input id to a vector of size embed_size.
    # For an Encoder-Decoder we'll need two Embedddings, each with their own vocab size and possible also a different seq_len.

    def __init__(self, config, vocab_size, seq_len):
        super().__init__()

        self.seq_len = seq_len
        self.embed_size = config['embed_size']
        self.token_embeddings = nn.Embeddings(vocab_size, config.embed_size) 		# Technically these are id embeddings.
        self.position_embeddings = nn.Embeddings(self.seq_len, config.embed_size) 	# We use a learned position encoding.
        self.layer_norm = nn.LayerNorm(config.embed_size) 				# This uses the default eps value.
        self.dropout = nn.Dropout(config['dropout_prob'])

    def forward(self, input_ids):

        assert seq_len == input_ids.size(1), "Sequence length must be the same as the size of dimension 1 in input_ids."
        position_ids = torch.arange(seq_len, dtype=torch.long).unsqueeze(0) 		# Create a [1,seq_len] tensor containing 0,1,2,...
        token_embeddings = self.token_embeddings(input_ids) 				# Some implementations add: "* math.sqrt(embed_size)".
        position_embeddings = self.position_embeddings(position_ids)
        embeddings = token_embeddings + position_embeddings				
        embeddings = self.layer_norm(embeddings) 
        embeddings = self.dropout(embeddings) 
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

    # Vanilla implementation of attention head that receives the full embedding and reduces the dimensionality via dedicated W matrices.
    # Alternative approach splits the embedding and gives each head only a subset - we don't do that here.
    # Essentially, an attention head receives an embedding and returns (attention_weights * values) where output dim is embedding size // number of heads.
    # This output dim formula is just convention, technically we can output any dimension.
    # The calculation of the attn_head_output_dim is done in the the multi head class since that one knows how many heads we have.
    
    def __init__(self, attn_head_input_dim, attn_head_output_dim, type='self'):
        super().__init__()
        
        self.type = type

        self.Wq = nn.Linear(attn_head_input_dim, attn_head_output_dim, bias=False)
        if type is not 'self':
            self.Wk = nn.Linear(attn_head_input_dim, attn_head_output_dim, bias=False)
            # Note: technically Wv can have an output_dim that is different from the output_dim of Wq and Wk.
            #       Wq and Wk must have the same ouput_dim since we're doing dot product with the outputs.
            self.Wv = nn.Linear(attn_head_input_dim, attn_head_output_dim, bias=False)
        self.dropout = nn.Dropout(config['dropout_prob'])

    def forward(self, embedding, causal_mask, input_for_cross_attention=None): 

        if self.type == 'self':
            head_context_vector = scaled_dot_product_attention(
                self.Wq(embedding),
                self.Wk(embedding),
                self.Wv(embedding),
                causal_mask,
                self.dropout
            )
        else:
            head_context_vector = scaled_dot_product_attention(
                self.Wq(embedding),							# For cross attention we use the query from the decoder.
                input_for_cross_attention, 						# Used for cross attention key.
                input_for_cross_attention, 						# User for cross attention value.
                causal_mask,
                self.drouput
            )
            
        return head_context_vector

    def scaled_dot_product_attention(query, key, value, causal_mask=None, dropout: nn.Dropout):

        # The q/k/v inputs are expected to be tensors with 3 dimensions: [batch_size, seq_len, embed_size].
        # Example is [1, 5, 768] for sequences of length 5 with embeddings of size 768 -> 768 is reduced by W matrices as per the above.
        # Transpose (-2,-1) puts the key in format [1, 768, 5] so that when we do Q * K the result is [1, 5, 5] as we multiply the "deepest" matrix.
        # This essentially is the "resonance" of each word with each other word.
        # When then multiply each softmaxed attention weight with the corresponding value, [1, 5, 5] * [1, 5, <val size>] to go to [1, 5, <val size>] (returned).
        # Note that in our implementation <val size> will be 5 as well, although technically is does not have to be.

        dim_of_key = key.size(-1)							# Can use dim of query as well, have to be the same.
        attention_scores = torch.bmm(query, key.transpose(-2,-1))/sqrt(dim_of_key)	# Normalized dot product.

        if mask is not None:
            # Softmax has e ** x in the numerator, and e ** -inf == 0. Having 0 as the attention score is our objective with the causal mask.
            attention_scores = attention_scores.masked_fill(mask == 0, float("-inf"))

            # Note to understand how the mask is used:
            # When we multiply query with key, we essentially measure the resonance of every word with every word in the sequence.
            # However, for the first word, we don't want to do that for any word that follows (and so on for 2nd, 3rd word ...).
            # 


        attention_weights = attention_scores.softmax(dim = -1)
        if dropout is not None:
           attention_weights = dropout(attention_weights)
        # return attention_weights.bmm(value), attention_weights			# To do: incorporate attention_weights for visualization.
        return attention_weights.bmm(value)						# Attention weights * values == head context vector.

def build_transformer(encoder_vocab_size, decoder_vocab_size, encoder_seq_len, decoder_seq_len) -> Transformer:

    transformer = Transformer(config, encoder_vocab_size, decoder_vocab_size, encoder_seq_len, decoder_seq_len)

#    Parameter initialization

    for p in transformer.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform(p)

    return transformer
