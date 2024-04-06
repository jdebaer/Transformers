import torch
import torch.nn as nn
import torch.nn.functional as F
from math import sqrt

##########################
########## HEAD ##########


class EncoderForSequenceClassification(nn.Module):

    def __init__(self, config):
        super().__init__()

        self.encoder = Encoder(config)
        self.dropout = nn.Dropout(config['dropout_prob'])
        self.classifier = nn.Linear(config['embed_size'], config['num_labels'])

    def forward(self,input_ids):

        context_vectorized_embeddings = self.Encoder(input_ids)[:, 0, :]
        context_vectorized_embeddings = self.dropout(context_vectorized_embeddings)
        classification = self.classifier(context_vectorized_embeddings)
        return classification

########## HEAD ##########
##########################

##########################
######### BODY  ##########

class Embeddings(nn.Module):

    # An Embeddings maps each input id to a vector of size embed_size which contains token and position embeddings.
    # For an Encoder-Decoder we'll need two Embedddings, each with their own vocab size and possible also a different seq_len.
    # Note: what gets pushed through these Embeddings are the batched training samples, delivered by our dataloaders.

    def __init__(self, config, vocab_size, seq_len):
        super().__init__()

        self.seq_len = seq_len
        self.embed_size = config['embed_size']
        self.token_embeddings = nn.Embedding(vocab_size, config['embed_size']) 		# Technically these are id embeddings.
        self.position_embeddings = nn.Embedding(self.seq_len, config['embed_size']) 	# We use a learned position encoding.
        self.layer_norm = nn.LayerNorm(config['embed_size']) 				# This uses the default eps value.
        self.dropout = nn.Dropout(config['dropout_prob'])

    def forward(self, input_ids):

        # input_ids dim is (batch, seq_len).
        # Note: during training, seq_len and the length of input_ids is the same, since we padding input_ids to reach seq_len.
        #       However, during validation, seq_len and length of input_ids will mostly differ, since for input_ids we're starting with just 1 id.
        if self.training:
              assert self.seq_len == input_ids.size(1), "During training, seq_len must match sequence dimension of input_ids."
        
        #position_ids = torch.arange(self.seq_len, dtype=torch.long).unsqueeze(0) 		# Create a [1,seq_len] tensor containing 0,1,2,...
        position_ids = torch.arange(input_ids.size(1), dtype=torch.long).unsqueeze(0) 		# Create a [1,seq_len] tensor containing 0,1,2,...
        token_embeddings = self.token_embeddings(input_ids) 				# Some implementations add: "* math.sqrt(embed_size)".
        position_embeddings = self.position_embeddings(position_ids)
        embeddings = token_embeddings + position_embeddings				# (batch_size, seq_len, embed_size) + (1, seq_len, embed_size)		
        embeddings = self.layer_norm(embeddings) 
        embeddings = self.dropout(embeddings) 
        return embeddings								# Dim is (batch, seq_len, embed_size). 

class Transformer(nn.Module):

    def __init__(self, config, encoder_vocab_size, decoder_vocab_size, encoder_seq_len, decoder_seq_len):
        super().__init__()

        self.encoder_embeddings = Embeddings(config, encoder_vocab_size, encoder_seq_len)	# This will be src_vocab_size and scr_seq_len, since encoder.
        self.decoder_embeddings = Embeddings(config, decoder_vocab_size, decoder_seq_len)	# This will be src_vocab_size and scr_seq_len, since encoder.
        self.encoder = Encoder(config, encoder_vocab_size, encoder_seq_len)
        self.decoder = Decoder(config, decoder_vocab_size, decoder_seq_len)
        self.projection_layer = ProjectionLayer(config, decoder_vocab_size)

    def encode(self, encoder_input_ids, encoder_mask): 					# At this point dim is (batch, seq_len).
        
        return self.encoder(encoder_input_ids, encoder_mask, self.encoder_embeddings)

    def decode(self, decoder_input_ids, decoder_mask, encoder_output, encoder_mask):	# At this point dim is (batch, seq_len).	

        return self.decoder(decoder_input_ids, decoder_mask, encoder_output, encoder_mask, self.decoder_embeddings)

    def project(self, decoder_output):
       
        return self.projection_layer(decoder_output)					# Output dim is (batch, seq_len, decoder_vocab_size)




class ProjectionLayer(nn.Module):

    def __init__(self, config, vocab_size):
        super().__init__()
 
        self.proj = nn.Linear(config['embed_size'], vocab_size)					# This is decoder_vocab_size in encoder/decoder.

    def forward(self, decoder_output):
        # (batch, seq_len, embed_size) -> (batch, seq_len, vocab_size) with the next word having the highest probability (if greedy is used).        
        # At each position in the seq_len dimension, we have an embedding that contains context both from encoder and decoder now, and that only
        # looked at itself + previous tokens/ids for the decoder self-attention. Each of these embeddings is now asked to predict the next word in
        # parallel. This is what happens during training. This is then compared with the shifted tokens/ids in the label.
        
        return torch.log_softmax(self.proj(decoder_output), dim = -1)				# log_softmax for more training stability.

class Decoder(nn.Module):

    def __init__(self, config, vocab_size, seq_len):						# We calculate seq_len, so it's not part of config.
        super().__init__()

        self.layer_norm = nn.LayerNorm(config['embed_size'])
        self.DecoderBlocks = nn.ModuleList(
           [DecoderBlock(config) for _ in range(config['num_decoderblocks'])] 
        )

    def forward(self, input_ids, decoder_mask, encoder_output, encoder_mask, embeddings):	# Encoder_mask is padding, decoder_mask is padding and causal.
        
        context_vectorized_embeddings = embeddings(input_ids) 
        for decoder_block in self.DecoderBlocks:
            context_vectorized_embeddings = decoder_block(context_vectorized_embeddings, decoder_mask, encoder_output, encoder_mask)

	# Since we are doing pre-layer normalization, we need to do one final normalization after 
        # all the EncoderBlocks have run, as we want to Encoder itself to output something normalized.
        norm_context_vectorized_embeddings = self.layer_norm(context_vectorized_embeddings)
        return norm_context_vectorized_embeddings 

class DecoderBlock(nn.Module):

    def __init__(self,config):
        super().__init__()

        self.layer_norm_1 = nn.LayerNorm(config['embed_size'])
        self.layer_norm_2 = nn.LayerNorm(config['embed_size'])
        self.layer_norm_3 = nn.LayerNorm(config['embed_size'])
        self.self_multi_head_attention = MultiHeadAttention(config, 'self')
        self.cross_multi_head_attention = MultiHeadAttention(config, 'cross')
        self.feed_forward = FeedForward(config)

    def forward(self, embedding, decoder_mask, encoder_output, encoder_mask):			# Encoder_mask is padding, decoder_mask is padding and causal.
        
        norm_embedding = self.layer_norm_1(embedding)
        # self_multihead_context_vector_skip is what you add to the upcoming skip connection.
        self_multihead_context_vector_skip = embedding + self.self_multi_head_attention(norm_embedding, decoder_mask, caller='decoder')
        norm_self_multihead_context_vector_skip = self.layer_norm_2(self_multihead_context_vector_skip)
        # cross_multihead_context_vector_skip is what you add to the upcoming skip connection.
        cross_multihead_context_vector_skip = self_multihead_context_vector_skip + self.cross_multi_head_attention(norm_self_multihead_context_vector_skip, encoder_mask, encoder_output, caller='decoder') 
        norm_cross_multihead_context_vector_skip = self.layer_norm_3(cross_multihead_context_vector_skip)
        encoder_block_context_vector = cross_multihead_context_vector_skip + self.feed_forward(norm_cross_multihead_context_vector_skip)
        # Return is not layer-normalized.
        return encoder_block_context_vector
        
class Encoder(nn.Module):

    def __init__(self, config, vocab_size, seq_len):						# We calculate seq_len, so it's not part of config.
        super().__init__()

        self.layer_norm = nn.LayerNorm(config['embed_size'])
        self.EncoderBlocks = nn.ModuleList(
           [EncoderBlock(config) for _ in range(config['num_encoderblocks'])] 
        )

    def forward(self, input_ids, mask, embeddings):						# This is the padding mask, no causal mask for encoder.

        context_vectorized_embeddings = embeddings(input_ids) 					# Output dim here is (batch, seq_len, embed_size).
        for encoder_block in self.EncoderBlocks:
            context_vectorized_embeddings = encoder_block(context_vectorized_embeddings, mask)	# Add more and more context.

	# Since we are doing pre-layer normalization, we need to do one final normalization after 
        # all the EncoderBlocks have run, as we want to Encoder itself to output something normalized.
        norm_context_vectorized_embeddings = self.layer_norm(context_vectorized_embeddings)
        return norm_context_vectorized_embeddings 

class EncoderBlock(nn.Module):
    
    def __init__(self, config):
        super().__init__()

        self.layer_norm_1 = nn.LayerNorm(config['embed_size'])
        self.layer_norm_2 = nn.LayerNorm(config['embed_size'])
        self.self_multi_head_attention = MultiHeadAttention(config, 'self')	# Encoders use self attention.
        self.feed_forward = FeedForward(config)

    def forward(self, embedding, mask):							# This is the padding mask, no causal mask for encoder.
  
        norm_embedding = self.layer_norm_1(embedding)					# We are doing pre-layer normalization.
        multihead_context_vector_skip = embedding + self.self_multi_head_attention(norm_embedding, mask, caller='encoder')
        norm_multihead_context_vector_skip = self.layer_norm_2(multihead_context_vector_skip)
        encoder_block_context_vector = multihead_context_vector_skip + self.feed_forward(norm_multihead_context_vector_skip)
        # Return is not layer-normalized.
        return encoder_block_context_vector

class FeedForward(nn.Module):

    def __init__(self,config):
        super().__init__()

        self.ff1 = nn.Linear(config['embed_size'], config['ff_intermediate_size']) # bias is True by default
        self.ff2 = nn.Linear(config['ff_intermediate_size'], config['embed_size']) # bias is True by default
        self.gelu = nn.GELU()									# Smoother RELU.
        self.dropout = nn.Dropout(config['dropout_prob'])

    def forward(self,x):
        x = self.ff1(x)
        x = self.gelu(x)
        x = self.ff2(x)
        x = self.dropout(x) 									# Can also put the dropout before ff2 (but after gelu/relu).
        return x

class MultiHeadAttention(nn.Module):
    
    def __init__(self, config, type='self'):
        super().__init__()

        # In this implementation, the output dimension of an attention head is the embedding size // the number of heads.
        # The dimensionality reduction is done by the W matrices, with each attention head having its own set of these.
        # An alternative implementation (e.g., https://www.youtube.com/watch?v=ISNdQcPhsts) is to have W matrices with the size of the embedding
        # that still output the embedding size, and then you go and split that output up as per the number of heads.

        self.type = type
        embed_size = config['embed_size']
        num_attention_heads = config['num_attention_heads']
        attention_head_input_dim = embed_size
        attention_head_output_dim = embed_size // num_attention_heads
        assert embed_size % num_attention_heads == 0, "Embedding size must be divisible by number of heads" 
        self.attention_heads = nn.ModuleList(
            [AttentionHead(config, attention_head_input_dim, attention_head_output_dim, type) for _ in range(num_attention_heads)]
        )
        # What we feed to Wo are the concatenated head context vectors. This concatenation will have the embedding size again.
        self.Wo = nn.Linear(embed_size, embed_size)						

    def forward(self, embedding, mask, encoder_output=None, caller=None):
    
        # Note on the mask: this can be a combination of a causal mask and a padding mask (decoder) or a padding mask only (encoder).
        if self.type == 'self':										# Self attention mode.
            concatenated_head_context_vectors = torch.cat(
               [attention_head(embedding, mask, caller=caller) for attention_head in self.attention_heads], dim=-1 
            )
        else:												# Cross attention mode.
            concatenated_head_context_vectors = torch.cat(
               [attention_head(embedding, mask, encoder_output, caller=caller) for attention_head in self.attention_heads], dim=-1 
            )
        assert concatenated_head_context_vectors.size(-1) == embedding.size(-1), "Concatenated head context vectors size must match embedding size."        
        multihead_context_vector = self.Wo(concatenated_head_context_vectors)
        assert multihead_context_vector.size(-1) == embedding.size(-1), "Output of Wo size must match embedding size"
        return multihead_context_vector

class AttentionHead(nn.Module):

    # Vanilla implementation of attention head that receives the full embedding and reduces the dimensionality via dedicated W matrices.
    # Alternative approach splits the embedding and gives each head only a subset - we don't do that here.
    # Essentially, an attention head receives an embedding and returns (attention_weights * values) where output dim is embedding size // number of heads.
    # This output dim formula is just convention, technically we can output any dimension.
    # The calculation of the attn_head_output_dim is done in the the multi head class since that one knows how many heads we have.
    
    def __init__(self, config, attn_head_input_dim, attn_head_output_dim, type='self'):
        super().__init__()
        
        self.type = type
        self.config = config

        self.Wq = nn.Linear(attn_head_input_dim, attn_head_output_dim, bias=False)
        if self.type == 'self':
            self.Wk = nn.Linear(attn_head_input_dim, attn_head_output_dim, bias=False)
            # Note: technically Wv can have an output_dim that is different from the output_dim of Wq and Wk.
            #       Wq and Wk must have the same ouput_dim since we're doing dot product with the outputs.
            self.Wv = nn.Linear(attn_head_input_dim, attn_head_output_dim, bias=False)
        self.dropout = nn.Dropout(config['dropout_prob'])

    def forward(self, embedding, mask, input_for_cross_attention=None, caller=None): 

        # Note on the mask: this can be a combination of a causal mask and a padding mask (decoder) or a padding mask only (encoder).

        if self.type == 'self':
            head_context_vector = self.scaled_dot_product_attention(
                self.Wq(embedding),
                self.Wk(embedding),
                self.Wv(embedding),
                self.dropout,
                mask,
                self.type,
                caller
            )
        else:
            head_context_vector = self.scaled_dot_product_attention(
                self.Wq(embedding),							# For cross attention we use the query from the decoder.
                input_for_cross_attention, 						# Used for cross attention key.
                input_for_cross_attention, 						# User for cross attention value.
                self.dropout,
                mask,
                self.type,
                caller
            )
            
        return head_context_vector

    def scaled_dot_product_attention(self, query, key, value, dropout: nn.Dropout, mask=None, type=None, caller=None):

        # The q/k/v inputs are expected to be tensors with 3 dimensions: [batch_size, seq_len, embed_size].
        # Example is [1, 5, 768] for sequences of length 5 with embeddings of size 768 -> 768 is reduced by W matrices as per the above.
        # Transpose (-2,-1) puts the key in format [1, 768, 5] so that when we do Q * K the result is [1, 5, 5] as we multiply the "deepest" matrix.
        # This essentially is the "resonance" of each word with each other word, attention_scores, with dim (batch_size, seq_len, seq_len).
        # When then multiply each softmaxed attention weight with the corresponding value, [1, 5, 5] * [1, 5, <val size>] to go to [1, 5, <val size>] (returned).
        # Note that in our implementation <val size> will be 5 as well, although technically is does not have to be.

        # Note on how this behaves during validation, and with cross attention:
        # In this case the incoming query will be (batch_size, <number of predicted ids so far>, embed size). Let's do an example with [1, 2, 768] 
        # meaning that 2 ids heve been predicted so far. In this case the matrix multiplications looks like this:
        # [1, 2, 768] x [1, 768, 5] ---> [1,2,5] which gives us the resonance of each predicted token with the output from the encoder.
        # Then we have [1, 2, 5] x [1, 5, <val size>] ---> [1,2,<val size>] which is what we need in the next step.
        # For the decoder self attention it's the same but it would start out as [1, 2, 768] x [1, 768, 2] -> [1, 2, 2].

        if self.config['edu']:
            print("----------------------------")
            print(caller)								# Will say "encoder" or "decoder".
            print(type)									# Will say "self" or "cross".
            print("query:")
            print(query.size())
            print(query)
            print("key:")
            print(key.size())
            print(key)
            print("key transposed:")
            print(key.transpose(-2,-1))

        dim_of_key = key.size(-1)							# Can use dim of query as well, have to be the same.
        if self.config['edu']:
            attention_scores = torch.bmm(query, key.transpose(-2,-1))			# Don't normalize so we can eyeball dot product easily.
        else:
            attention_scores = torch.bmm(query, key.transpose(-2,-1))/sqrt(dim_of_key)	# Normalized dot product.

        if self.config['edu']:
            print("attention_scores:")
            print(attention_scores.size())
            print(attention_scores)

        if mask is not None:
            if self.config['edu']:
                print("mask:")
                print(mask.size())
                print(mask)
            # Softmax has e ** x in the numerator, and e ** -inf == 0. Having 0 as the attention score is our objective with the causal mask.
            attention_scores = attention_scores.masked_fill(mask == 0, float("-inf"))
            if self.config['edu']:
                print("attention_scores after masking:")
                print(attention_scores.size())
                print(attention_scores)

            # Note to understand how the mask is used with the dot product (skipping the normalization here):
            # When we multiply query with key, we essentially measure the resonance of every word with every word in the sequence.
            # However, for the first word, we don't want to do that for any word that follows (and so on for 2nd, 3rd word ...).
            # 
            # 1,1,1    1,2,3   3, 6, 9    3 here is the resonance of the first word with itself, while 6 is the resonance of word 1 with word 2.
            # 2,2,2  * 1,2,3 = 6,12,18    However, we want to mask out the 6 since we only want words to resonate with themselves or previous words.
            # 3,3,3    1,2,3   9,18,27    Hence, we apply the mask, like this:
            #
            # 3, 6, 9                1, 0, 0        3, -inf, -inf                            1,   0,  0 -> sums to 1
            # 6,12,18 -> masked_fill(1, 1, 0) gives 6,   12, -inf which softmax turns into .33, .66,  0 -> sums to 1
            # 9,18,27                1, 1, 1        9,   18,   27                          .16, .33, .5 -> sums to 1
            #
            # Note that if we receive a padding-only mask, the mask will have this form [[[1,1,0]]] i.e., dim (1,1,seq_len), which leads to:
            #
            # 1,2,3                                1,2,-inf
            # 4,5,6 -> masked_fill([[[1,1,0]]]) -> 4,5,-inf
            # 7,8,9                                7,8,-inf
            #
        attention_weights = F.softmax(attention_scores, dim = -1)
        if self.config['edu']:
            print("attention_weights:")
            print(attention_weights.size())
            print(attention_weights)

        if dropout is not None:
           attention_weights = dropout(attention_weights)

        ## return attention_weights.bmm(value), attention_weights			# To do: incorporate attention_weights for visualization.
        #return attention_weights.bmm(value)						# Attention weights * values == head context vector.
        if self.config['edu']:
            print("value:")
            print(value.size())
            print(value)
        head_context_vector = torch.bmm(attention_weights, value)

        if self.config['edu']:
            print("head_context_vector:")
            print(head_context_vector.size())
            print(head_context_vector)

        return head_context_vector

def build_transformer(config, encoder_vocab_size, decoder_vocab_size, encoder_seq_len, decoder_seq_len) -> Transformer:

    if config['edu']:
        print("Building transformer in 'edu' mode.")

    transformer = Transformer(config, encoder_vocab_size, decoder_vocab_size, encoder_seq_len, decoder_seq_len)
#    Parameter initialization
    for p in transformer.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform(p)
    return transformer
