import torch
import torch.nn as nn
import torch.nn.functional as F
import math

# Encoder
class Positional_Encoding(nn.Module):
    def __init__(self, hidden_size, max_len=5000):
        super(Positional_Encoding, self).__init__()
        Position_Record = torch.zeros(max_len, hidden_size)  
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)  
        div_term = torch.exp(torch.arange(0, hidden_size, 2).float() * (-math.log(10000.0) / hidden_size))
        
        Position_Record[:, 0::2] = torch.sin(position * div_term)
        Position_Record[:, 1::2] = torch.cos(position * div_term)
        
        Position_Record = Position_Record.unsqueeze(0)  
        self.register_buffer('Position_Record', Position_Record)

    def forward(self, x):
        # x.shape: [batch_size, seq_len, hidden_size]
        x = x + self.Position_Record[:, :x.size(1), :]
        return x

class Self_Attention_Alibi(nn.Module):
    def __init__(self, hidden_size, n_head=2):
        super(Self_Attention_Alibi, self).__init__()
        
        torch.manual_seed(123)
        self.n_head = n_head
        self.d_per_head = hidden_size // n_head  

        self.W_query = nn.Parameter(torch.rand(n_head, self.d_per_head, hidden_size))
        self.W_key = nn.Parameter(torch.rand(n_head, self.d_per_head, hidden_size))
        self.W_value = nn.Parameter(torch.rand(n_head, self.d_per_head, hidden_size))
        
        self.output_layer = nn.Linear(hidden_size, hidden_size)

        self.alibi_slope = torch.tensor([-(i + 1) for i in range(n_head)], dtype=torch.float32).view(n_head, 1, 1)

    def forward(self, x, return_attention=False):
        batch_size, seq_len, hidden_size = x.size()
        all_heads = []
        
        distance = torch.arange(seq_len).view(1, -1) - torch.arange(seq_len).view(-1, 1)
        distance = distance.to(x.device)  

        for i in range(self.n_head):
            queries = x.matmul(self.W_query[i].T)  # Shape: [batch_size, seq_len, d_per_head]
            keys = x.matmul(self.W_key[i].T)       # Shape: [batch_size, seq_len, d_per_head]
            values = x.matmul(self.W_value[i].T)   # Shape: [batch_size, seq_len, d_per_head]

            scores = queries @ keys.transpose(-2, -1) / (self.d_per_head ** 0.5)  
            
            alibi_bias = self.alibi_slope[i] * distance  
            scores += alibi_bias.unsqueeze(0)  
            attention_weights = F.softmax(scores, dim=-1)
            head_output = attention_weights @ values  
            
            all_heads.append(head_output)
        
        multi_head_output = torch.cat(all_heads, dim=-1)  # Shape: [batch_size, seq_len, hidden_size]
        output = self.output_layer(multi_head_output)
        
        if return_attention:
            return output, attention_weights
        else:
            return output




class Self_Attention(nn.Module):
    def __init__(self, hidden_size, n_head=2):
        super(Self_Attention, self).__init__()
        
        torch.manual_seed(123)
        self.n_head = n_head
        self.d_per_head = hidden_size // n_head  # Dimension per head

        self.W_query = nn.Parameter(torch.rand(n_head, self.d_per_head, hidden_size))
        self.W_key = nn.Parameter(torch.rand(n_head, self.d_per_head, hidden_size))
        self.W_value = nn.Parameter(torch.rand(n_head, self.d_per_head, hidden_size))
        
        self.output_layer = nn.Linear(hidden_size, hidden_size)

    def forward(self, x, return_attention=False):
        batch_size, seq_len, hidden_size = x.size()
        all_heads = []
        
        for i in range(self.n_head):
            queries = x.matmul(self.W_query[i].T)  # Shape: [batch_size, seq_len, d_per_head]
            keys = x.matmul(self.W_key[i].T)       # Shape: [batch_size, seq_len, d_per_head]
            values = x.matmul(self.W_value[i].T)   # Shape: [batch_size, seq_len, d_per_head]
            
            scores = queries @ keys.transpose(-2, -1) / (self.d_per_head ** 0.5)
            attention_weights = F.softmax(scores, dim=-1)
            head_output = attention_weights @ values  # Shape: [batch_size, seq_len, d_per_head]
            
            all_heads.append(head_output)
        
        multi_head_output = torch.cat(all_heads, dim=-1) 
        output = self.output_layer(multi_head_output)
        
        if return_attention:
            return output, attention_weights
        else:
            return output  
class FeedForwardNetwork(nn.Module):
    def __init__(self, hidden_size, ffn_dim):
        super(FeedForwardNetwork, self).__init__()
        self.linear1 = nn.Linear(hidden_size, ffn_dim)  
        self.activation = nn.ReLU()
        self.linear2 = nn.Linear(ffn_dim, hidden_size)  

    def forward(self, x):
        x = self.linear1(x)
        x = self.activation(x)
        x = self.linear2(x)
        return x
    
class ResidualConnectionLayerNorm(nn.Module):
    def __init__(self, hidden_size, dropout=0.5):
        super(ResidualConnectionLayerNorm, self).__init__()
        self.layer_norm = nn.LayerNorm(hidden_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, sublayer):
       
        return x + self.dropout(sublayer(self.layer_norm(x)))

class Encoder(nn.Module):
    def __init__(self, input_size, hidden_size, n_layers, n_head, ffn_dim=300, max_len=5000, dropout=0.5):
        super(Encoder, self).__init__()
        self.embedding = nn.Embedding(num_embeddings=input_size, embedding_dim=hidden_size)
        self.positional_encoding = Positional_Encoding(hidden_size, max_len)
        self.self_attention1 = Self_Attention(hidden_size, n_head)
        self.self_attention2 = Self_Attention_Alibi(hidden_size, n_head)
        self.feed_forward = FeedForwardNetwork(hidden_size, ffn_dim)
        self.residual1 = ResidualConnectionLayerNorm(hidden_size, dropout)
        self.residual2 = ResidualConnectionLayerNorm(hidden_size, dropout)
        self.layers = n_layers
        
    def forward(self, x, return_attention=False):
        x = self.embedding(x)  # [batch_size, seq_len, hidden_size]
        ## Comment While Running Alibo 
        x = self.positional_encoding(x)  # Add positional encoding

        attn_maps = []
        for i in range(self.layers):
            ## Comment While Running Alibo 
            _ ,attn_map = self.self_attention1(x, return_attention=True)  
            attn_maps.append(attn_map)
            x = self.residual1(x, self.self_attention1)
            x = self.residual2(x, self.feed_forward)

            ## UnComment While Running Alibo 
            # _ ,attn_map = self.self_attention2(x, return_attention=True)  
            # attn_maps.append(attn_map)
            # x = self.residual1(x, self.self_attention2)
            # x = self.residual2(x, self.feed_forward)

        if return_attention:
            return x, attn_maps  
        return x


# Decoder
class Masked_Self_Attention_ALiBi(nn.Module):
    def __init__(self, hidden_size, n_head=2):
        super(Masked_Self_Attention_ALiBi, self).__init__()
        self.n_head = n_head
        self.d_per_head = hidden_size // n_head
        self.W_query = nn.Parameter(torch.rand(n_head, self.d_per_head, hidden_size))
        self.W_key = nn.Parameter(torch.rand(n_head, self.d_per_head, hidden_size))
        self.W_value = nn.Parameter(torch.rand(n_head, self.d_per_head, hidden_size))
        self.output_layer = nn.Linear(hidden_size, hidden_size)

        self.alibi_slope = -1.0  

    def forward(self, x, mask=None):
        batch_size = 32
        seq_len = 32
        all_heads = []

        distance = torch.arange(seq_len).view(1, -1) - torch.arange(seq_len).view(-1, 1)
        distance = distance.to(x.device)  

        alibi_bias = self.alibi_slope * distance 

        for i in range(self.n_head):

            queries = x.matmul(self.W_query[i].T)  # Shape: [batch_size, seq_len, d_per_head]
            keys = x.matmul(self.W_key[i].T)       # Shape: [batch_size, seq_len, d_per_head]
            values = x.matmul(self.W_value[i].T)   # Shape: [batch_size, seq_len, d_per_head]

            scores = queries @ keys.transpose(-2, -1) / (self.d_per_head ** 0.5)  # Shape: [batch_size, seq_len]
            scores += alibi_bias  # Broadcasts to [batch_size, seq_len]

            if mask is not None:
                scores = scores.masked_fill(mask == 0, float('-inf'))
                
            attention_weights = F.softmax(scores, dim=-1)
            head_output = attention_weights @ values  
            all_heads.append(head_output)

        multi_head_output = torch.cat(all_heads, dim=-1)  
        return self.output_layer(multi_head_output)
    
class Masked_Self_Attention(nn.Module):
    def __init__(self, hidden_size, n_head=2):
        super(Masked_Self_Attention, self).__init__()
        self.n_head = n_head
        self.d_per_head = hidden_size // n_head
        self.W_query = nn.Parameter(torch.rand(n_head, self.d_per_head, hidden_size))
        self.W_key = nn.Parameter(torch.rand(n_head, self.d_per_head, hidden_size))
        self.W_value = nn.Parameter(torch.rand(n_head, self.d_per_head, hidden_size))
        self.output_layer = nn.Linear(hidden_size, hidden_size)

    def forward(self, x, mask=None):
        batch_size, seq_len = x.size()
        all_heads = []

        for i in range(self.n_head):
            queries = x.matmul(self.W_query[i].T)
            keys = x.matmul(self.W_key[i].T)
            values = x.matmul(self.W_value[i].T)

            scores = queries @ keys.transpose(-2, -1) / (self.d_per_head ** 0.5)

            if mask is not None:
                mask = mask[:, :32, :32]

                scores = scores.masked_fill(mask == 0, float('-inf'))

            attention_weights = F.softmax(scores, dim=-1)
            head_output = attention_weights @ values
            all_heads.append(head_output)

        multi_head_output = torch.cat(all_heads, dim=-1)
        return self.output_layer(multi_head_output)

class Encoder_Decoder_Attention(nn.Module):
    def __init__(self, hidden_size, n_head=2):
        super(Encoder_Decoder_Attention, self).__init__()

        self.n_head = n_head
        self.d_per_head = hidden_size // n_head 

        self.W_query = nn.Parameter(torch.rand(n_head, self.d_per_head, hidden_size))
        self.W_key = nn.Parameter(torch.rand(n_head, self.d_per_head, hidden_size))
        self.W_value = nn.Parameter(torch.rand(n_head, self.d_per_head, hidden_size))

        self.output_layer = nn.Linear(hidden_size, hidden_size)

    def forward(self, decoder_input, encoder_output, mask=None, return_attention=False):
        batch_size, seq_len, hidden_size = decoder_input.size()
        _, seq_len_enc, _ = encoder_output.size()
        
        all_heads = []
        all_attention_weights = []

        for i in range(self.n_head):
            queries = decoder_input.matmul(self.W_query[i].T)  # Shape: [batch_size, seq_len, d_per_head]
            keys = encoder_output.matmul(self.W_key[i].T)      # Shape: [batch_size, seq_len_enc, d_per_head]
            values = encoder_output.matmul(self.W_value[i].T)  # Shape: [batch_size, seq_len_enc, d_per_head]

            scores = queries @ keys.transpose(-2, -1) / (self.d_per_head ** 0.5)

            if mask is not None:           
                mask = mask[:, :32, :32]
                scores = scores.masked_fill(mask == 0, float('-inf'))

            # Compute attention weights
            attention_weights = F.softmax(scores, dim=-1)
            all_attention_weights.append(attention_weights)

            head_output = attention_weights @ values  
            all_heads.append(head_output)

        multi_head_output = torch.cat(all_heads, dim=-1)  
        output = self.output_layer(multi_head_output)

        if return_attention:
            return output, torch.stack(all_attention_weights, dim=1)  
        else:
            return output

class FeedForwardNetwork(nn.Module):
    def __init__(self, hidden_size, ffn_dim):
        super(FeedForwardNetwork, self).__init__()
        self.linear1 = nn.Linear(hidden_size, ffn_dim)  
        self.activation = nn.ReLU()
        self.linear2 = nn.Linear(ffn_dim, hidden_size)  

    def forward(self, x):
        x = self.linear1(x)
        x = self.activation(x)
        x = self.linear2(x)
        return x

class ResidualConnectionLayerNorm(nn.Module):
    def __init__(self, hidden_size, dropout=0.5):
        super(ResidualConnectionLayerNorm, self).__init__()
        self.layer_norm = nn.LayerNorm(hidden_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, sublayer):
        normalized_input = self.layer_norm(x)
        sublayer_output = sublayer(normalized_input)
        return x + self.dropout(sublayer_output)

class Decoder(nn.Module):
    def __init__(self, hidden_size, ffn_dim, n_head, dropout=0.5):
        super(Decoder, self).__init__()
        self.decoder_input_projection = nn.Linear(32, 64)
        self.self_attention1 = Masked_Self_Attention(hidden_size, n_head)  # Masked self-attention
        self.self_attention2 = Masked_Self_Attention_ALiBi(hidden_size, n_head)  # Masked self-attention
        self.encoder_decoder_attention = Encoder_Decoder_Attention(hidden_size, n_head)  # Encoder-decoder attention
        self.feed_forward = FeedForwardNetwork(hidden_size, ffn_dim)
        self.residual_connection_self_attention = ResidualConnectionLayerNorm(hidden_size, dropout)
        self.residual_connection_encoder_decoder = ResidualConnectionLayerNorm(hidden_size, dropout)
        self.residual_connection_feed_forward = ResidualConnectionLayerNorm(hidden_size, dropout)

    def forward(self, decoder_input, encoder_output, self_attention_mask):
        decoder_input = decoder_input.float()
        decoder_input = self.decoder_input_projection(decoder_input)
        # Comment While Running Alibo 
        self_attention_output = self.residual_connection_self_attention(decoder_input, 
            lambda x: self.self_attention1(x, mask=self_attention_mask)) 
        ## UnComment While Running Alibo 
        # self_attention_output = self.residual_connection_self_attention(decoder_input, 
        #     lambda x: self.self_attention2(x, mask=self_attention_mask))
        
        # Encoder-decoder attention
        encoder_decoder_output = self.residual_connection_encoder_decoder(self_attention_output, 
            lambda x: self.encoder_decoder_attention(x, encoder_output))
        
        # Feed-forward network
        output = self.residual_connection_feed_forward(self_attention_output, 
            lambda x: self.feed_forward(x))

        return output


