# Import torch modules
import torch
import torch.nn as nn
import torch.nn.functional as F

# Import EinOps
from einops import rearrange, reduce

# Define a layer norm module with EinOps-style parameters
class LayerNorm(nn.Module):
    def __init__(self, d_model, eps=1e-6):
        super().__init__()
        self.gamma = nn.Parameter(torch.ones(d_model))
        self.beta = nn.Parameter(torch.zeros(d_model))
        self.eps = eps

    def forward(self, x):
        mean = rearrange(x, 'b d ... -> b ... d', d=0)
        std = torch.sqrt(rearrange((x - mean) ** 2, 'b d ... -> b ... d', d=0) + self.eps)
        return self.gamma * (x - mean) / std + self.beta

# Define a residual block module with EinOps-style parameters
class ResidualBlock(nn.Module):
    def __init__(self, d_model, dropout=0.1):
        super().__init__()
        self.norm_1 = LayerNorm(d_model)
        self.norm_2 = LayerNorm(d_model)
        self.dropout_1 = nn.Dropout(dropout)
        self.dropout_2 = nn.Dropout(dropout)

    def forward(self, x):
        return x + self.dropout_2(self.norm_2(F.relu(self.dropout_1(self.norm_1(x)))))

# Define a rotary embedding module with EinOps-style parameters
class RotaryEmbedding(nn.Module):
    def __init__(self, d_model, dropout=0.1, n_positions=1024):
        super().__init__()
        self.w = nn.Parameter(torch.empty(n_positions, d_model).normal_(mean=0, std=d_model ** -0.5))
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.dropout(self.w[:x.shape[-1], :])

# Define a function to rotate half of the last dimension of a tensor
def rotate_half(x):
    return torch.cat([x[..., int(x.shape[-1] / 2):], x[..., :int(x.shape[-1] / 2)]], dim=-1)

# Define a function to apply positional embeddings to the input tensor using the rotary embedding module and the rotation function
def positional_embeddings(x, r, rotate):
    pe = torch.cat([r(x), rotate(r(x))], dim=-1)
    return rearrange(pe, 'b d p ... -> b p ... d', d=0)

# Define a Swish-Gated Linear Unit (SwiGLU) module, which is a variant of Swish that applies its gate only to the linear component of the input tensor
class SwiGLU(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return x * torch.sigmoid(x)

# Define a ParallelTransformerBlock module
class ParallelTransformerBlock(nn.Module):
    def __init__(self, d_model, n_heads, dropout=0.1):
        super().__init__()
        self.norm_1 = LayerNorm(d_model)
        self.norm_2 = LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

        # Calculate the inner dimensions of the attention and feedforward layers
        self.attention_inner_dim = d_model // n_heads
        self.feedforward_inner_dim = d_model * 4

        # Initialize the rotary embedding layer, which is used to generate positional embeddings for each head
        self.rotary_embedding = RotaryEmbedding(self.attention_inner_dim, dropout=dropout)

        # Initialize the fused attention and feedforward projection layer, which is used to project the input into the inner dimensions of the attention and feedforward layers
        self.fused_projection = nn.Linear(d_model, self.attention_inner_dim * n_heads * 3 + self.feedforward_inner_dim)

        # Initialize the attention output layer, which is used to project the output of the attention layer back into its original dimension
        self.attention_output = nn.Linear(self.attention_inner_dim * n_heads, d_model)

        # Initialize the feedforward output layer, which is used to project the output of the feedforward layer back into its original dimension
        self.feedforward_output = nn.Linear(self.feedforward_inner_dim, d_model)

    # Define a function to return a causal mask that is used to mask out all positions that are not in the causal region of each position in a sequence
    def get_mask(self, x):
        return rearrange(x, 'b p ... -> b ... p', p=0) != rearrange(x, 'b p ... -> b ... p', p=0).new_ones(x.shape[1]).cumsum(dim=-1) - 1

    # Define a function to return a rotary embedding that is used to generate positional embeddings for each head
    def get_rotary_embedding(self, x):
        return rearrange(self.rotary_embedding(x), 'b d p ... -> b p ... d', d=0)

    # Define the forward pass
    def forward(self, x):
        # Project the input into its inner dimensions using the fused attention and feedforward projection layer
        fused_projection = self.fused_projection(x)

        # Split it into its attention queries, keys, values, and feedforward inner tensors
        attention_queries, attention_keys, attention_values, feedforward = rearrange(fused_projection, 'b (d p) (d p) (d p) f -> b p (d h) (d h) (d h) f', d=0, p=1, f=-1, h=self.attention_inner_dim), rearrange(fused_projection, 'b (d p) (d p) (d p) f -> b p (d h) (d h) (d h) f', d=0, p=2, f=-1, h=self.attention_inner_dim), rearrange(fused_projection, 'b (d p) (d p) (d p) f -> b p (d h) (d h) (d h) f', d=0, p=3, f=-1, h=self.attention_inner_dim), rearrange(fused_projection, 'b (d p) (d p) (d p) f -> b f', d=0, p=0, f=-1)

        # Split the attention queries into its heads
        attention_queries = rearrange(attention_queries, 'b p h -> b h p', h=0)

        # Generate the rotary embeddings for each head
        rotary_embedding = self.get_rotary_embedding(attention_queries[:, 0])

        # Apply the rotary embeddings to the attention queries and keys
        attention_queries = positional_embeddings(attention_queries, self.rotary_embedding, rotate_half)
        attention_keys = positional_embeddings(attention_keys, self.rotary_embedding, rotate_half)

        # Scale the attention queries by the square root of the dimension of each head
        attention_queries = attention_queries / rearrange(attention_queries, 'b h p ... -> b h ...', h=0).new_tensor(self.attention_inner_dim ** 0.5)

        # Calculate the similarity between each query and key using a dot product
        similarity = rearrange(attention_queries, 'b h p ... -> b p h ...', p=1) @ rearrange(attention_keys, 'b h p ... -> b p h ...', p=1).transpose(-2, -1)

        # Mask out all positions that are not in the causal region of each position in a sequence using a causal mask
        mask = self.get_mask(similarity)
        similarity[mask] = -float('inf')

        # Apply softmax to the similarity to get attention weights for each query and key pair
        attention_weights = F.softmax(similarity, dim=-1)

        # Aggregate the values using the attention weights for each query and key pair
        attention_values = (attention_weights @ attention_values).transpose(-2, -1).contiguous()

        # Merge the heads back into one tensor and project it back into its original dimension using the attention output layer
        attention = self.attention_output(attention_values)

        # Add it with the output of the feedforward layer, which is projected back into its original dimension using the feedforward output layer, and return it as output of this block
        return self.norm_2(attention + self.dropout(SwiGLU()(self.feedforward_output(F.relu(self.norm_1(feedforward))))))
