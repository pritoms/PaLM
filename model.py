import torch
import torch.nn as nn
from parallel_transformer import *

# Define a function to return a nn.Sequential model that implements the parallel transformer
def PaLM(n_tokens, d_model, depth, n_heads):
    # Create an embedding layer with the given number of tokens and dimension
    embedding = nn.Embedding(n_tokens, d_model)

    # Create a list of residual blocks, each containing a parallel transformer block, with the given depth
    blocks = [ResidualBlock(d_model) for _ in range(depth)] + [nn.LayerNorm(d_model)]

    # Add a linear projection layer to the end of the list of residual blocks, which projects back into its original dimension using the feedforward output layer, and return it as output of this block
    linear_projection = nn.Linear(d_model, n_tokens)

    # Tie the weights of the embedding layer and linear projection layer together, which is not common, but works well in this case
    linear_projection.weight = embedding.weight

    # Initialize all weights in the embedding layer using normal distribution with standard deviation 0.02
    embedding.weight.data.normal_(mean=0, std=0.02)

    # Return a nn.Sequential model
    return nn.Sequential(embedding, *[ParallelTransformerBlock(d_model, n_heads) for _ in range(depth)], *blocks, linear_projection)
