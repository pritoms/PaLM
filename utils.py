import torch
import torch.nn as nn
from collections import Counter
import re
import os
torch.manual_seed(0)

# Define a function to return the number of tokens in a text file, which is also the number of classes in the language model, by counting the number of unique tokens in the file, excluding special characters and empty lines
def get_n_tokens(path):
    # Remove all special characters except spaces and newline characters from each line in the file using regular expression, then remove all empty lines from the file, then count the number of unique tokens in it, then add one to it because we will be using 0 as an index for padding tokens, and return it as the number of tokens in the file
    return len(Counter(re.sub('[^\s\n]+', '', open(path).read()).split('\n'))) + 1

# Define a class to load a text file in batches of BATCH_SIZE, which will be used to generate sentences with length SEQ_LEN during training and testing, respectively, and will be put on GPU if USE_CUDA is True, otherwise it will be put on CPU
class TextDataLoader:
    def __init__(self, path, batch_size, seq_len, use_cuda):
        # Initialize a list of batches to be returned from this class
        self.batches = []

        # Initialize a list of tokens in the file by removing all special characters except spaces and newline characters from each line in the file using regular expression, then remove all empty lines from the file, then convert it into a list of tokens
        self.tokens = re.sub('[^\s\n]+', '', open(path).read()).split('\n')

        # Initialize an index that points to the current token in self.tokens
        self.i = 0

        # Initialize a list of indices that points to the current position in each batch in self.batches
        self.b = [0] * batch_size

        # Initialize a list of buffers that stores the current sentence in each batch in self.batches, with length SEQ_LEN
        self.buffers = [[0] * seq_len for _ in range(batch_size)]

        # Initialize a flag that indicates whether the end of the file has been reached or not, which will be used to stop generating more batches
        self.end = False

        # Initialize a list of indices that points to the current position in each buffer in self.buffers
        self.j = [0] * batch_size

        # Initialize a list of masks that are used to mask out all positions that are not in the causal region of each position in a sequence
        self.mask = [[1 if i <= j else 0 for i in range(seq_len)] for j in range(seq_len)]

    # Define a function to return the next batch from this class
    def next_batch(self):
        # If the end of the file has not been reached yet, then generate the next batch by iterating through all tokens in self.tokens until it runs out of tokens in self.tokens or it generates one batch, and return it as output
        if not self.end:
            while self.i < len(self.tokens) and sum(self.b) < len(self.b):
                for batch_i in range(len(self.b)):
                    # If the current position in this batch is less than SEQ_LEN, then append the current token in self.tokens to the current sentence in this buffer, then increment the position in this buffer
                    if self.b[batch_i] < len(self.buffers[0]):
                        self.buffers[batch_i][self.j[batch_i]] = int(self.tokens[self.i])
                        self.j[batch_i] = min(self.j[batch_i] + 1, len(self.buffers[0]) - 1)

                    # If the current position in this batch has reached SEQ_LEN, then append the mask for this batch to the list of batches, then reset the position and buffer for this batch
                    if self.b[batch_i] == len(self.buffers[0]):
                        self.batches += [self.mask]
                        self.b[batch_i] = 0
                        self.buffers[batch_i] = [0] * len(self.buffers[0])
                        self.j[batch_i] = 0

                    # Increment the position in this batch
                    self.b[batch_i] += 1

                # Increment the index that points to the current token in self.tokens
                self.i += 1

            # If the end of the file has been reached, then append the mask for each batch to the list of batches, then set the flag that indicates whether the end of the file has been reached or not to True
            if self.i >= len(self.tokens):
                self.batches += [self.mask] * len(self.b)
                self.end = True

        # Return the next batch from the list of batches
        return torch.stack(self.batches[:len(self.b)])
