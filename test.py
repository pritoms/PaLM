import torch
import torch.nn as nn
from model import *
from utils import *
import time
from tqdm import tqdm

# Set random seed for reproducibility
torch.manual_seed(0)

# Define constants for the dataset and model hyperparameters, and training hyperparameters
DATA_PATH = 'data' # Path to the dataset file, which should be a text file containing one sentence per line, with no special characters except spaces and newline characters, and no empty lines
MODEL_PATH = 'model.pt' # Path to the model file, which will be used to load the model during testing, and will be created if it does not exist yet
D_MODEL = 256 # Dimension of each token in the dataset file, which is used to initialize the embedding layer and linear projection layer in the model
DEPTH = 12 # Number of residual blocks in the model, which is also the number of parallel transformer blocks in the model because each residual block contains one parallel transformer block
N_HEADS = 8 # Number of heads in each parallel transformer block in the model
BATCH_SIZE = 256 # Batch size used during testing, which is also the maximum length of a sentence in the dataset file
SEQ_LEN = 256 # Sequence length used during testing, which is also the number of positions that each sentence in the dataset file will be padded or truncated to
USE_CUDA = True # Flag that indicates whether CUDA should be used or not, which will determine whether the model and dataset will be put on GPU or not

# Initialize a PaLM Language Model with the given hyperparameters
model = PaLM(get_n_tokens(DATA_PATH), D_MODEL, DEPTH, N_HEADS)

# Load the model from MODEL_PATH if it exists, otherwise save it to MODEL_PATH for later use during testing
try:
    model.load_state_dict(torch.load(MODEL_PATH))
except FileNotFoundError:
    torch.save(model.state_dict(), MODEL_PATH)

# Put the model on GPU if USE_CUDA is True, otherwise put it on CPU
if USE_CUDA:
    model.cuda()
else:
    model.cpu()

# Initialize a DataLoader object to load the dataset file in batches of BATCH_SIZE, which will be used to generate sentences with length SEQ_LEN during training and testing, respectively, and will be put on GPU if USE_CUDA is True, otherwise it will be put on CPU
data = TextDataLoader(DATA_PATH, BATCH_SIZE, SEQ_LEN, USE_CUDA)

# Put the model on evaluation mode
model.eval()

# Initialize a list of losses to be calculated for each step in this evaluation
losses = []

# Initialize a tqdm progress bar to show the progress of this evaluation during testing
with tqdm(total=len(data)) as pbar:
    # Evaluate the model over all batches in the dataset until it runs out of batches
    while True:
        try:
            # Generate a batch of sentences with length SEQ_LEN to be used as input to the model and target for the linear projection layer in the model
            batch = data.next_batch()

            # Put the batch on GPU if USE_CUDA is True, otherwise put it on CPU
            if USE_CUDA:
                batch = batch.cuda()
            else:
                batch = batch.cpu()

            # Pass the input through the model to get the output of the linear projection layer in the model
            output = model(batch)

            # Calculate the cross entropy loss between the output and its target, then calculate its mean across all batches
            loss = nn.CrossEntropyLoss(ignore_index=0, reduction='mean', smoothing=0.1)(output.reshape(-1, output.shape[-1]), batch.flatten()).item()

            # Append this loss to the list of losses
            losses.append(loss)

            # Update the progress bar with this loss
            pbar.set_postfix(loss=f'{sum(losses) / len(losses):.3f}')
            pbar.update()

        except StopIteration:
            # Break out of the while loop when it runs out of batches in the dataset
            break
