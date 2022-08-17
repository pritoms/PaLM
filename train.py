import torch
import torch.nn as nn
from model import *
from utils import *
import time
from tqdm import tqdm
from tensorboardX import SummaryWriter

# Set random seed for reproducibility
torch.manual_seed(0)

# Define constants for the dataset and model hyperparameters, and training hyperparameters
DATA_PATH = 'data' # Path to the dataset file, which should be a text file containing one sentence per line, with no special characters except spaces and newline characters, and no empty lines
MODEL_PATH = 'model.pt' # Path to the model file, which will be used to save and load the model during training and testing, respectively, and will be created if it does not exist yet
D_MODEL = 256 # Dimension of each token in the dataset file, which is used to initialize the embedding layer and linear projection layer in the model
DEPTH = 12 # Number of residual blocks in the model, which is also the number of parallel transformer blocks in the model because each residual block contains one parallel transformer block
N_HEADS = 8 # Number of heads in each parallel transformer block in the model
BATCH_SIZE = 256 # Batch size used during training, which is also the maximum length of a sentence in the dataset file
LR = 0.00025 # Learning rate used during training
SEQ_LEN = 256 # Sequence length used during training, which is also the number of positions that each sentence in the dataset file will be padded or truncated to
INIT_SCALE = 0.1 # Scale used to initialize all weights in the model using normal distribution with standard deviation INIT_SCALE / D_MODEL ** 0.5
N_EPOCHS = 100 # Number of epochs used during training
N_STEPS_PER_EPOCH = 100 # Number of steps to be taken per epoch during training, which is also the number of iterations to be taken per epoch during training because each step contains one iteration
N_STEPS_PER_EVAL = 20 # Number of steps to be taken per evaluation during training, which is also the number of iterations to be taken per evaluation during training because each step contains one iteration
CLIP_GRAD = 0.1 # Maximum absolute value of the gradient of all weights in the model, which will be clipped if it exceeds this value
USE_CUDA = True # Flag that indicates whether CUDA should be used or not, which will determine whether the model and dataset will be put on GPU or not

# Initialize a SummaryWriter object to write training logs to the tensorboard log directory
log_writer = SummaryWriter('logs/' + str(int(time.time())))

# Initialize a nn.CrossEntropyLoss object to calculate the cross entropy loss between the output of the linear projection layer in the model and its target, which is also a smoothed version of the target using label smoothing
loss_function = nn.CrossEntropyLoss(ignore_index=0, reduction='mean', smoothing=0.1)

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

# Initialize an optimizer to optimize all weights in the model using Adam with learning rate LR, and initialize a scheduler to decrease learning rate by 10 times every N_EPOCHS // 3 epochs starting from the second epoch until the end of training
optimizer = torch.optim.Adam(model.parameters(), lr=LR)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=N_EPOCHS // 3, gamma=0.1)

# Start training
for epoch in range(N_EPOCHS):
    # Begin each epoch by putting the model in training mode
    model.train()

    # Initialize a list of losses to be calculated for each step in this epoch
    losses = []

    # Initialize a tqdm progress bar to show the progress of this epoch during training
    with tqdm(total=N_STEPS_PER_EPOCH) as pbar:
        # Create N_STEPS_PER_EPOCH steps per epoch during training, which also corresponds to N_STEPS_PER_EPOCH iterations per epoch during training because each step contains one iteration
        for step in range(N_STEPS_PER_EPOCH):
            # Zero the gradients of all weights in the model at the beginning of each step
            optimizer.zero_grad()

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
            loss = loss_function(output.reshape(-1, output.shape[-1]), batch.flatten())

            # Backpropagate through all weights in the model to calculate their gradients using this loss
            loss.backward()

            # Clip all gradients of all weights in the model using CLIP_GRAD if they exceed CLIP_GRAD, then update all weights using their gradients
            nn.utils.clip_grad_norm_(model.parameters(), CLIP_GRAD)
            optimizer.step()

            # Append this loss to the list of losses
            losses.append(loss.item())

            # Update the progress bar with this loss and the learning rate in this step
            pbar.set_postfix(loss=f'{sum(losses) / len(losses):.3f}', lr=f'{optimizer.param_groups[0]["lr"]:.6f}')
            pbar.update()

    # Calculate and log the average loss across all steps in this epoch to the tensorboard log directory
    log_writer.add_scalar('loss', sum(losses) / len(losses), epoch)

    # Decrease the learning rate by 10 times using the scheduler at the end of each epoch
    scheduler.step()

    # Evaluate the model after N_STEPS_PER_EPOCH steps per epoch, which also corresponds to N_STEPS_PER_EPOCH iterations per epoch during training because each step contains one iteration
    if (epoch + 1) % N_STEPS_PER_EVAL == 0:
        # Put the model on evaluation mode
        model.eval()

        # Initialize a list of losses to be calculated for each step in this evaluation
        losses = []

        # Initialize a tqdm progress bar to show the progress of this evaluation during training
        with tqdm(total=N_STEPS_PER_EVAL) as pbar:
            # Create N_STEPS_PER_EVAL steps per evaluation during training, which also corresponds to N_STEPS_PER_EVAL iterations per evaluation during training because each step contains one iteration
            for step in range(N_STEPS_PER_EVAL):
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
                loss = loss_function(output.reshape(-1, output.shape[-1]), batch.flatten())

                # Append this loss to the list of losses
                losses.append(loss.item())

                # Update the progress bar with this loss
                pbar.set_postfix(loss=f'{sum(losses) / len(losses):.3f}')
                pbar.update()

        # Calculate and log the average loss across all steps in this evaluation to the tensorboard log directory
        log_writer.add_scalar('eval loss', sum(losses) / len(losses), epoch)
