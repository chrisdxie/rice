import torch

# Label stuff
BACKGROUND_LABEL = 0
TABLE_LABEL = 1
OBJECTS_LABEL = 2

# PyTorch stuff
DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

# Tensorboard stuff
BASE_TENSORBOARD_DIR = ''  # TODO: Change this hard-coded path to the appropriate directory

# Graph node status
NODE_STATUS_SPLIT = 1
NODE_STATUS_ADDED = 2