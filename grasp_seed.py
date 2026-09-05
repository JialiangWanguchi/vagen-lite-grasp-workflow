"""Imported in VERL actor processes before LoRA initialization."""
import random
import numpy as np
import torch

from experiment_config import CFG
SEED=CFG['seed']
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
print(f'GRASP_INITIALIZATION_SEED {SEED}',flush=True)
