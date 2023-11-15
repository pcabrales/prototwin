import os
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
from monai import transforms
from monai.config import print_config
from monai.utils import first, set_determinism
from torch import nn
from torch.cuda.amp import GradScaler, autocast
from tqdm import tqdm
from generative.networks.nets import AutoencoderKL, DiffusionModelUNet, PatchDiscriminator
from generative.networks.schedulers import DDPMScheduler


from torch.utils.data import DataLoader
from torchvision.transforms import Compose, Normalize
from utils import set_seed, DoseActivityDataset, JointCompose, Resize3D

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using {device}")