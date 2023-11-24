### ONLY FOR GOOGLE COLAB
# Mounting google drive
# from google.colab import drive
# drive.mount('/content/drive')
# import os
# os.chdir("drive/MyDrive/Colab Notebooks/prototwin/deep-learning-dose-activity-dictionary")
# !pip install livelossplot
###

from train_model import train
from test_model import test
from utils import set_seed, DoseActivityDataset, plot_slices, plot_ddp, back_and_forth, JointCompose, Resize3D, GaussianBlob, ResizeCropAndPad3D
seed = 42
set_seed(seed)
import os
import json
import torch
import numpy as np
from torch.utils.data import DataLoader
from torchvision.transforms import Compose, Normalize
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device, " : ", torch.cuda.get_device_name(torch.cuda.current_device()))

# Creating the dataset
input_dir = "data/dataset_1/input"
output_dir = "data/dataset_1/output"
dataset_dir = "data/dataset_1"

# Statistics of the dataset (previously found for the entire Prostate dataset)
mean_input = 0.002942
std_input = 0.036942
max_input = 1.977781
min_input = 0.0

mean_output = 0.00000057475
std_output = 0.00000662656
max_output = 0.00060621166
min_output = 0.0

mean_CT = 65.3300
std_CT = 170.0528
max_CT = 1655
min_CT = -1000

###
mean_output = 0
std_output = max_output
mean_input = 0
std_input = max_input
###
    
# Load the dictionary from the JSON file
with open(os.path.join(dataset_dir, 'energy_beam_dict.json'), 'r') as file:
    energy_beam_dict = json.load(file)

# Reshape of the images
img_size = (160, 64, 64)

# Loading the CT
from scipy.ndimage import zoom
large_CT = np.load ('data/dataset_1/CT.npy')

# Displacement of the center for each dimension (Notation consistent with TOPAS)
TransX = -20 ###-15
TransY= 0
TransZ = -10
HLX = img_size[0] // 2
HLY = img_size[1]
HLZ = img_size[2]

cropped_CT = large_CT[large_CT.shape[0]//2 + TransX - HLX : large_CT.shape[0]//2 + TransX + HLX,
                      large_CT.shape[1]//2 + TransY - HLY : large_CT.shape[1]//2 + TransY + HLY,
                      large_CT.shape[2]//2 + TransZ - HLZ : large_CT.shape[2]//2 + TransZ + HLZ]

CT = zoom(cropped_CT, (img_size[0] / cropped_CT.shape[0], img_size[1] / cropped_CT.shape[1], img_size[2] / cropped_CT.shape[2]))

# CT = np.flip(CT, axis=0)  # Flipping dim=0 because we have to? Not sure
CT_flag = False
if CT_flag: 
    in_channels = 2
    CT = (CT - mean_CT) / std_CT  # Normalise
else: in_channels = 1

# Transformations
input_transform = Compose([
    Normalize(mean_input, std_input)
])

output_transform = Compose([
    Normalize(mean_output, std_output)
])

joint_transform = JointCompose([
    # Resize3D(img_size)
    ResizeCropAndPad3D(img_size)
])


set_seed(seed)

num_samples = 948
# Create dataset applying the transforms
dataset = DoseActivityDataset(input_dir=input_dir, output_dir=output_dir,
                              input_transform=input_transform, output_transform=output_transform, joint_transform=joint_transform,
                              CT_flag=CT_flag, CT=CT, num_samples=num_samples, energy_beam_dict=energy_beam_dict)

# Split dataset into 80% training, 15% validation, 5% testing
train_size = int(0.8 * len(dataset))
val_size = int(0.15 * len(dataset))
test_size = len(dataset) - train_size - val_size
train_dataset, val_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, val_size, test_size])

# Create DataLoaders for training
batch_size = 1
num_workers = 1
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

# Create the model
patches = False
from models.SwinUNETR import SwinUNETR
model = SwinUNETR(in_channels=in_channels, out_channels=1, img_size=img_size).to(device)
# from models.TransBTS import TransBTS
# model = TransBTS(img_dim=img_size).to(device)
# from models.models import UNetV13
# model = UNetV13().to(device)


model_dir = 'models/trained-models/unet-v14.pth'
timing_dir = 'models/training-times/training-time-unet-v14.txt'
losses_dir = 'models/losses/unet-v14-loss.csv'
n_epochs = 50
save_plot_dir = "images/unet-v14-loss.png"
# trained_model = train(model, train_loader, val_loader, epochs=n_epochs, mean_output=mean_output, std_output=std_output,
#                       model_dir=model_dir, timing_dir=timing_dir, save_plot_dir=save_plot_dir, losses_dir=losses_dir)

# Loading the trained model
model_dir = "models/trained-models/unet-v14.pth"
trained_model = torch.load(model_dir, map_location=torch.device(device))

###
# input_transform = Compose([
#     GaussianBlob(30, 5),
#     Normalize(mean_input, std_input)
# ])

# output_transform = Compose([
#     Normalize(mean_output, std_output)
# ])
# joint_transform = JointCompose([
#     Resize3D(img_size)
# ])

# blob_dataset = DoseActivityDataset(input_dir=input_dir, output_dir=output_dir,
#                               input_transform=input_transform, output_transform=output_transform, joint_transform=joint_transform,
#                               num_samples=30, CT=CT)

# plot_loader = DataLoader(blob_dataset, batch_size=batch_size, shuffle=True, num_workers=1)
# test_loader = plot_loader
###
plot_loader = test_loader

# Plotting slices of the dose
save_plot_dir = "images/unet-v14-sample.png"
plot_slices(trained_model, plot_loader, device, CT_flag=CT_flag, CT_manual=CT, 
            mean_input=mean_input, std_input=std_input, mean_output=mean_output, std_output=std_output,
            save_plot_dir=save_plot_dir, patches=patches) 
 
# Plotting the dose-depth profiles
save_plot_dir = "images/unet-v14-ddp.png"
plot_ddp(trained_model, plot_loader, device, mean_output=mean_output,
         std_output=std_output, save_plot_dir=save_plot_dir, patches=patches, patch_size=img_size[2]//2)

results_dir = 'models/test-results/unet-v14-results.txt'
save_plot_dir = 'images/unet-v14-range-hist.png'
test(trained_model, test_loader, device, results_dir=results_dir, mean_output=mean_output, std_output=std_output, save_plot_dir=save_plot_dir)

# dose2act_model_dir = "models/trained-models/reverse-SwinUNETR-v1.pth"
# dose2act_model = torch.load(dose2act_model_dir, map_location=torch.device(device))
# act2dose_model_dir = "models/trained-models/unet-v14.pth"
# act2dose_model = torch.load(act2dose_model_dir, map_location=torch.device(device))
# back_and_forth(dose2act_model, act2dose_model, plot_loader, device, reconstruct_dose=False, num_cycles=1, y_slice=32, 
#                mean_act=mean_input, std_act=std_input, mean_dose=mean_output, std_dose=std_output, save_plot_dir="images/reconstructed_act_blob_1cycle.png")