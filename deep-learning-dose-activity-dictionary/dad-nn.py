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
from utils import set_seed, DoseActivityDataset, plot_slices, plot_ddp, GaussianBlurFloats, Random3DCrop, JointCompose, Resize3D
seed = 42
set_seed(seed)
import torch
from torch.utils.data import DataLoader
from torchvision.transforms import Compose, Normalize, RandomHorizontalFlip, RandomVerticalFlip
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Creating the dataset
input_dir = "data/dataset_1/input"
output_dir = "data/dataset_1/output"

# Statistics of the dataset (previously found for the entire Prostate dataset)
mean_input = 0.002942
std_input = 0.036942
max_input = 1.977781
min_input = 0.0

mean_output = 0.00000057475
std_output = 0.00000662656
max_output = 0.00060621166
min_output = 0.0


# Transformations
input_transform = Compose([
    GaussianBlurFloats(p=0.3, sigma=1),
    Normalize(mean_input, std_input)
])

output_transform = Compose([
    Normalize(mean_output, std_output)
])


patch_size = 64
joint_transform = JointCompose([
    RandomHorizontalFlip(p=0.3),
    RandomVerticalFlip(p=0.3),
    # Random3DCrop(56)
    Resize3D((patch_size, patch_size, patch_size))
])
set_seed(seed)

# Create dataset applying the transforms
dataset = DoseActivityDataset(input_dir=input_dir, output_dir=output_dir,
                              input_transform=input_transform, output_transform=output_transform, joint_transform=joint_transform,
                              num_samples=948)

# Split dataset into 80% training, 15% validation, 5% testing
train_size = int(0.8 * len(dataset))
val_size = int(0.15 * len(dataset))
test_size = len(dataset) - train_size - val_size
train_dataset, val_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, val_size, test_size])

# Create DataLoaders for training
batch_size = 8  # Largest batch size without running out of memory
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=2)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=2)

from models.SwinUNETR import SwinUNETR#TransBTS
# Create the model
patches = False
model = SwinUNETR(patch_size, 1, 1, feature_size=12).to(device)
###
# input_batch,_ = next(iter(train_loader))
# output_batch = model(input_batch)
# print(output_batch.shape)
# stop
###

model_dir = 'models/trained-models/SwinUNETR-v1.pth'
timing_dir = 'models/training-times/training-time-SwinUNETR-v1.txt'
n_epochs = 1
save_plot_dir = "images/SwinUNETR-v1-loss.png"
# trained_model = train(model, train_loader, val_loader, epochs=n_epochs, ###
#                       model_dir=model_dir, timing_dir=timing_dir, save_plot_dir=save_plot_dir)

# Loading the trained model
model_dir = "models/trained-models/SwinUNETR-v1.pth"
trained_model = torch.load(model_dir, map_location=torch.device(device))

###
# input_transform = Compose([
#     Normalize(mean_input, std_input)
# ])
# output_transform = Compose([
#     Normalize(mean_output, std_output)
# ])

# # Create dataset without cropping
# num_samples_plot = 3
# plotting_set = DoseActivityDataset(input_dir=input_dir, output_dir=output_dir, 
#                                    input_transform=input_transform, output_transform=output_transform, 
#                                    num_samples=num_samples_plot)
# plot_loader = DataLoader(plotting_set, batch_size=num_samples_plot, shuffle=False, num_workers=0)
plot_loader = test_loader
###

# Plotting slices of the dose
plot_slices(trained_model, plot_loader, device, mean_input=mean_input, std_input=std_input,
            mean_output=mean_output, std_output=std_output,
            save_plot_dir = "images/SwinUNETR-v1-sample.png", y_slice=patch_size//2, patches=patches, patch_size=patch_size) 
 
# Plotting the dose-depth profiles
save_plot_dir = "images/SwinUNETR-v1-ddp.png"
plot_ddp(trained_model, plot_loader, device, mean_output=mean_output,
         std_output=std_output, save_plot_dir=save_plot_dir, patches=patches, patch_size=patch_size)

results_dir = 'models/test-results/SwinUNETR-v1-results.txt'
test(trained_model, test_loader, device, results_dir=results_dir, mean_output=mean_output, std_output=std_output)
