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
from utils import set_seed, DoseActivityDataset, plot_slices, plot_ddp, GaussianBlurFloats, back_and_forth, JointCompose, Resize3D
seed = 42
set_seed(seed)
import torch
from torch.utils.data import DataLoader
from torchvision.transforms import Compose, Normalize
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


img_size = (128, 64, 64)
joint_transform = JointCompose([
    Resize3D(img_size)
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
num_workers = 2
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

from models.SwinUNETR import SwinUNETR
# Create the model
patches = False
model = SwinUNETR(img_size=img_size, in_channels=1, out_channels=1, feature_size=12).to(device)

model_dir = 'models/trained-models/SwinUNETR-v4.pth'
timing_dir = 'models/training-times/training-time-SwinUNETR-v4.txt'
n_epochs = 60
save_plot_dir = "images/SwinUNETR-v4-loss.png"
trained_model = train(model, train_loader, val_loader, epochs=n_epochs,
                      model_dir=model_dir, timing_dir=timing_dir, save_plot_dir=save_plot_dir)

# Loading the trained model
model_dir = "models/trained-models/SwinUNETR-v4.pth"
trained_model = torch.load(model_dir, map_location=torch.device(device))

plot_loader = test_loader

# Plotting slices of the dose
plot_slices(trained_model, plot_loader, device, mean_input=mean_input, std_input=std_input,
            mean_output=mean_output, std_output=std_output,
            save_plot_dir = "images/SwinUNETR-v4-sample.png", y_slice=img_size[2]//2, patches=patches, patch_size=img_size[2]//2) 
 
# Plotting the dose-depth profiles
save_plot_dir = "images/SwinUNETR-v4-ddp.png"
plot_ddp(trained_model, plot_loader, device, mean_output=mean_output,
         std_output=std_output, save_plot_dir=save_plot_dir, patches=patches, patch_size=img_size[2]//2)

results_dir = 'models/test-results/SwinUNETR-v4-results.txt'
test(trained_model, test_loader, device, results_dir=results_dir, mean_output=mean_output, std_output=std_output)


# dose2act_model_dir = "models/trained-models/SwinUNETR-v4.pth"
# dose2act_model = torch.load(dose2act_model_dir, map_location=torch.device(device))
# act2dose_model_dir = "models/trained-models/SwinUNETR-v2.pth"
# act2dose_model = torch.load(act2dose_model_dir, map_location=torch.device(device))
# back_and_forth(dose2act_model, act2dose_model, test_loader, device, reconstruct_dose=False, num_cycles=3, y_slice=32, 
#                mean_act=mean_input, std_act=std_input, mean_dose=mean_output, std_dose=std_output, save_plot_dir="images/reconstructed_act.png")