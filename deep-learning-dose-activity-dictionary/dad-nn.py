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
from utils import set_seed, DoseActivityDataset, plot_slices, plot_ddp, DoseActivityDatasetReshaped ###
seed = 42
set_seed(seed)
import torch
from torch.utils.data import DataLoader
from torchvision.transforms import Compose, Normalize#, RandAugment, RandomRotation, GaussianBlur, RandomHorizontalFlip, RandomVerticalFlip
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
    # GaussianBlur(kernel_size=3, sigma=1.0),
    Normalize(mean_input, std_input)
])

output_transform = Compose([
    Normalize(mean_output, std_output)
])

# joint_transform = Compose([
#     RandomRotation(5),
#     RandAugment(magnitude=2),
#     RandomHorizontalFlip(),
#     RandomVerticalFlip()
# ])

# Create dataset applying the transforms
dataset = DoseActivityDatasetReshaped(input_dir=input_dir, output_dir=output_dir,
                              input_transform=input_transform, output_transform=output_transform,
                              num_samples=948, size_reshape=64)

# Split dataset into 70% training, 20% validation, 10% testing
train_size = int(0.7 * len(dataset))
val_size = int(0.2 * len(dataset))
test_size = len(dataset) - train_size - val_size
train_dataset, val_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, val_size, test_size])

# Create DataLoaders for training
batch_size = 8  # Largest batch size without running out of memory
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=2)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=2)


from models.TransBTS import TransBTS
# Create the model
model = TransBTS().to(device)

model_dir = 'models/trained-models/TransBTS-v1.pth'
timing_dir = 'models/training-times/training-time-TransBTS-v1.txt'
n_epochs = 100
save_plot_dir = "images/TransBTS-v1-loss.png"
trained_model = train(model, train_loader, val_loader, epochs=n_epochs,
                      model_dir=model_dir, timing_dir=timing_dir, save_plot_dir=save_plot_dir,
                      mean_output=mean_output, std_output=std_output)

# Loading the trained model
model_dir = "models/trained-models/TransBTS-v1.pth"
trained_model = torch.load(model_dir, map_location=torch.device(device))

# Plotting slices of the dose
plot_slices(trained_model, val_loader, device, mean_input=mean_input, std_input=std_input,
            mean_output=mean_output, std_output=std_output,
            save_plot_dir = "images/TransBTS-v1-sample.png", y_slice = 32)

# Plotting the dose-depth profiles
save_plot_dir = "images/TransBTS-v1-ddp.png"
plot_ddp(trained_model, train_loader, device, mean_output=mean_output,
         std_output=std_output, save_plot_dir=save_plot_dir, y_slice = 32)

results_dir = 'models/test-results/TransBTS-v1-results.txt'
test(trained_model, test_loader, device, results_dir=results_dir, mean_output=mean_output, std_output=std_output)