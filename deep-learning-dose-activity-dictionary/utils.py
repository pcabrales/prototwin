import random
import torch
import numpy as np
import os
from torch.utils.data import Dataset
from scipy.ndimage import gaussian_filter
from scipy.interpolate import interp1d
import seaborn as sns 
import matplotlib.pyplot as plt

def set_seed(seed):
    """
    Set all the random seeds to a fixed value to take out any randomness
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.enabled = False
    return True


# Creating a dataset for the dose/activity input/output pairs
class DoseActivityDataset(Dataset):
    """
    Create the dataset where the activity is the input and the dose is the output.
    The relevant transforms are applied.
    """
    def __init__(self, input_dir, output_dir, num_samples=5, input_transform=None, output_transform=None, joint_transform=None):
        self.input_dir = input_dir
        self.output_dir = output_dir
        self.input_transform = input_transform
        self.output_transform = output_transform
        self.joint_transform = joint_transform
        self.file_names = os.listdir(input_dir)[:num_samples]

    def __len__(self):
        return len(self.file_names)

    def __getitem__(self, idx):
        # Load activity and dose images (numpy arrays)
        input_volume = np.load(os.path.join(self.input_dir, self.file_names[idx]))
        output_volume = np.load(os.path.join(self.output_dir, self.file_names[idx]))

        # Convert numpy arrays to PyTorch tensors
        input_volume = torch.tensor(input_volume, dtype=torch.float32)
        output_volume = torch.tensor(output_volume, dtype=torch.float32)

        # Apply transforms
        if self.input_transform:
            input_volume = self.input_transform(input_volume)
        if self.output_transform:
            output_volume = self.output_transform(output_volume)
        if self.joint_transform:
            input_volume = self.joint_transform(input_volume)
            output_volume = self.joint_transform(output_volume)
        return input_volume, output_volume


# Function to get means, standard deviations, minimum and maximum values of the selected data
def dataset_statistics(input_dir, output_dir, num_samples=5):
    dataset = DoseActivityDataset(input_dir=input_dir, output_dir=output_dir, num_samples=948)
    # Input images (activity)
    input_data = [x[0] for x in dataset]
    input_data = torch.stack(input_data)
    mean_input = input_data.mean()
    std_input = input_data.std()
    max_input = input_data.max()
    min_input= input_data.min()

    print(f'Max. input pixel value: {max_input:0.6f}')
    print(f'\nMin. input pixel value: {min_input:0.6f}')
    print(f'\nMean input pixel value normalized: {mean_input:0.6f}')
    print(f'\nStandard deviation of the input pixel values: {std_input:0.6f}')

    # Output images (dose)
    output_data = [x[1] for x in dataset]
    output_data = torch.stack(output_data)
    mean_output = output_data.mean()
    std_output = output_data.std()
    max_output = output_data.max()
    min_output = output_data.min()

    print(f'\n\nMax. output pixel value: {max_output:0.11f}')
    print(f'\nMin. output pixel value: {min_output:0.11f}')
    print(f'\nMean output pixel value normalized: {mean_output:0.11f}')
    print(f'\nStandard deviation of the output pixel values: {std_output:0.11f}')

    return [mean_input, std_input, min_input, max_input, mean_output, std_output, min_output, max_output]
    

# CUSTOM TRANSFORMS
# Torchvision transforms do not work on floating point numbers
class MinMaxNormalize:
    def __init__ (self, min_tensor, max_tensor):
        self.min_tensor = min_tensor
        self.max_tensor = max_tensor
    def __call__(self, img):
        return (img - self.min_tensor)/(self.max_tensor - self.min_tensor)


class GaussianBlurFloats:
    def __init__(self, sigma=2):
        self.sigma = sigma

    def __call__(self, img):
        # Convert tensor to numpy array
        image_array = img.cpu().numpy()

        # Apply Gaussian filter to each channel
        blurred_array = gaussian_filter(image_array, sigma=self.sigma)

        # Convert back to tensor
        blurred_tensor = torch.tensor(blurred_array, dtype=img.dtype, device=img.device)

        return blurred_tensor


# CUSTOM LOSSES
# Dice loss
def dice_loss(output, target, smooth=1e-10):
    # Compute the intersection and the sum of the two sets along specified dimensions
    intersection = (target * output).sum(dim=(1, 2, 3))
    total_sum = target.sum(dim=(1, 2, 3)) + output.sum(dim=(1, 2, 3))
    # Compute Dice coefficient
    dice_coeff = (2. * intersection + smooth) / (total_sum + smooth)
    # Compute and return Dice loss averaged over the batch
    return 1. - dice_coeff.mean()

# Relative error
def RE_loss(output, target, mean_output=0, std_output=1):  # Relative error loss
    output = mean_output + output * std_output  # undoing normalization
    target = mean_output + target * std_output
    abs_diff = output - target
    max_intensity= torch.amax(target, dim=[1,2,3])
    loss = abs_diff / max_intensity.view(-1, 1, 1, 1) * 100  # Loss is a tensor in which each pixel contains the relative error
    return loss

# MSE loss
def mse_loss(output, target, mean_output=0, std_output=1):  # Relative error loss
    output = mean_output + output * std_output  # undoing normalization
    target = mean_output + target * std_output
    loss = torch.sum((output - target)**2)
    return loss

# Range deviation
def range_loss(output, target, range=0.9):
    ''' This is the difference between output and target in the depth at which
    the dose reaches a certain percentage of the Bragg Peak dose after the Bragg Peak.
    This is done for every curve in the transversal plane where the dose is not zero.
    '''
    max_global = torch.amax(target, dim=(1, 2, 3))  # Overall max for each image
    max_along_depth, idx_max_along_depth= torch.max(target, dim=1)  # Max at each transversal point
    indices_keep = max_along_depth > 0.1 * max_global.unsqueeze(-1).unsqueeze(-1)  # Unsqueeze to match dimensions of the tensors. These are the indices of the transversal Bragg Peaks higher than 1% of the highest peak BP
    max_along_depth = max_along_depth[indices_keep] # Only keep the max, or bragg peaks, of transversal points with non-null dose
    idx_max_along_depth = idx_max_along_depth[indices_keep]
    target_permuted = torch.permute(target, (1, 0, 2, 3))
    output_permuted = torch.permute(output, (1, 0, 2, 3))
    new_shape = [150] + [torch.sum(indices_keep).item()]
    indices_keep = indices_keep.expand(150, -1, -1, -1)
    ddp_data = target_permuted[indices_keep].reshape(new_shape)
    ddp_output_data = output_permuted[indices_keep].reshape(new_shape)

    depth = np.arange(150)  # in mm´
    ddp = interp1d(depth, ddp_data, axis=0, kind='cubic')
    ddp_output = interp1d(depth, ddp_output_data, axis=0, kind='cubic')
    depth_extended = np.linspace(min(depth), max(depth), 10000)
    dose_at_range = range * max_along_depth.numpy()

    ddp_depth_extended = ddp(depth_extended)
    ddp_output_depth_extended = ddp_output(depth_extended)
    # n_plot = 115
    # plt.plot(depth_extended, ddp_depth_extended[:, n_plot])
    # plt.plot(depth_extended, ddp_output_depth_extended[:, n_plot])

    mask = depth_extended[:, np.newaxis] > idx_max_along_depth.numpy()  # mask to only consider the range after the bragg peak (indices smaller than the index at the BP)
    ddp_depth_extended[mask] = 0
    ddp_output_depth_extended[mask] = 0
    depth_at_range = depth_extended[np.abs(ddp_depth_extended - dose_at_range).argmin(axis=0)]
    depth_at_range_output = depth_extended[np.abs(ddp_output_depth_extended - dose_at_range).argmin(axis=0)]

    # plt.plot(depth_at_range[n_plot], dose_at_range[n_plot], marker=".", markersize=10)
    # plt.plot(depth_at_range_output[n_plot], dose_at_range[n_plot], marker=".", markersize=10)
    return torch.tensor(depth_at_range_output - depth_at_range)


# Plotting results

def plot_slices(trained_model, loader, device, mean_input=0, std_input=1,
                mean_output=0, std_output=1, y_slice = 30,
                save_plot_dir = "images/sample.png"):
    # Loading a few examples
    input, target = next(iter(loader))
    trained_model.eval()  # Putting the model in validation mode
    output = trained_model(input.to(device))

    output = output.detach().cpu()  # Detaching from the computational graph
    torch.cuda.empty_cache()  # Freeing up RAM 

    sns.set()
    n_plots = 3
    fig, axs = plt.subplots(n_plots, 4, figsize=[13, 8])

    input_scaled = mean_input + input * std_input
    output_scaled = mean_output + output * std_output  # undoing normalization
    target_scaled = mean_output + target * std_output

    font_size = 15

    # Add titles to the columns
    column_titles = ['Input (Activation)', 'Target (Reference dose)', 'Output (Calculated dose)', 'Error = |Output - Target|']
    for ax, col in zip(axs[0], column_titles):
        ax.set_title(col, fontsize=font_size)

    for idx in range(n_plots):
        input_img = input_scaled[idx].cpu().detach().squeeze(0)[:,y_slice,:]
        out_img = output_scaled[idx].cpu().detach().squeeze(0)[:,y_slice,:]
        target_img = target_scaled[idx].cpu().detach().squeeze(0)[:,y_slice,:]
        diff_img = abs(target_img - out_img)
        c1 = axs[idx, 0].imshow(np.flipud(input_img).T, cmap='jet', aspect='auto')
        axs[idx, 0].set_xticks([])
        axs[idx, 0].set_yticks([])
        if idx == 0:
            axs[idx, 0].plot([40, 140], [10, 10], linewidth=12, color='black')
            axs[idx, 0].plot([40, 140], [10, 10], linewidth=8, color='white', label='1 cm')
            axs[idx, 0].text(75, 19, '10 cm', color='white', fontsize=font_size)

        c2 = axs[idx, 1].imshow(np.flipud(target_img).T, cmap='jet', aspect='auto')
        axs[idx, 1].set_xticks([])
        axs[idx, 1].set_yticks([])
        axs[idx, 2].imshow(np.flipud(out_img).T, cmap='jet', vmax=torch.max(target_img), aspect='auto')
        axs[idx, 2].set_xticks([])
        axs[idx, 2].set_yticks([])
        axs[idx, 3].imshow(np.flipud(diff_img).T, cmap='jet', vmax=torch.max(target_img), aspect='auto')
        axs[idx, 3].set_xticks([])
        axs[idx, 3].set_yticks([])

    energy_beam_1 = 144
    energy_beam_2 = 167
    energy_beam_3 = 137

    fig.text(0.0, 0.81, f'{energy_beam_1} MeV Beam', va='center', rotation='vertical', fontsize=font_size, fontstyle='italic')
    fig.text(0.0, 0.51, f'{energy_beam_2} MeV Beam', va='center', rotation='vertical', fontsize=font_size, fontstyle='italic')
    fig.text(0.0, 0.2, f'{energy_beam_3} MeV Beam', va='center', rotation='vertical', fontsize=font_size, fontstyle='italic')

    cbar_ax1 = fig.add_axes([0.029, 0.01, 0.22, 0.03])
    cbar_ax2 = fig.add_axes([0.28, 0.01, 0.7, 0.03])

    cbar1 = fig.colorbar(c1, cax=cbar_ax1, orientation='horizontal')
    cbar1.set_label(label=r'$\beta^+$ Decay Count $\ / \ mm^3$', size=font_size)
    cbar1.ax.tick_params(labelsize=font_size)

    cbar2 = fig.colorbar(c2, cax=cbar_ax2, orientation='horizontal')
    cbar2.set_label(label='Dose ($Gy$)', size=font_size)
    cbar2.ax.xaxis.get_offset_text().set(size=font_size)
    cbar2.ax.tick_params(labelsize=font_size)

    fig.tight_layout()
    fig.subplots_adjust(bottom=0.06, left=0.022)

    fig.savefig(save_plot_dir, dpi=300, bbox_inches='tight')


# Plotting dose-depth profile
def plot_ddp(trained_model, loader, device, mean_output=0, std_output=1,
             y_slice = 30, save_plot_dir = "images/ddp.png"):
    # Loading a few examples
    input, target = next(iter(loader))
    trained_model.eval()  # Putting the model in validation mode
    output = trained_model(input.to(device))

    output = output.detach().cpu()  # Detaching from the computational graph
    torch.cuda.empty_cache()  # Freeing up RAM 

    sns.set()
    n_plots = 3
    fig, axs = plt.subplots(n_plots, 1, figsize=[12, 12])

    output_scaled = mean_output + output * std_output  # undoing normalization
    target_scaled = mean_output + target * std_output    

    axs[0].set_title("Dose profile")

    for idx in range(n_plots):
        y_slice = 30
        out_img = output_scaled[idx].cpu().detach().squeeze(0).numpy()
        target_img = target_scaled[idx].cpu().detach().squeeze(0).numpy()
        out_profile = np.sum(out_img, axis=(1,2))
        target_profile = np.sum(target_img, axis=(1,2))
        distance = np.flip(np.arange(len(out_profile)))
        axs[idx].plot(distance, out_profile, label="Calculated Dose", linewidth=2)
        axs[idx].plot(distance, target_profile, label="Target Dose", linewidth=2)
        axs[idx].legend()
        axs[idx].grid(True)
        axs[idx].set_xlabel("Depth (mm)")
        axs[idx].set_ylabel("Dose deposited (Gy)")

    fig.savefig(save_plot_dir, dpi=300, bbox_inches='tight')


