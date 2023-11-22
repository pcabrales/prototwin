import random
import torch
import torch.nn.functional as F
import numpy as np
import os
from torch.utils.data import Dataset
from scipy.ndimage import gaussian_filter
from scipy.interpolate import interp1d
import unfoldNd
import seaborn as sns 
import matplotlib.pyplot as plt
import pymedphys

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


# Creating a 128x128 dataset for the dose/activity input/output pairs
class DoseActivityDataset(Dataset):
    """
    Create the dataset where the activity is the input and the dose is the output.
    The relevant transforms are applied.
    """
    def __init__(self, input_dir, output_dir, num_samples=5, input_transform=None, output_transform=None, joint_transform=None, CT=None, energy_beam_dict=None):
        self.input_dir = input_dir
        self.output_dir = output_dir
        self.input_transform = input_transform
        self.output_transform = output_transform
        self.joint_transform = joint_transform
        self.energy_beam_dict = energy_beam_dict
        self.file_names = os.listdir(input_dir)[:num_samples]
        self.CT = CT

    def __len__(self):
        return len(self.file_names)

    def __getitem__(self, idx):
        # Load activity and dose images (numpy arrays)
        input_volume = np.load(os.path.join(self.input_dir, self.file_names[idx]))
        output_volume = np.load(os.path.join(self.output_dir, self.file_names[idx]))

        # Convert numpy arrays to PyTorch tensors
        input_volume = torch.tensor(input_volume, dtype=torch.float32).unsqueeze(0)
        output_volume = torch.tensor(output_volume, dtype=torch.float32).unsqueeze(0)

        # Apply transforms
        if self.input_transform:
            input_volume = self.input_transform(input_volume)
        if self.output_transform:
            output_volume = self.output_transform(output_volume)
        if self.joint_transform:
            input_volume, output_volume = self.joint_transform(input_volume, output_volume)
        if self.CT is not None:
            input_volume = torch.cat((input_volume, torch.tensor(self.CT, dtype=torch.float32).unsqueeze(0)))
        if self.energy_beam_dict is not None:
            beam_number = self.file_names[idx][0:4]
            beam_energy = self.energy_beam_dict.get(beam_number, 0.0)
            # if (beam_energy == 0.0):
            #     print(beam_number)
            beam_energy = float(beam_energy) / 1000  # from keV to MeV
        else: 
            beam_energy = "N/A"
        
        return input_volume, output_volume, beam_energy


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

class JointCompose:
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, input_volume, output_volume):
        for transform in self.transforms:
            seed = torch.randint(0, 2**32, (1,)).item()
            
            torch.manual_seed(seed)
            input_volume = transform(input_volume)
            
            torch.manual_seed(seed)
            output_volume = transform(output_volume)
                
        return input_volume, output_volume

# Torchvision transforms do not work on floating point numbers
class MinMaxNormalize:
    def __init__ (self, min_tensor, max_tensor):
        self.min_tensor = min_tensor
        self.max_tensor = max_tensor
    def __call__(self, img):
        return (img - self.min_tensor)/(self.max_tensor - self.min_tensor)


class GaussianBlurFloats:
    def __init__(self, p=1, sigma=2):
        self.sigma = sigma
        self.p = p  # Probability of applying the  blur

    def __call__(self, img):
        # Convert tensor to numpy array
        image_array = img.squeeze(0).cpu().numpy()

        if random.random() < self.p:
            # Apply Gaussian filter to each channel
            self.sigma = random.random() * self.sigma  # A random number between 0 and the max sigma
            blurred_array = gaussian_filter(image_array, sigma=self.sigma)
        else: 
            blurred_array = image_array

        # Convert back to tensor
        blurred_tensor = torch.tensor(blurred_array, dtype=img.dtype, device=img.device).unsqueeze(0)

        return blurred_tensor

class Random3DCrop: # Crop image to be multiple of 8
    def __init__(self, output_size):
        assert isinstance(output_size, (int, tuple))
        if isinstance(output_size, int):
            self.output_size = (output_size, output_size, output_size)
        else:
            assert len(output_size) == 3
            self.output_size = output_size

    def __call__(self, img):
        h, w, d = img.shape[-3:]

        new_h, new_w, new_d = self.output_size

        top = torch.randint(0, h - new_h + 1, (1,)).item()
        left = torch.randint(0, w - new_w + 1, (1,)).item()
        front = torch.randint(0, d - new_d + 1, (1,)).item()
        
        return img[:,
                   top: top + new_h,
                   left: left + new_w,
                   front: front + new_d]
        
class Resize3D:
    # This function resizes by interpolatingg
    def __init__(self, size):
        self.size = size

    def __call__(self, img):
        img = img.unsqueeze(0)  # Interpolate takes input tensors that have a batch and channel dimensions
        img = F.interpolate(img, size=self.size, mode='trilinear', align_corners=True)
        return img.squeeze(0)
    
class ResizeCropAndPad3D:
    # This function resizes by cropping and padding
    def __init__(self, size):
        self.size = size

    def __call__(self, img):
        img = img.unsqueeze(1)  # Pad takes input tensors that have a batch and channel dimensions
        pad_right = pad_left = pad_top = pad_bottom = pad_front = pad_back = 0
        if self.size[0] > img.shape[2]:
            pad_left = self.size[0] - img.shape[2]
        elif self.size[0] < img.shape[2]:
            img = img[:, :, img.shape[2] - self.size[0] : , 
                      :, :]
        if self.size[1] > img.shape[3]:
            pad_top = (self.size[1] - img.shape[3]) // 2
            pad_bottom = self.size[1] - img.shape[3] - pad_top
        elif self.size[1] < img.shape[3]: 
            crop_top = (img.shape[3] - self.size[1]) // 2
            crop_bottom = img.shape[3] - self.size[1] - crop_top
            img = img[:, :, :, crop_top : img.shape[3] - crop_bottom, 
                      :]
        if self.size[2] > img.shape[4]:
            pad_front = (self.size[2] - img.shape[4]) // 2
            pad_back = self.size[2] - img.shape[4] - pad_front
        elif self.size[2] < img.shape[4]:
            crop_front = (img.shape[4] - self.size[2]) // 2
            crop_back = img.shape[4] - self.size[2] - crop_front
            img = img[:, :, :, :, crop_front : img.shape[4] - crop_back]
            
        img = F.pad(img, (pad_front, pad_back, pad_top, pad_bottom, pad_left, pad_right), mode='constant', value=torch.min(img))
        return img.squeeze(1)
        
class GaussianBlob:
    def __init__(self, size, sigma, type_blob="gaussian"):
        self.size = size
        self.sigma = sigma
        self.type_blob = type_blob

    def __call__(self, img):
        img = img.squeeze(0)
        max_overall = torch.max(img)
        idx_max_overall = np.unravel_index(torch.argmax(img), img.shape)
        bool_img = img > 0.1 * max_overall  # Looking for the position just after the BP, since we will place the blur between the start and that position
        bp_position = torch.where(bool_img.any(dim=1).any(dim=1))[0][0].item()
        
        # Create a 3D Gaussian kernel
        ax = torch.arange(-self.size // 2 + 1., self.size // 2 + 1.)
        xx, yy, zz = torch.meshgrid(ax, ax, ax)
        kernel = 0.5 * max_overall * torch.exp(-(xx**2 + yy**2 + zz**2) / (2 * self.sigma**2))
        
        center = (random.randint(bp_position, img.shape[0] - 1), idx_max_overall[1], idx_max_overall[2])
        
        # Find the start and end indices for slicing
        z_start = max(center[0] - self.size // 2, 0)
        y_start = max(center[1] - self.size // 2, 0)
        x_start = max(center[2] - self.size // 2, 0)
        z_end = min(center[0] + self.size // 2 + 1, img.shape[0]) - 1
        y_end = min(center[1] + self.size // 2 + 1, img.shape[1]) - 1
        x_end = min(center[2] + self.size // 2 + 1, img.shape[2]) - 1

        # Adjust the kernel size for edge cases
        kernel = kernel[(z_start - center[0] + self.size // 2):(z_end - center[0] + self.size // 2),
                        (y_start - center[1] + self.size // 2):(y_end - center[1] + self.size // 2),
                        (x_start - center[2] + self.size // 2):(x_end - center[2] + self.size // 2)]

        # Add the Gaussian blob to the tensor
        
        if self.type_blob == "blank":
            img[z_start:z_end, y_start:y_end, x_start:x_end] = 0
        else:
            img[z_start:z_end, y_start:y_end, x_start:x_end] += kernel
            
        return img.unsqueeze(0)


# CUSTOM LOSSES
# Relative error
def RE_loss(output, target, mean_output=0, std_output=1):  # Relative error loss
    output = output.squeeze(1)  # Eliminate channel dim
    target = target.squeeze(1)
    output = mean_output + output * std_output  # undoing normalization
    target = mean_output + target * std_output
    abs_diff = output - target
    max_intensity= torch.amax(target, dim=[1,2,3])
    loss = abs_diff / max_intensity.view(-1, 1, 1, 1) * 100  # Loss is a tensor in which each pixel contains the relative error
    return loss

def manual_permute(tensor, dims):
    for i in range(len(dims)):
        if i != dims[i]:
            tensor = tensor.transpose(i, dims[i])
            dims = [dims.index(j) if j == i else j for j in dims]
    return tensor

def post_BP_loss(output, target, device="cpu", mean_output=0, std_output=1):
    output = output.squeeze(1)  # Eliminate channel dim
    target = target.squeeze(1)
    longitudinal_size = output.shape[1]
    output = mean_output + output * std_output  # undoing normalization
    target = mean_output + target * std_output
    max_global = torch.amax(target, dim=(1, 2, 3))  # Overall max for each image
    max_along_depth, idx_max_along_depth = torch.max(target, dim=1)  # Max at each transversal point
    indices_keep = max_along_depth > 0.01 * max_global.unsqueeze(-1).unsqueeze(-1)  # Unsqueeze to match dimensions of the tensors. These are the indices of the transversal Bragg Peaks higher than 1% of the highest peak BP
    max_along_depth = max_along_depth[indices_keep] # Only keep the max, or bragg peaks, of transversal points with non-null dose
    idx_max_along_depth = idx_max_along_depth[indices_keep]
    # target_permuted = torch.permute(target, (1, 0, 2, 3))
    # output_permuted = torch.permute(output, (1, 0, 2, 3))
    target_permuted = manual_permute(target, (1, 0, 2, 3))
    output_permuted = manual_permute(output, (1, 0, 2, 3))
    new_shape = [longitudinal_size] + [torch.sum(indices_keep).item()]
    indices_keep = indices_keep.expand(longitudinal_size, -1, -1, -1)
    ddp_data = target_permuted[indices_keep].reshape(new_shape)
    ddp_output_data = output_permuted[indices_keep].reshape(new_shape)
    
    depth = torch.arange(longitudinal_size, dtype=torch.float32).to(device)  # in mm
    mask_pre_BP = depth[:, None] > idx_max_along_depth    # mask to only consider the range after the bragg peak (indices smaller than the index at the BP)
    mask_post_BP = ddp_data < 0.1 * max_along_depth  # mask to only consider the range between the BP and the drop until 0.1 * BP
    mask = mask_pre_BP | mask_post_BP
    ddp_data[mask] = 0
    ddp_output_data[mask] = 0
    diff_target_output = ddp_data - ddp_output_data
    diff_target_output = (diff_target_output - mean_output) / std_output
    
    # n_plot = 200
    # plt.figure()
    # plt.plot(depth, ddp_data[:, n_plot], marker=".", markersize=10)
    # plt.plot(depth, ddp_output_data[:, n_plot], marker=".", markersize=10)
    return torch.mean((diff_target_output) ** 2)
    

# Range deviation
def range_loss(output, target, range_val=0.9, device="cpu", mean_output=0, std_output=1):
    ''' This is the difference between output and target in the depth at which
    the dose reaches a certain percentage of the Bragg Peak dose after the Bragg Peak.
    This is done for every curve in the transversal plane where the BP is larger than 0.05 of the max BP.
    '''
    output = output.squeeze(1)  # Eliminate channel dim
    target = target.squeeze(1)
    longitudinal_size = output.shape[1]
    output = mean_output + output * std_output  # undoing normalization
    target = mean_output + target * std_output
    max_global = torch.amax(target, dim=(1, 2, 3))  # Overall max for each image
    max_along_depth, idx_max_along_depth = torch.max(target, dim=1)  # Max at each transversal point
    indices_keep = max_along_depth > 0.1 * max_global.unsqueeze(-1).unsqueeze(-1)  # Unsqueeze to match dimensions of the tensors. These are the indices of the transversal Bragg Peaks higher than 1% of the highest peak BP
    max_along_depth = max_along_depth[indices_keep] # Only keep the max, or bragg peaks, of transversal points with non-null dose
    idx_max_along_depth = idx_max_along_depth[indices_keep]
    # target_permuted = torch.permute(target, (1, 0, 2, 3))
    # output_permuted = torch.permute(output, (1, 0, 2, 3))
    target_permuted = manual_permute(target, (1, 0, 2, 3))
    output_permuted = manual_permute(output, (1, 0, 2, 3))
    new_shape = [longitudinal_size] + [torch.sum(indices_keep).item()]
    indices_keep = indices_keep.expand(longitudinal_size, -1, -1, -1)
    ddp_data = target_permuted[indices_keep].reshape(new_shape)
    ddp_output_data = output_permuted[indices_keep].reshape(new_shape)

    depth = torch.arange(longitudinal_size, dtype=torch.float32).to(device)  # in mm
    depth_extended = torch.linspace(min(depth), max(depth), 10000).to(device)
    
    # ddp = interp1d(depth, ddp_data, axis=0, kind='cubic')
    # ddp_output = interp1d(depth, ddp_output_data, axis=0, kind='cubic')
    # ddp_depth_extended = ddp(depth_extended)
    # ddp_output_depth_extended = ddp_output(depth_extended)
    
    ddp_depth_extended  = torch_cubic_interp1d_2d(depth, ddp_data, depth_extended)
    ddp_output_depth_extended  = torch_cubic_interp1d_2d(depth, ddp_output_data, depth_extended)
    max_along_depth = torch.amax(ddp_depth_extended, dim=0)
    max_along_output_depth = torch.amax(ddp_output_depth_extended, dim=0)
    dose_at_range = range_val * max_along_depth
    dose_at_range_output = range_val * max_along_output_depth
    
    # n_plot = 115
    # plt.plot(depth_extended, ddp_depth_extended[:, n_plot])
    # plt.plot(depth_extended, ddp_output_depth_extended[:, n_plot])

    # mask = depth_extended[:, np.newaxis] > idx_max_along_depth.numpy()  # mask to only consider the range after the bragg peak (indices smaller than the index at the BP)
    mask = depth_extended[:, None] > idx_max_along_depth  # mask to only consider the range after the bragg peak (indices smaller than the index at the BP)
    ddp_depth_extended[mask] = 0
    ddp_output_depth_extended[mask] = 0
    depth_at_range = depth_extended[torch.argmin(torch.abs(ddp_depth_extended - dose_at_range), dim=0)]
    depth_at_range_output = depth_extended[torch.argmin(torch.abs(ddp_output_depth_extended  - dose_at_range_output), dim=0)]

    # plt.plot(depth_at_range[n_plot], dose_at_range[n_plot], marker=".", markersize=10)
    # plt.plot(depth_at_range_output[n_plot], dose_at_range[n_plot], marker=".", markersize=10)
    return depth_at_range - depth_at_range_output


def plot_range_histogram(range_list, save_plot_dir="images/hist_BP_deviation.png"):
    plt.figure(figsize=(8,6))
    bins = 10
    plt.hist(range_list.flatten(), bins=bins, color='blue', alpha=0.5, label='No Reconstruction')

    plt.title('Deviation - Bragg Peak (BP)')
    plt.xlabel('Reference BP - Reconstructed BP (mm)')
    plt.ylabel('Frequency')

    plt.legend()
    plt.savefig(save_plot_dir, dpi=500)
    return None


# Range deviation
def gamma_index(output, target, tolerance=0.03, beta=5, mean_output=0, std_output=1, threshold=0.2):
    # So far, we only consider neighbouring pixels, which we assume have a distance = 0 between them
    # Tolerance is set as default to 3%, threshold = 0.2  as set in Sonia Martinot's paper
    
    output = mean_output + output * std_output  # undoing normalization
    target = mean_output + target * std_output
    target = target.squeeze(1)  # no need to squeeze **output** as it is done when we unfold it

    max_target = torch.amax(target, dim=(1, 2, 3)) # Overall max for each image in the batch
    target = target / max_target.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)  # normalising to the maximum dose so that we can later directly apply the threshold
    output = output / max_target.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
    
    # Define Unfold layer
    unfold = unfoldNd.UnfoldNd(kernel_size=3, stride=1, padding=1)
    
    # Apply Unfold 
    # Output shape: (batch_size, channels * kernel_size * kernel_size, L)
    output_unfolded = unfold(output)
    
    # finding the values in the target vector larger than a certain threshold wrt the max
    target_flat = target.flatten(start_dim=1, end_dim=3)
    voxels_threshold = target_flat > threshold

    target_threshold = target_flat[voxels_threshold]
    num_voxels_above_threshold = target_threshold.shape[0]
    
    # permuted to select the spatial indices without touching the kernel dimension, which was in dim=1 but we are moving it to dim=0, switching
    # it with batch size, which we do want to consider
    output_unfolded = manual_permute(output_unfolded, (1, 0, 2))
    output_threshold = output_unfolded[:, voxels_threshold]  # only keeping those above the threshold

    diff = torch.abs(output_threshold - target_threshold.unsqueeze(0))  # difference between the output and target

    gamma = torch.amin(diff, dim=0) / tolerance  # minimum among all neighbours, as in the gamma index function
    
    gamma_index_batch = torch.sum(sigmoid((1 - gamma), beta)) / num_voxels_above_threshold
    
    return gamma_index_batch


def sigmoid(x, beta):
    return 1 / (1 + torch.exp(-beta * x))

def pymed_gamma(gamma_pymed_list, batch_output, batch_target, dose_percent_threshold, distance_mm_threshold, threshold, mean_output=0, std_output=1):
# for pymed gamma calculation:
    for idx in range(batch_target.shape[0]):
        output = batch_output[idx].unsqueeze(0)
        target = batch_target[idx].unsqueeze(0)
        output = mean_output + output * std_output  # undoing normalization
        target = mean_output + target * std_output
        axes_reference = (np.arange(output.shape[2]), np.arange(output.shape[3]), np.arange(output.shape[4]))    
        gamma = pymedphys.gamma(
            axes_reference, target.squeeze(0).squeeze(0).cpu().detach().numpy(), 
            axes_reference, output.squeeze(0).squeeze(0).cpu().detach().numpy(),
            dose_percent_threshold = dose_percent_threshold,
            distance_mm_threshold = distance_mm_threshold, 
            lower_percent_dose_cutoff=threshold*100)
        valid_gamma = gamma[~np.isnan(gamma)]
        pass_ratio = np.sum(valid_gamma <= 1) / len(valid_gamma)
        gamma_pymed_list.append(pass_ratio)
    return gamma_pymed_list

# Correcting the indexing error and implementing the cubic interpolation for 2D y array
def torch_cubic_interp1d_2d(x, y, x_new):
    n = len(x)
    xdiff = x[1:] - x[:-1]
    
    # Make sure y has two dimensions
    if len(y.shape) == 1:
        y = y[:, None]
    
    ydiff = y[1:, :] - y[:-1, :]
    
    # Computing the slopes
    slopes = ydiff / xdiff[:, None]
    
    # Computing the second derivatives
    alpha = xdiff[1:] / (xdiff[:-1] + xdiff[1:])
    beta = 1 - alpha
    dydx = 0.5 * (slopes[:-1, :] + slopes[1:, :])
    d2ydx2 = (slopes[1:, :] - slopes[:-1, :]) / (xdiff[:-1, None] + xdiff[1:, None])
    
    # Boundary conditions for natural cubic spline
    d2ydx2_first = 2 * (alpha[0, None] * slopes[0, :] - dydx[0, :]) / ((1 - alpha[0]) * xdiff[0])
    d2ydx2_last = 2 * (dydx[-1, :] - beta[-1, None] * slopes[-1, :]) / (xdiff[-1] * (1 - beta[-1]))
    
    d2ydx2 = torch.cat([d2ydx2_first[None, :], d2ydx2, d2ydx2_last[None, :]], dim=0)
    
    # Finding the segment each x_new is in
    indices = torch.searchsorted(x[1:], x_new)
    indices = torch.clamp(indices, 0, n-2)
    
    x0 = x[indices]
    x1 = x[indices + 1]
    a0 = y[indices, :]
    a1 = y[indices + 1, :]
    s0 = slopes[indices, :]
    s1 = slopes[torch.clamp(indices + 1, 0, n-2), :]
    d2a0 = d2ydx2[indices, :]
    d2a1 = d2ydx2[torch.clamp(indices + 1, 0, n-2), :]
    
    # Computing the cubic polynomials for each segment
    t = (x_new - x0)[:, None] / (x1 - x0)[:, None]
    h00 = (1 + 2*t) * (1 - t)**2
    h10 = t * (1 - t)**2
    h01 = t**2 * (3 - 2*t)
    h11 = t**2 * (t - 1)
    
    interpolated = h00 * a0 + h10 * (x1 - x0)[:, None] * s0 + h01 * a1 + h11 * (x1 - x0)[:, None] * s1
    return interpolated

# Plotting results

def plot_slices(trained_model, loader, device, CT_flag=False, CT_manual=None, mean_input=0, std_input=1,
                mean_output=0, std_output=1, z_slice = None,
                save_plot_dir = "images/sample.png", patches=False, patch_size=56):
        
    input, output, target, beam_energy = get_input_output_target(trained_model, loader, device, patches, patch_size)
    n_plots = 3
    
    if CT_flag:  # This is if the network was trained with the CT as a second channel
        CT = input[:, 1, :, :, :]  # CT is the second channel
        vmin_CT = -1
        vmax_CT = 2.5
    elif CT_manual is not None:  # Alternatively, the user can manually pass the CT as input 
        CT = torch.stack([torch.tensor(CT_manual)] * n_plots)    
        CT_flag = True
        vmin_CT = -125
        vmax_CT = 225
    
    input = input[:, 0, :, :, :]  # Plot the activity only
    
    sns.set()
    fig, axs = plt.subplots(n_plots, 4, figsize=[13, 8])

    input_scaled = mean_input + input * std_input
    output_scaled = mean_output + output * std_output  # undoing normalization
    target_scaled = mean_output + target * std_output
    font_size = 15
    max_target = torch.max(target_scaled[0:n_plots])
    max_input = torch.max(input_scaled[0:n_plots])

    # Add titles to the columns
    column_titles = ['Input (Activation)', 'Target (Reference dose)', 'Output (Calculated dose)', 'Error = |Output - Target|']
    for ax, col in zip(axs[0], column_titles):
        ax.set_title(col, fontsize=font_size)

    for idx in range(n_plots):
        input_img = input_scaled[idx].cpu().detach().squeeze(0).squeeze(0)
        out_img = output_scaled[idx].cpu().detach().squeeze(0).squeeze(0)
        target_img = target_scaled[idx].cpu().detach().squeeze(0).squeeze(0)
        if z_slice is None:
            idcs_max_target = torch.where(target_img == torch.max(target_img))
            z_slice_idx = idcs_max_target[-1].item()  # Plotting the slice where the value of the dose is maximum
        else:
            z_slice_idx = z_slice  # if we specify a z slice we use it
            
        input_img = input_img[:,:,z_slice_idx]
        out_img = out_img[:,:,z_slice_idx]
        target_img = target_img[:,:,z_slice_idx]
        
        if CT_flag:
            # mask = np.where(target_img > 0.1 * torch.max(target_img), 0.8, 0.0)  # Masking 0 values to be able to see the CT ###
            mask_input = ((input_img - torch.min(input_img)) / (torch.max(input_img) - torch.min(input_img))).numpy() * 1.4
            mask_target = ((target_img - torch.min(target_img)) / (torch.max(target_img) - torch.min(target_img))).numpy() * 1.4
            mask_input[mask_input > 1.0] = 1.0
            mask_target[mask_target > 1.0] = 1.0
            
            CT_idx = CT[idx].cpu().detach().squeeze(0).squeeze(0)[:,:,z_slice_idx]
            for plot_column in range(4):
                axs[idx, plot_column].imshow(np.flipud(CT_idx).T, cmap='gray', vmin=vmin_CT, vmax=vmax_CT)
            
        else:    
            mask_input = np.ones_like(input_img).astype(float)  # Leave all if no CT is provided
            mask_target = np.ones_like(input_img).astype(float)  # Leave all if no CT is provided
            
        mask_input = np.flipud(mask_input).T
        mask_target = np.flipud(mask_target).T
        
        diff_img = abs(target_img - out_img)
        c1 = axs[idx, 0].imshow(np.flipud(input_img).T, vmax=max_input, cmap='jet', aspect='auto', alpha=mask_input)
        axs[idx, 0].set_xticks([])
        axs[idx, 0].set_yticks([])
        if idx == 0:
            axs[idx, 0].plot([40, 140], [10, 10], linewidth=12, color='black')
            axs[idx, 0].plot([40, 140], [10, 10], linewidth=8, color='white', label='1 cm')
            axs[idx, 0].text(75, 19, '10 cm', color='white', fontsize=font_size)

        c2 = axs[idx, 1].imshow(np.flipud(target_img).T, vmax=max_target, cmap='jet', aspect='auto', alpha=mask_target)
        axs[idx, 1].set_xticks([])
        axs[idx, 1].set_yticks([])
        axs[idx, 2].imshow(np.flipud(out_img).T, cmap='jet', vmax=max_target, aspect='auto', alpha=mask_target)
        axs[idx, 2].set_xticks([])
        axs[idx, 2].set_yticks([])
        axs[idx, 3].imshow(np.flipud(diff_img).T, cmap='jet', vmax=max_target, aspect='auto', alpha=mask_target)
        axs[idx, 3].set_xticks([])
        axs[idx, 3].set_yticks([])

    if isinstance(beam_energy[0].item(), float):
        energy_beam_1 = beam_energy[0]
        energy_beam_2 = beam_energy[1]
        energy_beam_3 = beam_energy[2]

        fig.text(0.0, 0.81, f'{energy_beam_1:.1f} MeV Beam', va='center', rotation='vertical', fontsize=font_size, fontstyle='italic')
        fig.text(0.0, 0.51, f'{energy_beam_2:.1f} MeV Beam', va='center', rotation='vertical', fontsize=font_size, fontstyle='italic')
        fig.text(0.0, 0.2, f'{energy_beam_3:.1f} MeV Beam', va='center', rotation='vertical', fontsize=font_size, fontstyle='italic')

    cbar_ax1 = fig.add_axes([0.029, 0.01, 0.22, 0.03])
    cbar_ax2 = fig.add_axes([0.3, 0.01, 0.66, 0.03])

    cbar1 = fig.colorbar(c1, cax=cbar_ax1, orientation='horizontal')
    cbar1.set_label(label=r'$\beta^+$ Decay Count $\ / \ mm^3$', size=font_size)
    cbar1.ax.tick_params(labelsize=font_size)

    cbar2 = fig.colorbar(c2, cax=cbar_ax2, orientation='horizontal', format='%.1e')
    cbar2.set_label(label='Dose ($Gy$)', size=font_size)
    cbar2.ax.xaxis.get_offset_text().set(size=font_size)
    cbar2.ax.tick_params(labelsize=font_size)

    fig.tight_layout()
    fig.subplots_adjust(bottom=0.06, left=0.022)

    fig.savefig(save_plot_dir, dpi=600, bbox_inches='tight')


# Plotting dose-depth profile
def plot_ddp(trained_model, loader, device, mean_output=0, std_output=1,
             save_plot_dir = "images/ddp.png", patches=False, patch_size=56):
    _, output, target, _ = get_input_output_target(trained_model, loader, device, patches, patch_size)
    sns.set()
    n_plots = 3
    fig, axs = plt.subplots(n_plots, 1, figsize=[n_plots * 4, 12])

    output_scaled = mean_output + output * std_output  # undoing normalization
    target_scaled = mean_output + target * std_output    

    font_size = 15
    axs[0].set_title("Dose profile", fontsize=font_size)

    for idx in range(n_plots):
        out_img = output_scaled[idx].cpu().detach().squeeze(0).squeeze(0).numpy()
        target_img = target_scaled[idx].cpu().detach().squeeze(0).squeeze(0).numpy()
        out_profile = np.sum(out_img, axis=(1,2))
        target_profile = np.sum(target_img, axis=(1,2))
        distance = np.flip(np.arange(len(out_profile)))
        axs[idx].plot(distance, out_profile, label="Calculated Dose", linewidth=2)
        axs[idx].plot(distance, target_profile, label="Target Dose", linewidth=2)
        # axs[idx].plot(distance, target_profile, label="Dose", linewidth=2)
        axs[idx].legend(fontsize=font_size)
        axs[idx].grid(True)
        axs[idx].set_xlabel("Depth (mm)", fontsize=font_size)
        axs[idx].set_ylabel("Dose deposited (Gy)", fontsize=font_size)
        axs[idx].tick_params(axis='both', which='minor', labelsize=font_size)

    fig.savefig(save_plot_dir, dpi=600, bbox_inches='tight')


def get_input_output_target(trained_model, loader, device, patches, patch_size):

    # Loading a few examples
    iter_loader = iter(loader)
    input, target, beam_energy = next(iter_loader)
    trained_model.eval()  # Putting the model in validation mode
    if patches:
        input, output, target = apply_model_to_patches(trained_model, input, target, patch_size)
    else: 
        output = trained_model(input.to(device))

    output = output.detach().cpu()  # Detaching from the computational graph
    torch.cuda.empty_cache()  # Freeing up RAM 
    
    while input.shape[0] < 3:
        # If the batch size is 1 or 2, load more examples
        input_i, target_i, beam_energy_i = next(iter_loader)
        if patches:
            input_i, output_i, target_i = apply_model_to_patches(trained_model, input, target, patch_size)
        else:
            output_i = trained_model(input_i.to(device))

        output_i = output_i.detach().cpu()  # Detaching from the computational graph
        torch.cuda.empty_cache()  # Freeing up RAM 
        
        input = torch.cat((input, input_i))
        output = torch.cat((output, output_i))
        target = torch.cat((target, target_i))
        beam_energy = torch.cat((beam_energy, beam_energy_i))

    return input, output, target, beam_energy

def apply_model_to_patches(trained_model, input, target, patch_size):
    # Initialize an empty tensor to hold the reconstructed image
    reconstructed_input = torch.zeros_like(input)
    reconstructed_output = torch.zeros_like(input)
    reconstructed_target = torch.zeros_like(input)
    
    # Loop through the image to get patches and their positions
    positions = []
    input_patches = []
    target_patches = []

    x_stop = input.shape[1] - patch_size + 1  # Final patch starting position
    x_patch_start = list(range(0, x_stop, patch_size))
    if x_patch_start[-1] != x_stop - 1: x_patch_start.append(x_stop - 1)
    y_stop = input.shape[2] - patch_size + 1 
    y_patch_start = list(range(0, y_stop, patch_size))
    if y_patch_start[-1] != y_stop - 1: y_patch_start.append(y_stop - 1)
    z_stop = input.shape[3] - patch_size + 1
    z_patch_start = list(range(0, z_stop, patch_size))
    if z_patch_start[-1] != z_stop - 1: z_patch_start.append(z_stop - 1)
    
    for x in x_patch_start:
        for y in y_patch_start:
            for z in z_patch_start:
                input_patches.append(input[:, x:x+patch_size, y:y+patch_size, z:z+patch_size])
                target_patches.append(target[:, x:x+patch_size, y:y+patch_size, z:z+patch_size])
                positions.append((x, y, z))

    # Apply model to each patch
    for input_patch, target_patch, (x, y, z) in zip(input_patches, target_patches, positions):
        # Assuming your model expects a 5D tensor of shape (batch, channel, x, y, z)
        output_patch = trained_model(input_patch)
        
        # Add the output patch to the corresponding position in the reconstructed image
        reconstructed_input[:, x:x+patch_size, y:y+patch_size, z:z+patch_size] = input_patch
        reconstructed_output[:, x:x+patch_size, y:y+patch_size, z:z+patch_size] = output_patch
        reconstructed_target[:, x:x+patch_size, y:y+patch_size, z:z+patch_size] = target_patch

    return reconstructed_input, reconstructed_output, reconstructed_target


def back_and_forth(dose2act_model, act2dose_model, act2dose_loader, device, reconstruct_dose=False, num_cycles=1, y_slice=32, mean_act=0, std_act=1, mean_dose=0, std_dose=1, save_plot_dir="images/reconstructed.png"):
    # If dose is set to true, then the dose image is converted to activity and back to dose again.
    # If it is set to False, that is done instead for activity images.
    # num_cycles defined the number of times that the cycle is applied

    # Loading a few examples
    iter_loader = iter(act2dose_loader)
    act, dose, _ = next(iter_loader)
    dose_original = dose.detach().cpu()
    act_original = act.detach().cpu()
    torch.cuda.empty_cache()  # Freeing up RAM 
    
    dose2act_model.eval()  # Putting the model in validation mode
    act2dose_model.eval()  # Putting the model in validation mode
    
    for i in range(num_cycles):
        if reconstruct_dose:
            act = dose2act_model(dose.to(device))
            act = act.detach().cpu()  
            torch.cuda.empty_cache()  # Freeing up RAM 
            dose = act2dose_model(act.to(device))
            dose = dose.detach().cpu()
            torch.cuda.empty_cache()  # Freeing up RAM 
        else:
            dose = act2dose_model(act.to(device))
            dose = dose.detach().cpu()
            torch.cuda.empty_cache()  # Freeing up RAM 
            act = dose2act_model(dose.to(device))
            act = act.detach().cpu()  
            torch.cuda.empty_cache()  # Freeing up RAM 
    
    while dose.shape[0] < 3:
        # If the batch size is 1 or 2, load more examples
        act_i, dose_i, _ = next(iter_loader)
        dose_i_original = dose.detach().cpu()
        act_i_original = act.detach().cpu()
        torch.cuda.empty_cache()  # Freeing up RAM 
        for i in range(num_cycles):
            if reconstruct_dose:
                act_i = dose2act_model(dose.to(device))
                act_i = act.detach().cpu()  
                torch.cuda.empty_cache()  # Freeing up RAM 
                dose_i = act2dose_model(act.to(device))
                dose_i = dose_i.detach().cpu()
                torch.cuda.empty_cache()  # Freeing up RAM 
            else:
                dose_i = act2dose_model(act.to(device))
                dose_i = dose_i.detach().cpu()
                torch.cuda.empty_cache()  # Freeing up RAM 
                act_i = dose2act_model(dose.to(device))
                act_i = act_i.detach().cpu()  
                torch.cuda.empty_cache()  # Freeing up RAM 
        
        dose = torch.cat((dose, dose_i))
        dose_original = torch.cat((dose_original, dose_i_original))
        act = torch.cat((act, act_i))
        act_original = torch.cat((act_original, act_i_original))
    
    sns.set()
    n_plots = 3
    font_size = 14
    fig, axs = plt.subplots(n_plots, 3, figsize=[12, 12])
    
    if reconstruct_dose:
        reconstruced_imgs = mean_dose + dose * std_dose  # undoing normalization
        original_imgs = mean_dose + dose_original * std_dose  # undoing normalization

        axs[0, 0].set_title("Original dose", fontsize=font_size)
        title_reconstructed = "Reconstructed dose after " + str(num_cycles) + " cycle(s)"

    else:
        reconstruced_imgs = mean_act + act * std_act  # undoing normalization
        original_imgs = mean_act + act_original * std_act  # undoing normalization

        axs[0, 0].set_title("Original activity", fontsize=font_size)
        title_reconstructed = "Reconstructed activity after " + str(num_cycles) + " cycle(s)"
        
    axs[0, 1].set_title(title_reconstructed, fontsize=font_size)
    axs[0, 2].set_title("|Reconstructed - Original|", fontsize=font_size)
    for idx in range(n_plots):
        original_img = original_imgs[idx].squeeze(0).squeeze(0)[:,y_slice,:]
        reconstructed_img = reconstruced_imgs[idx].squeeze(0).squeeze(0)[:,y_slice,:]
        diff_img = abs(reconstructed_img - original_img)
        c1 = axs[idx, 0].imshow(np.flipud(original_img).T, cmap='jet', aspect='auto')
        axs[idx, 0].set_xticks([])
        axs[idx, 0].set_yticks([])
        c2 = axs[idx, 1].imshow(np.flipud(reconstructed_img).T, vmax=torch.max(original_img), cmap='jet', aspect='auto')
        axs[idx, 1].set_xticks([])
        axs[idx, 1].set_yticks([])
        axs[idx, 2].imshow(np.flipud(diff_img).T, cmap='jet', vmax=torch.max(original_img), aspect='auto')
        axs[idx, 2].set_xticks([])
        axs[idx, 2].set_yticks([])


    fig.tight_layout()
    fig.savefig(save_plot_dir, dpi=300, bbox_inches='tight')


    return None



# def find_max_closest_to_edge(loader):
#     closest_edge_distance = float('inf')
#     closest_max_index = None
#     closest_tensor = None
    
#     for _, batch in loader:
#         # Find the maximum value in each 3D image of the batch
#         max_vals, _ = torch.max(batch.view(batch.size(0), -1), dim=1)
        
#         # Iterate through the batch to find the index of the overall maximum in the last dimension
#         for i, max_val in enumerate(max_vals):
#             # Get the positions of the overall maximum values
#             pos = (batch[i] == max_val).nonzero(as_tuple=True)
#             # Consider the last dimension (W)
#             max_index_last_dim = pos[-1]
#             # Calculate the distance to the closest edge in the last dimension
#             edge_distance = torch.min(max_index_last_dim, batch.shape[-1] - 1 - max_index_last_dim)
#             edge_distance = max_index_last_dim
#             if edge_distance < closest_edge_distance:
#                 closest_edge_distance = edge_distance
#                 closest_max_index = max_index_last_dim
#                 closest_tensor = batch[i]

    # # Now plot the 2D slice
    # sample_2d_slice = closest_tensor[:,32,:].cpu().detach().numpy()
    # plt.imshow(sample_2d_slice, cmap='gray')
    # plt.colorbar()
    # plt.title(f'2D slice of the sample with max value closest to the edge')
    # plt.show()

    # plt.savefig("images/test_closest_edge")

#     return closest_max_index.item(), closest_tensor
