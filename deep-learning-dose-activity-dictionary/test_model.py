import torch
import torch.nn as nn
from tqdm import tqdm
from utils import RE_loss, range_loss, post_BP_loss, gamma_index
import pymedphys
import numpy as np

def test(trained_model, test_loader, device, results_dir='.',
         mean_output=0, std_output=1):
    # Test loop (after the training is complete)
    RE_loss_list = []
    l2_loss_list = []
    l2_loss = nn.MSELoss()
    R100_list = []
    R90_list = []
    R50_list = []
    R10_list = []
    gamma_list = []
    gamma_pymed_list = []
    threshold = 0.1
    
    with torch.no_grad():
        for batch_input, batch_target, _ in tqdm(test_loader):
            batch_input = batch_input.to(device)
            batch_output = trained_model(batch_input)
            batch_output = batch_output.detach().cpu()
            torch.cuda.empty_cache()
            RE_loss_list.append(torch.abs(RE_loss(batch_output, batch_target, mean_output=mean_output, std_output=std_output)))   ### set it to absolute value
            l2_loss_list.append(l2_loss(batch_output, batch_target))
            R100_list.append(range_loss(batch_output, batch_target, 1.0, mean_output=mean_output, std_output=std_output))
            R90_list.append(range_loss(batch_output, batch_target, 0.9, mean_output=mean_output, std_output=std_output))
            R50_list.append(range_loss(batch_output, batch_target, 0.5, mean_output=mean_output, std_output=std_output))
            R10_list.append(range_loss(batch_output, batch_target, 0.1, mean_output=mean_output, std_output=std_output))
            gamma_list.append(gamma_index(batch_output, batch_target, tolerance=0.03, beta=5, mean_output=mean_output, std_output=std_output, threshold=threshold))
            
            # for pymed gamma calculation:
            for idx in range(batch_input.shape[0]):
                output = batch_output[idx].unsqueeze(0)
                target = batch_target[idx].unsqueeze(0)
                output = mean_output + output * std_output  # undoing normalization
                target = mean_output + target * std_output
                axes_reference = (np.arange(output.shape[2]), np.arange(output.shape[3]), np.arange(output.shape[4]))    
                gamma = pymedphys.gamma(
                    axes_reference, target.squeeze(0).squeeze(0).cpu().detach().numpy(), 
                    axes_reference, output.squeeze(0).squeeze(0).cpu().detach().numpy(),
                    dose_percent_threshold = 3,
                    distance_mm_threshold = 1, 
                    lower_percent_dose_cutoff=threshold*100)
                valid_gamma = gamma[~np.isnan(gamma)]
                pass_ratio = np.sum(valid_gamma <= 1) / len(valid_gamma)
                gamma_pymed_list.append(pass_ratio)
           
            
    RE_loss_list = torch.cat(RE_loss_list)
    R100_list = torch.cat(R100_list)
    R90_list = torch.cat(R90_list)
    R50_list = torch.cat(R50_list)
    R10_list = torch.cat(R10_list)
    l2_loss_list = torch.tensor(l2_loss_list)
    gamma_list = torch.tensor(gamma_list)
    gamma_pymed_list = torch.tensor(gamma_pymed_list)
    
    text_results = f"Relative Error: {torch.mean(RE_loss_list)} +- {torch.std(RE_loss_list)}\n" \
           f"R100: {torch.mean(R100_list)} +- {torch.std(R100_list)}\n" \
           f"R90: {torch.mean(R90_list)} +- {torch.std(R90_list)}\n" \
           f"R50: {torch.mean(R50_list)} +- {torch.std(R50_list)}\n" \
           f"R10: {torch.mean(R10_list)} +- {torch.std(R10_list)}\n" \
           f"L2 Loss: {torch.mean(l2_loss_list)} +- {torch.std(l2_loss_list)}\n" \
           f"Gamma index: {torch.mean(gamma_list)} +- {torch.std(gamma_list)}\n" \
           f"Gamma index (good one - pymed one): {torch.mean(gamma_pymed_list)} +- {torch.std(gamma_pymed_list)}"
    print(text_results)

    # Save to file
    with open(results_dir, "w") as file:
        file.write(text_results)
        
    return None



