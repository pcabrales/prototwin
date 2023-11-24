import torch
import torch.nn as nn
from tqdm import tqdm
from utils import RE_loss, range_loss, post_BP_loss, gamma_index, pymed_gamma, plot_range_histogram
import numpy as np

def test(trained_model, test_loader, device, results_dir='.',
         mean_output=0, std_output=1, save_plot_dir='images/hist.png'):
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
    threshold = 0.1  # Minimum relative dose considered for gamma index
    tolerance = 0.03  # Tolerance per unit for gamma index
    
    with torch.no_grad():
        for batch_input, batch_target, _ in tqdm(test_loader):
            batch_input = batch_input.to(device)
            batch_output = trained_model(batch_input)
            batch_output = batch_output.detach().cpu()
            torch.cuda.empty_cache()
            RE_loss_list.append(RE_loss(batch_output, batch_target, mean_output=mean_output, std_output=std_output))   ### set it to absolute value
            l2_loss_list.append(l2_loss(batch_output, batch_target))
            R100_list.append(range_loss(batch_output, batch_target, 1.0, mean_output=mean_output, std_output=std_output))
            R90_list.append(range_loss(batch_output, batch_target, 0.9, mean_output=mean_output, std_output=std_output))
            R50_list.append(range_loss(batch_output, batch_target, 0.5, mean_output=mean_output, std_output=std_output))
            R10_list.append(range_loss(batch_output, batch_target, 0.1, mean_output=mean_output, std_output=std_output))
            gamma_list.append(gamma_index(batch_output, batch_target, tolerance=tolerance, beta=5, mean_output=mean_output, std_output=std_output, threshold=threshold))
            gamma_pymed_list = pymed_gamma(gamma_pymed_list, batch_output, batch_target, dose_percent_threshold=tolerance*100, 
                                          distance_mm_threshold=1, threshold=threshold, mean_output=mean_output, std_output=std_output)
           
            
    RE_loss_list = torch.cat(RE_loss_list)
    R100_list = torch.cat(R100_list)
    plot_range_histogram(R100_list, save_plot_dir)
    R90_list = torch.cat(R90_list)
    R50_list = torch.cat(R50_list)
    R10_list = torch.cat(R10_list)
    l2_loss_list = torch.tensor(l2_loss_list)
    gamma_list = torch.tensor(gamma_list)
    gamma_pymed_list = torch.tensor(gamma_pymed_list)
    
    text_results = f"Relative Error: {torch.mean(torch.abs(RE_loss_list))} +- {torch.std(torch.abs(RE_loss_list))}\n" \
           f"R100: {torch.mean(torch.abs(R100_list))} +- {torch.std(torch.abs(R100_list))}\n" \
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



