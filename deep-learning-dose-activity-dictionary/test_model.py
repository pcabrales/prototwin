import torch
import torch.nn as nn
from tqdm import tqdm
from utils import RE_loss, range_loss, post_BP_loss

def test(trained_model, test_loader, device, results_dir='.',
         mean_output=0, std_output=1):
    # Test loop (after the training is complete)
    RE_loss_list = []
    l2_loss_list = []
    l2_loss = nn.MSELoss()
    R90_list = []
    R50_list = []
    R10_list = []
    post_BP_loss_list = []
    with torch.no_grad():
        for batch_input, batch_target in tqdm(test_loader):
            batch_input = batch_input.to(device)
            batch_output = trained_model(batch_input)
            batch_output = batch_output.detach().cpu()
            torch.cuda.empty_cache()
            RE_loss_list.append(RE_loss(batch_output, batch_target, mean_output=mean_output, std_output=std_output))
            l2_loss_list.append(l2_loss(batch_output, batch_target))
            R90_list.append(range_loss(batch_output, batch_target, 0.9, mean_output=mean_output, std_output=std_output))
            R50_list.append(range_loss(batch_output, batch_target, 0.5, mean_output=mean_output, std_output=std_output))
            R10_list.append(range_loss(batch_output, batch_target, 0.1, mean_output=mean_output, std_output=std_output))
            post_BP_loss_list.append(post_BP_loss(batch_output, batch_target, mean_output=mean_output, std_output=std_output))
    RE_loss_list = torch.cat(RE_loss_list)
    R90_list = torch.cat(R90_list)
    R50_list = torch.cat(R50_list)
    R10_list = torch.cat(R10_list)
    l2_loss_list = torch.tensor(l2_loss_list)
    post_BP_loss_list = torch.tensor(post_BP_loss_list)

    text_results = f"Relative Error: {torch.mean(RE_loss_list)} +- {torch.std(RE_loss_list)}\n" \
           f"R90: {torch.mean(R90_list)} +- {torch.std(R90_list)}\n" \
           f"R90 squared: {torch.mean(R90_list **2)}\n" \
           f"R50: {torch.mean(R50_list)} +- {torch.std(R50_list)}\n" \
           f"R50 squared: {torch.mean(R50_list **2)}\n" \
           f"R10: {torch.mean(R10_list)} +- {torch.std(R10_list)}\n" \
           f"R10 squared: {torch.mean(R10_list **2)}\n" \
           f"L2 Loss: {torch.mean(l2_loss_list)} +- {torch.std(l2_loss_list)}\n" \
           f"Post BP Loss: {torch.mean(post_BP_loss_list)} +- {torch.std(post_BP_loss_list)}"
    print(text_results)

    # Save to file
    with open(results_dir, "w") as file:
        file.write(text_results)
        
    return None



