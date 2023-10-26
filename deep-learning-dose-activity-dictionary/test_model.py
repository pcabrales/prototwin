import torch
import torch.nn as nn
from tqdm import tqdm
from utils import RE_loss, range_loss

def test(trained_model, test_loader, device, results_dir='.'):
    # Test loop (after the training is complete)
    RE_loss_list = []
    l2_loss_list = []
    l2_loss = nn.MSELoss()
    R90_list = []
    R50_list = []
    R10_list = []
    with torch.no_grad():
        for batch_input, batch_target in tqdm(test_loader):
            batch_input = batch_input.to(device)
            batch_output = trained_model(batch_input)
            batch_output = batch_output.detach().cpu()
            torch.cuda.empty_cache()
            RE_loss_list.append(RE_loss(batch_output, batch_target))
            l2_loss_list.append(l2_loss(batch_output, batch_target))
            R90_list.append(range_loss(batch_output, batch_target, 0.9))
            R50_list.append(range_loss(batch_output, batch_target, 0.5))
            R10_list.append(range_loss(batch_output, batch_target, 0.1))

    RE_loss_list = torch.cat(RE_loss_list)
    l2_loss_list = torch.tensor(l2_loss_list)
    R90_list = torch.cat(R90_list)
    R50_list = torch.cat(R50_list)
    R10_list = torch.cat(R10_list)

    text_results = f"Relative Error: {torch.mean(RE_loss_list)} +- {torch.std(RE_loss_list)}\n" \
           f"R90: {torch.mean(R90_list)} +- {torch.std(R90_list)}\n" \
           f"R90 squared: {torch.mean(R90_list **2)}\n" \
           f"R50: {torch.mean(R50_list)} +- {torch.std(R50_list)}\n" \
           f"R50 squared: {torch.mean(R50_list **2)}\n" \
           f"R10: {torch.mean(R10_list)} +- {torch.std(R10_list)}\n" \
           f"R10 squared: {torch.mean(R10_list **2)}\n" \
           f"L2 Loss: {torch.mean(l2_loss_list)} +- {torch.std(l2_loss_list)}"
    print(text_results)

    # Save to file
    with open(results_dir, "w") as file:
        file.write(text_results)
        
    return None



