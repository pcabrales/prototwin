import torch
import torch.nn as nn
import time
from tqdm import tqdm
import csv
from livelossplot import PlotLosses
from utils import gamma_index
import matplotlib.pyplot as plt
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def train(model, train_loader, val_loader, epochs=10, model_dir='.', timing_dir = '.',
          save_plot_dir='.', mean_output=0, std_output=1, losses_dir='.'):
    start_time = time.time()  # Timing the training time
    # Initializing the optimizer for the model parameters
    optim = torch.optim.AdamW(model.parameters(), lr=0.001)
    liveloss = PlotLosses()  # Object to track validation and training losses across epochs
    l2_loss = nn.MSELoss()
    beta = 2e-2
    threshold = 0.2
    alpha = 0 / 1000  # ratio between the losses
    training_losses = []
    val_losses = [] 
    for epoch in range(epochs):
        logs = {}
        train_loss = 0.0
        val_loss = 0.0

        # Training loop
        batch = 0
        for batch_input, batch_target, _ in tqdm(train_loader):
            batch_input = batch_input.to(device)
            batch_target = batch_target.to(device)
            optim.zero_grad()  # resetting gradients
            batch_output = model(batch_input)  # generating images
            loss = l2_loss(batch_output, batch_target)
            if epoch * batch_target.shape[0] + batch > 150 and beta < 5:
                beta = beta * 1.03
            # loss = (1-alpha) * l2_loss(batch_output, batch_target) + alpha * (1 - gamma_index(batch_output, batch_target, beta=beta, mean_output=mean_output, std_output=std_output, threshold=threshold))
            loss.backward()  # backprop
            optim.step()
            train_loss += loss.item()
            batch += 1
            with open(timing_dir, "w") as file:
                file.write(f'epoch {epoch} batch {batch}\n')
                
        # Validation loop
        with torch.no_grad():
            for batch_input, batch_target, _ in tqdm(val_loader):
                batch_input = batch_input.to(device)
                batch_target = batch_target.to(device)
                batch_output = model(batch_input)
                loss = l2_loss(batch_output, batch_target)
                # loss = (1-alpha) * l2_loss(batch_output, batch_target) + alpha * (1 - gamma_index(batch_output, batch_target, beta=beta, mean_output=mean_output, std_output=std_output, threshold=threshold))
                val_loss += loss.item()

        # Calculate average losses (to make it independent of batch size)
        avg_train_loss = train_loss / len(train_loader)
        avg_val_loss = val_loss / len(val_loader)

        # Log the losses for plotting
        logs['log loss'] = avg_train_loss
        training_losses.append(avg_train_loss)
        logs['val_log loss'] = avg_val_loss
        val_losses.append(avg_val_loss)

        liveloss.update(logs)
        liveloss.send()
        
    # End and save timing
    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f'Training time: {elapsed_time} seconds')
    # Save to file
    with open(timing_dir, "w") as file:
        file.write(f'Training time: {elapsed_time} seconds')
        
    # Save losses
    with open(losses_dir, 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['Epoch', 'Training Loss', 'Validation Loss'])
        for epoch, (train_loss, val_loss) in enumerate(zip(training_losses, val_losses)):
            writer.writerow([epoch, train_loss, val_loss])

    torch.save(model, model_dir)
    plt.savefig(save_plot_dir, dpi=300, bbox_inches='tight')
    return model