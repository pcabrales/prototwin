import torch
import torch.nn as nn
import time
from tqdm import tqdm
from livelossplot import PlotLosses
from utils import RE_loss
from utils import range_loss
import matplotlib.pyplot as plt
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def train(model, train_loader, val_loader, epochs=10, model_dir='.', timing_dir = '.',
          save_plot_dir='.'):
    start_time = time.time()  # Timing the training time
    # Initializing the optimizer for the model parameters
    optim = torch.optim.AdamW(model.parameters(), lr=0.001)
    liveloss = PlotLosses()  # Object to track validation and training losses across epochs
    l2_loss = nn.MSELoss()
    e=0###
    for epoch in range(epochs):
        logs = {}
        train_loss = 0.0
        val_loss = 0.0

        # Training loop
        b=0###
        for batch_input, batch_target in tqdm(train_loader):
            batch_input = batch_input.to(device)
            batch_target = batch_target.to(device)
            optim.zero_grad()  # resetting gradients
            batch_output = model(batch_input)  # generating images
            loss = l2_loss(batch_output, batch_target)
            loss.backward()  # backprop
            optim.step()
            train_loss += loss.item()
            b+=1###
            with open(timing_dir, "w") as file:###
                file.write(f'epoch {e} batch {b}\n')###
        e+=1###
        # Validation loop
        with torch.no_grad():
            for batch_input, batch_target in tqdm(val_loader):
                batch_input = batch_input.to(device)
                batch_target = batch_target.to(device)
                batch_output = model(batch_input)
                loss = l2_loss(batch_output, batch_target)
                val_loss += loss.item()

        # Calculate average losses (to make it independent of batch size)
        avg_train_loss = train_loss / len(train_loader)
        avg_val_loss = val_loss / len(val_loader)

        # Log the losses for plotting
        logs['log loss'] = avg_train_loss
        logs['val_log loss'] = avg_val_loss

        liveloss.update(logs)
        liveloss.send()
    # End and save timing
    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f'Training time: {elapsed_time} seconds')
    # Save to file
    with open(timing_dir, "w") as file:
        file.write(f'Training time: {elapsed_time} seconds')

    torch.save(model, model_dir)
    plt.savefig(save_plot_dir, dpi=300, bbox_inches='tight')
    return model