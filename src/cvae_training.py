import argparse
import pandas as pd
import torch
import torch.optim as optim
import torch.nn.functional as F  # Add this line

import pyro
import pyro.distributions as dist
from torch.utils.data import DataLoader, TensorDataset
from auction_cvae import CVAE

def load_data(data_path):
    df = pd.read_csv(data_path)
    contexts = df[['context1', 'context2', 'context3', 'context4']].values
    true_CTRs = df[['true_CTR']].values
    bids = df[['bid']].values
    outcomes = df[['outcome']].values
    return contexts, true_CTRs, bids, outcomes

def train_cvae(data_path, epochs, batch_size, lr, model_path):
    # Load and preprocess data
    data = pd.read_csv(data_path)
    context = torch.tensor(data[['context1', 'context2', 'context3', 'context4']].values, dtype=torch.float32)
    items = torch.tensor(data[['true_CTR', 'bid']].values, dtype=torch.float32)
    
    dataset = TensorDataset(context, items)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # Define model parameters
    context_dim = context.shape[1]
    item_dim = items.shape[1]
    z_dim = 20  # You can adjust this

    # Initialize model, optimizer, and loss function
    model = CVAE(context_dim=context_dim, item_dim=item_dim, z_dim=z_dim, use_cuda=False)
    optimizer = optim.Adam(model.parameters(), lr=lr)

    # Training loop
    for epoch in range(epochs):
        total_loss = 0
        for batch_context, batch_items in dataloader:
            optimizer.zero_grad()
            recon_batch, mu, logvar = model(batch_context, batch_items)
            loss = model.loss_function(recon_batch, batch_items, mu, logvar)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        
        print(f'Epoch {epoch+1}/{epochs}, Loss: {total_loss/len(dataloader):.4f}')

    # Save the model
    torch.save(model.state_dict(), model_path)
    print(f"Model saved to {model_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train CVAE model")
    parser.add_argument("--data_path", type=str, required=True, help="Path to the CSV data file")
    parser.add_argument("--epochs", type=int, default=100, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=64, help="Batch size for training")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--model_path", type=str, required=True, help="Path to save the trained model")

    args = parser.parse_args()
    train_cvae(args.data_path, args.epochs, args.batch_size, args.lr, args.model_path)