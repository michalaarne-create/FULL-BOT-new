"""
Behavior Cloning - pretrain policy na ludzkich danych.

Uczy mapowanie: observation → action (supervised learning)
"""

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
import json
from tqdm import tqdm

class BCPolicy(nn.Module):
    """Prosta sieć do BC - tylko kordy (Phase 1)"""
    
    def __init__(self, obs_dim=2, action_dim=2, hidden_dim=128):
        super().__init__()
        
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim),
            nn.Tanh()  # Output w [-1, 1]
        )
    
    def forward(self, obs):
        return self.net(obs)

class BCDataset(Dataset):
    """Dataset dla BC"""
    
    def __init__(self, observations, actions):
        self.observations = torch.FloatTensor(observations)
        self.actions = torch.FloatTensor(actions)
    
    def __len__(self):
        return len(self.observations)
    
    def __getitem__(self, idx):
        return self.observations[idx], self.actions[idx]

def train_bc(dataset_path='data/processed/hover_dataset.npz', 
             epochs=50, 
             batch_size=64, 
             lr=1e-3,
             output_dir='models/saved'):
    """Trenuj BC policy"""
    
    # Load dataset
    print(f"📂 Loading dataset: {dataset_path}")
    data = np.load(dataset_path)
    observations = data['observations']
    actions = data['actions']
    
    print(f"   Samples: {len(observations)}")
    print(f"   Obs shape: {observations.shape}")
    print(f"   Action shape: {actions.shape}")
    
    # Split train/val (90/10)
    split_idx = int(0.9 * len(observations))
    train_obs, val_obs = observations[:split_idx], observations[split_idx:]
    train_act, val_act = actions[:split_idx], actions[split_idx:]
    
    train_dataset = BCDataset(train_obs, train_act)
    val_dataset = BCDataset(val_obs, val_act)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)
    
    print(f"   Train samples: {len(train_dataset)}")
    print(f"   Val samples: {len(val_dataset)}")
    
    # Model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\n🔧 Device: {device}")
    
    model = BCPolicy(obs_dim=2, action_dim=2, hidden_dim=128).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()
    
    print(f"📊 Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Training
    print(f"\n🚀 Starting training - {epochs} epochs\n")
    
    best_val_loss = float('inf')
    history = {'train_loss': [], 'val_loss': []}
    
    for epoch in range(epochs):
        # Train
        model.train()
        train_losses = []
        
        for obs_batch, act_batch in train_loader:
            obs_batch = obs_batch.to(device)
            act_batch = act_batch.to(device)
            
            pred_actions = model(obs_batch)
            loss = criterion(pred_actions, act_batch)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            train_losses.append(loss.item())
        
        # Validation
        model.eval()
        val_losses = []
        
        with torch.no_grad():
            for obs_batch, act_batch in val_loader:
                obs_batch = obs_batch.to(device)
                act_batch = act_batch.to(device)
                
                pred_actions = model(obs_batch)
                loss = criterion(pred_actions, act_batch)
                val_losses.append(loss.item())
        
        train_loss = np.mean(train_losses)
        val_loss = np.mean(val_losses)
        
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        
        # Print progress
        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(f"Epoch {epoch+1:3d}/{epochs} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")
        
        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            output_path = Path(output_dir)
            output_path.mkdir(parents=True, exist_ok=True)
            
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_loss': train_loss,
                'val_loss': val_loss,
            }, output_path / 'bc_policy_best.pt')
    
    print(f"\n✅ Training complete!")
    print(f"   Best val loss: {best_val_loss:.4f}")
    print(f"   Model saved to: {output_dir}/bc_policy_best.pt")
    
    # Save history
    history_path = Path(output_dir) / 'bc_training_history.json'
    with open(history_path, 'w') as f:
        json.dump(history, f, indent=2)
    
    return model, history


if __name__ == "__main__":
    train_bc(
        dataset_path='data/processed/hover_dataset.npz',
        epochs=50,
        batch_size=64,
        lr=1e-3
    )