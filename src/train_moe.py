"""
Training script for Mixture of Experts models.

This script demonstrates:
1. How to set up and train a Mixture of Experts model
2. How to implement load balancing loss
3. How to visualize expert utilization
"""

import os
import argparse
import time
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

# Import our MoE implementation
from models.moe import SimpleMoEModel, MoETransformerBlock


def parse_args():
    parser = argparse.ArgumentParser(description="Train a Mixture of Experts model")
    
    # Model parameters
    parser.add_argument("--input_size", type=int, default=784, help="Input size")
    parser.add_argument("--hidden_size", type=int, default=512, help="Hidden size")
    parser.add_argument("--output_size", type=int, default=10, help="Output size (num classes)")
    parser.add_argument("--num_experts", type=int, default=8, help="Number of experts")
    parser.add_argument("--top_k", type=int, default=2, help="Number of experts to select")
    
    # Training parameters
    parser.add_argument("--batch_size", type=int, default=128, help="Batch size")
    parser.add_argument("--epochs", type=int, default=10, help="Number of epochs")
    parser.add_argument("--lr", type=float, default=0.001, help="Learning rate")
    parser.add_argument("--load_balancing_coef", type=float, default=0.01, 
                        help="Coefficient for the load balancing loss")
    parser.add_argument("--dropout", type=float, default=0.1, help="Dropout rate")
    
    # Other parameters
    parser.add_argument("--save_dir", type=str, default="./checkpoints", help="Directory to save checkpoints")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu", 
                        help="Device to use (cuda/cpu)")
    
    return parser.parse_args()


def prepare_mnist_data():
    """
    Download and prepare MNIST dataset as a simple example.
    """
    from torchvision import datasets, transforms
    
    # Download MNIST dataset
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
    
    train_dataset = datasets.MNIST('./data', train=True, download=True, transform=transform)
    test_dataset = datasets.MNIST('./data', train=False, transform=transform)
    
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)
    
    return train_loader, test_loader


def train_epoch(model, train_loader, optimizer, device, load_balancing_coef):
    """
    Train the model for one epoch.
    """
    model.train()
    total_loss = 0.0
    task_loss = 0.0
    balance_loss = 0.0
    correct = 0
    total = 0
    
    for batch_idx, (data, target) in enumerate(tqdm(train_loader, desc="Training")):
        data, target = data.to(device), target.to(device)
        
        # Flatten the input for our simple model
        data = data.view(data.size(0), -1)
        
        optimizer.zero_grad()
        
        # Forward pass
        output, gates = model(data)
        
        # Task loss (classification)
        classification_loss = F.cross_entropy(output, target)
        
        # Load balancing loss
        if isinstance(model.moe_layer.moe, torch.nn.Module) and hasattr(model.moe_layer.moe, 'load_balancing_loss'):
            lb_loss = model.moe_layer.moe.load_balancing_loss(gates)
            loss = classification_loss + load_balancing_coef * lb_loss
            balance_loss += lb_loss.item()
        else:
            loss = classification_loss
        
        # Backward pass
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        task_loss += classification_loss.item()
        
        # Calculate accuracy
        pred = output.argmax(dim=1)
        correct += pred.eq(target).sum().item()
        total += target.size(0)
        
    avg_loss = total_loss / len(train_loader)
    avg_task_loss = task_loss / len(train_loader)
    avg_balance_loss = balance_loss / len(train_loader)
    accuracy = 100.0 * correct / total
    
    return avg_loss, avg_task_loss, avg_balance_loss, accuracy


def evaluate(model, test_loader, device):
    """
    Evaluate the model on the test set.
    """
    model.eval()
    test_loss = 0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for data, target in tqdm(test_loader, desc="Evaluation"):
            data, target = data.to(device), target.to(device)
            
            # Flatten the input for our simple model
            data = data.view(data.size(0), -1)
            
            # Forward pass
            output, _ = model(data)
            
            # Task loss (classification)
            test_loss += F.cross_entropy(output, target).item()
            
            # Calculate accuracy
            pred = output.argmax(dim=1)
            correct += pred.eq(target).sum().item()
            total += target.size(0)
    
    avg_loss = test_loss / len(test_loader)
    accuracy = 100.0 * correct / total
    
    return avg_loss, accuracy


def plot_expert_usage(model, test_loader, device, num_experts, k=2):
    """
    Plot the expert usage statistics.
    """
    model.eval()
    
    # Reset expert counts
    if hasattr(model.moe_layer.moe, 'expert_counts'):
        model.moe_layer.moe.expert_counts.zero_()
    
    expert_frequency = torch.zeros(num_experts).to(device)
    num_samples = 0
    
    with torch.no_grad():
        for data, _ in tqdm(test_loader, desc="Collecting expert statistics"):
            data = data.to(device)
            
            # Flatten the input for our simple model
            data = data.view(data.size(0), -1)
            
            # Forward pass
            _, gates = model(data)
            
            # Count how often each expert is selected in top-k
            _, top_indices = torch.topk(gates, k=k, dim=1)
            
            for expert_idx in range(num_experts):
                # Count how many times this expert appears in top_indices
                expert_frequency[expert_idx] += (top_indices == expert_idx).sum().float()
                
            num_samples += data.size(0) * k
    
    # Normalize
    expert_frequency = expert_frequency / num_samples
    
    # Plot
    plt.figure(figsize=(10, 5))
    plt.bar(range(num_experts), expert_frequency.cpu().numpy())
    plt.xlabel('Expert ID')
    plt.ylabel('Usage Frequency')
    plt.title('Expert Usage Distribution')
    plt.savefig('expert_usage.png')
    print(f"Expert usage plot saved as 'expert_usage.png'")


def main(args):
    # Set random seed for reproducibility
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)
    
    # Create save directory if it doesn't exist
    os.makedirs(args.save_dir, exist_ok=True)
    
    # Prepare data
    train_loader, test_loader = prepare_mnist_data()
    
    # Create model
    model = SimpleMoEModel(
        input_size=args.input_size,
        hidden_size=args.hidden_size,
        output_size=args.output_size,
        num_experts=args.num_experts,
        k=args.top_k
    )
    model = model.to(args.device)
    
    # Create optimizer
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    
    # Train the model
    print(f"Starting training on {args.device}...")
    print(f"Model has {sum(p.numel() for p in model.parameters() if p.requires_grad):,} trainable parameters")
    
    best_accuracy = 0.0
    
    for epoch in range(1, args.epochs + 1):
        start_time = time.time()
        
        # Train for one epoch
        train_loss, train_task_loss, train_balance_loss, train_acc = train_epoch(
            model, train_loader, optimizer, args.device, args.load_balancing_coef
        )
        
        # Evaluate on the test set
        test_loss, test_acc = evaluate(model, test_loader, args.device)
        
        epoch_time = time.time() - start_time
        
        # Print metrics
        print(f"Epoch {epoch}/{args.epochs} | "
              f"Time: {epoch_time:.2f}s | "
              f"Train Loss: {train_loss:.4f} | "
              f"Task Loss: {train_task_loss:.4f} | "
              f"Balance Loss: {train_balance_loss:.4f} | "
              f"Train Acc: {train_acc:.2f}% | "
              f"Test Loss: {test_loss:.4f} | "
              f"Test Acc: {test_acc:.2f}%")
        
        # Save checkpoint if this is the best model so far
        if test_acc > best_accuracy:
            best_accuracy = test_acc
            checkpoint_path = os.path.join(args.save_dir, "best_model.pth")
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'test_accuracy': test_acc,
                'args': args,
            }, checkpoint_path)
            print(f"Saved best model checkpoint to {checkpoint_path}")
    
    print(f"Training complete! Best test accuracy: {best_accuracy:.2f}%")
    
    # Plot expert usage at the end
    plot_expert_usage(model, test_loader, args.device, args.num_experts, args.top_k)


if __name__ == "__main__":
    args = parse_args()
    main(args) 