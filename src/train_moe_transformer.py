"""
Training script for Mixture of Experts Transformer Language Model.

This script demonstrates how to train a transformer-based language model
that uses Mixture of Experts layers to improve parameter efficiency.
"""

import os
import math
import argparse
import time
import numpy as np
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.nn import functional as F

from models.moe_transformer import MoETransformer, MoETransformerConfig, moe_load_balancing_loss


def parse_args():
    parser = argparse.ArgumentParser(description="Train a MoE Transformer Language Model")
    
    # Model parameters
    parser.add_argument("--vocab_size", type=int, default=50257, help="Vocabulary size")
    parser.add_argument("--sequence_length", type=int, default=128, help="Maximum sequence length")
    parser.add_argument("--n_layer", type=int, default=6, help="Number of layers")
    parser.add_argument("--n_head", type=int, default=8, help="Number of attention heads")
    parser.add_argument("--n_embd", type=int, default=512, help="Embedding dimension")
    
    # MoE parameters
    parser.add_argument("--num_experts", type=int, default=8, help="Number of experts")
    parser.add_argument("--top_k", type=int, default=2, help="Number of experts to select")
    parser.add_argument("--moe_layers", type=str, default="all", 
                        help="Which layers to make MoE (all, even, odd, last4, etc.)")
    parser.add_argument("--capacity_factor", type=float, default=1.0, 
                        help="Capacity factor for expert assignment")
    
    # Training parameters
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size")
    parser.add_argument("--epochs", type=int, default=5, help="Number of epochs")
    parser.add_argument("--lr", type=float, default=0.001, help="Learning rate")
    parser.add_argument("--load_balancing_coef", type=float, default=0.01, 
                        help="Coefficient for the load balancing loss")
    parser.add_argument("--dropout", type=float, default=0.1, help="Dropout rate")
    parser.add_argument("--grad_norm_clip", type=float, default=1.0, help="Gradient clipping")
    
    # Other parameters
    parser.add_argument("--dataset", type=str, default="wikitext", choices=["wikitext", "c4"], 
                        help="Dataset to use for training")
    parser.add_argument("--save_dir", type=str, default="./moe_transformer_checkpoints", 
                        help="Directory to save checkpoints")
    parser.add_argument("--save_every", type=int, default=1000, help="Save every N steps")
    parser.add_argument("--eval_every", type=int, default=500, help="Evaluate every N steps")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu", 
                        help="Device to use (cuda/cpu)")
    
    return parser.parse_args()


def configure_moe_layers(n_layer, moe_layers_config):
    """Configure which transformer layers will use MoE."""
    if moe_layers_config == "all":
        return list(range(n_layer))
    elif moe_layers_config == "even":
        return [i for i in range(n_layer) if i % 2 == 0]
    elif moe_layers_config == "odd":
        return [i for i in range(n_layer) if i % 2 == 1]
    elif moe_layers_config == "last4":
        return list(range(n_layer - 4, n_layer))
    elif moe_layers_config.startswith("layer"):
        # Parse layer indices like "layer0,2,4"
        layers = moe_layers_config[5:].split(",")
        return [int(l) for l in layers]
    else:
        try:
            return [int(moe_layers_config)]
        except:
            print(f"Invalid MoE layers configuration: {moe_layers_config}, using 'all'")
            return list(range(n_layer))


def prepare_wikitext_data(args):
    """Download and prepare WikiText dataset."""
    from datasets import load_dataset
    
    # Load WikiText-103
    dataset = load_dataset('wikitext', 'wikitext-103-v1')
    
    # Tokenize the dataset
    tokenizer = torch.hub.load('huggingface/pytorch-transformers', 'tokenizer', 'gpt2')
    
    def tokenize_function(examples):
        return tokenizer(examples['text'])
    
    tokenized_datasets = dataset.map(
        tokenize_function, 
        batched=True, 
        remove_columns=["text"]
    )
    
    # Concatenate all texts and split into chunks of sequence_length
    def group_texts(examples):
        concatenated = {k: sum(examples[k], []) for k in examples.keys()}
        total_length = len(concatenated[list(concatenated.keys())[0]])
        
        # Drop the last chunk if it's too small
        total_length = (total_length // args.sequence_length) * args.sequence_length
        
        # Create chunks
        result = {
            k: [t[i:i+args.sequence_length] for i in range(0, total_length, args.sequence_length)]
            for k, t in concatenated.items()
        }
        
        # Create labels (which are the same as input_ids but shifted)
        result["labels"] = result["input_ids"].copy()
        
        return result
    
    lm_datasets = tokenized_datasets.map(
        group_texts,
        batched=True,
    )
    
    # Create data loaders
    train_dataset = lm_datasets["train"]
    val_dataset = lm_datasets["validation"]
    
    def collate_fn(examples):
        input_ids = torch.tensor([example["input_ids"] for example in examples])
        labels = torch.tensor([example["labels"] for example in examples])
        return {"input_ids": input_ids, "labels": labels}
    
    train_loader = DataLoader(
        train_dataset, 
        batch_size=args.batch_size, 
        shuffle=True,
        collate_fn=collate_fn
    )
    
    val_loader = DataLoader(
        val_dataset, 
        batch_size=args.batch_size,
        collate_fn=collate_fn
    )
    
    return train_loader, val_loader, tokenizer.vocab_size


def train_epoch(model, train_loader, optimizer, device, args):
    """Train the model for one epoch."""
    model.train()
    total_loss = 0.0
    lm_loss = 0.0
    balance_loss = 0.0
    
    # Set up progress bar
    pbar = tqdm(enumerate(train_loader), total=len(train_loader), desc=f"Training")
    
    for batch_idx, batch in pbar:
        # Move batch to device
        input_ids = batch["input_ids"].to(device)
        targets = batch["labels"].to(device)
        
        # Forward pass
        optimizer.zero_grad()
        outputs = model(input_ids, targets)
        
        # Get losses
        lm_loss_val = outputs["loss"]
        
        # Add load balancing loss
        routing_logits = outputs.get('routing_logits', None)
        lb_loss = 0.0
        if routing_logits and args.load_balancing_coef > 0:
            lb_loss = moe_load_balancing_loss(routing_logits, args.num_experts)
            loss = lm_loss_val + args.load_balancing_coef * lb_loss
            balance_loss += lb_loss.item()
        else:
            loss = lm_loss_val
        
        # Backward pass
        loss.backward()
        
        # Gradient clipping
        if args.grad_norm_clip > 0:
            nn.utils.clip_grad_norm_(model.parameters(), args.grad_norm_clip)
            
        # Update weights
        optimizer.step()
        
        # Bookkeeping
        total_loss += loss.item()
        lm_loss += lm_loss_val.item()
        
        # Update progress bar
        pbar.set_postfix({
            'loss': loss.item(),
            'ppl': math.exp(min(lm_loss_val.item(), 20))
        })
        
        # Save checkpoint at regular intervals
        global_step = batch_idx + 1
        if global_step % args.save_every == 0:
            save_checkpoint(model, optimizer, global_step, args)
        
        # Evaluate at regular intervals
        if global_step % args.eval_every == 0:
            eval_loss, eval_ppl = evaluate(model, val_loader, device)
            print(f"\nEvaluation - Loss: {eval_loss:.4f}, Perplexity: {eval_ppl:.2f}")
            model.train()
    
    # Return average losses for the epoch
    avg_loss = total_loss / len(train_loader)
    avg_lm_loss = lm_loss / len(train_loader)
    avg_balance_loss = balance_loss / len(train_loader) if balance_loss > 0 else 0
    
    return avg_loss, avg_lm_loss, avg_balance_loss


def evaluate(model, val_loader, device, max_batches=None):
    """Evaluate the model on the validation set."""
    model.eval()
    total_loss = 0
    total_batches = 0
    
    with torch.no_grad():
        for i, batch in enumerate(tqdm(val_loader, desc="Evaluation")):
            if max_batches and i >= max_batches:
                break
                
            # Move batch to device
            input_ids = batch["input_ids"].to(device)
            targets = batch["labels"].to(device)
            
            # Forward pass
            outputs = model(input_ids, targets)
            loss = outputs["loss"]
            
            total_loss += loss.item()
            total_batches += 1
    
    avg_loss = total_loss / total_batches
    perplexity = math.exp(min(avg_loss, 20))  # Cap perplexity for numerical stability
    
    return avg_loss, perplexity


def generate_sample(model, tokenizer, prompt="Once upon a time", max_length=100, device="cpu"):
    """Generate a sample text from the model."""
    if not hasattr(tokenizer, "encode"):
        # If we're using huggingface tokenizer
        input_ids = tokenizer(prompt, return_tensors="pt")["input_ids"].to(device)
    else:
        # If we're using tiktoken
        input_ids = torch.tensor(tokenizer.encode(prompt)).unsqueeze(0).to(device)
    
    # Generate text
    with torch.no_grad():
        output_ids = model.generate(
            input_ids, 
            max_new_tokens=max_length,
            temperature=0.8,
            top_k=50
        )
    
    # Decode the generated tokens
    if not hasattr(tokenizer, "decode"):
        # If we're using huggingface tokenizer
        generated_text = tokenizer.decode(output_ids[0], skip_special_tokens=True)
    else:
        # If we're using tiktoken
        generated_text = tokenizer.decode(output_ids[0].tolist())
    
    return generated_text


def save_checkpoint(model, optimizer, step, args):
    """Save a checkpoint of the model."""
    os.makedirs(args.save_dir, exist_ok=True)
    checkpoint_path = os.path.join(args.save_dir, f"checkpoint_{step}.pt")
    
    # Save model, optimizer, and other training state
    torch.save({
        'step': step,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'config': model.config.__dict__,
    }, checkpoint_path)
    
    print(f"Checkpoint saved at {checkpoint_path}")


def main(args):
    # Set random seed for reproducibility
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)
    
    # Load and prepare data
    print(f"Loading and preparing dataset: {args.dataset}")
    if args.dataset == "wikitext":
        train_loader, val_loader, vocab_size = prepare_wikitext_data(args)
        args.vocab_size = vocab_size
    else:
        raise NotImplementedError(f"Dataset {args.dataset} not implemented yet")
        
    # Configure MoE layers
    moe_layers = configure_moe_layers(args.n_layer, args.moe_layers)
    print(f"Using MoE for layers: {moe_layers}")
    
    # Create model configuration
    config = MoETransformerConfig(
        vocab_size=args.vocab_size,
        sequence_length=args.sequence_length,
        n_layer=args.n_layer,
        n_head=args.n_head,
        n_embd=args.n_embd,
        dropout=args.dropout,
        # MoE specific parameters
        num_experts=args.num_experts,
        top_k=args.top_k,
        capacity_factor=args.capacity_factor,
        moe_layers=moe_layers
    )
    
    # Create model
    print("Creating model...")
    model = MoETransformer(config)
    model = model.to(args.device)
    
    # Create optimizer
    optimizer = optim.AdamW(model.parameters(), lr=args.lr)
    
    # Training loop
    print(f"Starting training on {args.device}...")
    best_val_loss = float('inf')
    
    for epoch in range(args.epochs):
        start_time = time.time()
        
        # Train for one epoch
        train_loss, train_lm_loss, train_balance_loss = train_epoch(
            model, train_loader, optimizer, args.device, args
        )
        
        # Evaluate
        val_loss, val_ppl = evaluate(model, val_loader, args.device)
        
        # Calculate epoch time
        epoch_time = time.time() - start_time
        
        # Print metrics
        print(f"Epoch {epoch+1}/{args.epochs} | "
              f"Time: {epoch_time:.2f}s | "
              f"Train Loss: {train_loss:.4f} | "
              f"LM Loss: {train_lm_loss:.4f} | "
              f"Balance Loss: {train_balance_loss:.8f} | "
              f"Val Loss: {val_loss:.4f} | "
              f"Val PPL: {val_ppl:.2f}")
        
        # Save if it's the best model so far
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            save_checkpoint(model, optimizer, f"best_epoch{epoch+1}", args)
            
        # Generate and print a sample
        try:
            print("\nGenerated Sample:")
            sample = generate_sample(model, model.tokenizer, device=args.device)
            print(f"{sample}\n")
        except Exception as e:
            print(f"Error generating sample: {e}")
    
    print(f"Training complete! Best validation loss: {best_val_loss:.4f}")


if __name__ == "__main__":
    args = parse_args()
    main(args) 