"""
Evaluation script for running the MoE model on the MathQA dataset.

This script handles loading the MathQA dataset and evaluating our MoE model on it.
"""

import os
import json
import argparse
import torch
import numpy as np
from tqdm import tqdm
from datasets import load_dataset
from transformers import AutoTokenizer

# Import our MoE models
from models.moe import SimpleMoEModel
from models.moe_transformer import MoETransformer, MoETransformerConfig


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate MoE model on MathQA dataset")
    
    # Model parameters
    parser.add_argument("--model_type", type=str, default="transformer", choices=["simple", "transformer"],
                        help="Type of MoE model to use")
    parser.add_argument("--model_path", type=str, default=None, help="Path to saved model checkpoint")
    parser.add_argument("--vocab_size", type=int, default=50257, help="Vocabulary size for transformer model")
    parser.add_argument("--n_layer", type=int, default=12, help="Number of layers for transformer model")
    parser.add_argument("--n_head", type=int, default=12, help="Number of attention heads for transformer model")
    parser.add_argument("--n_embd", type=int, default=768, help="Embedding dimension for transformer model")
    parser.add_argument("--sequence_length", type=int, default=1024, help="Max sequence length")
    
    # MoE parameters
    parser.add_argument("--num_experts", type=int, default=8, help="Number of experts")
    parser.add_argument("--top_k", type=int, default=2, help="Number of experts to select")
    parser.add_argument("--moe_layers", type=str, default="all", 
                        help="Which layers to make MoE (all, even, odd, last4, etc.)")
    
    # Dataset parameters
    parser.add_argument("--dataset_name", type=str, default="miike-ai/mathqa", 
                        help="Hugging Face dataset name")
    parser.add_argument("--max_examples", type=int, default=None, 
                        help="Maximum number of examples to evaluate")
    
    # Evaluation parameters
    parser.add_argument("--batch_size", type=int, default=4, help="Batch size for evaluation")
    parser.add_argument("--output_dir", type=str, default="./mathqa_results", 
                        help="Directory to save evaluation results")
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


def load_model(args):
    """Load the MoE model based on arguments."""
    if args.model_type == "simple":
        # For simple MoE model, we would need to define input size and output size
        # This is just a placeholder, needs to be adapted to actual requirements
        input_size = args.n_embd  # Placeholder 
        hidden_size = args.n_embd
        output_size = args.vocab_size
        
        model = SimpleMoEModel(
            input_size=input_size,
            hidden_size=hidden_size,
            output_size=output_size,
            num_experts=args.num_experts,
            k=args.top_k
        )
    else:  # transformer
        # Configure MoE layers
        moe_layers = configure_moe_layers(args.n_layer, args.moe_layers)
        
        # Create model configuration
        config = MoETransformerConfig(
            vocab_size=args.vocab_size,
            sequence_length=args.sequence_length,
            n_layer=args.n_layer,
            n_head=args.n_head,
            n_embd=args.n_embd,
            num_experts=args.num_experts,
            top_k=args.top_k,
            moe_layers=moe_layers
        )
        
        # Create model
        model = MoETransformer(config)
    
    # Load checkpoint if provided
    if args.model_path and os.path.exists(args.model_path):
        print(f"Loading model from {args.model_path}")
        checkpoint = torch.load(args.model_path, map_location=args.device)
        if 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
        else:
            model.load_state_dict(checkpoint)
    
    model = model.to(args.device)
    model.eval()  # Set to evaluation mode
    
    return model


def load_tokenizer(args):
    """Load the appropriate tokenizer."""
    if args.model_type == "transformer":
        # Use GPT-2 tokenizer for the transformer model
        tokenizer = AutoTokenizer.from_pretrained("gpt2")
        tokenizer.pad_token = tokenizer.eos_token
    else:
        # For the simple model, we just need a basic tokenizer that works with 
        # raw text (this is just a placeholder and needs to be adapted)
        tokenizer = AutoTokenizer.from_pretrained("gpt2")
        tokenizer.pad_token = tokenizer.eos_token
    
    return tokenizer


def load_mathqa_dataset(args):
    """Load the MathQA dataset from Hugging Face."""
    dataset = load_dataset(args.dataset_name)
    
    # If there's a specific split we want to use, specify it here
    if "train" in dataset:
        data = dataset["train"]
    else:
        # Take the first split if "train" is not available
        split_name = list(dataset.keys())[0]
        data = dataset[split_name]
    
    # Limit the number of examples if specified
    if args.max_examples and args.max_examples < len(data):
        indices = np.random.choice(len(data), args.max_examples, replace=False)
        data = data.select(indices)
    
    return data


def preprocess_example(example, tokenizer):
    """Preprocess a MathQA example for the model."""
    # For MathQA, we typically have a question and need to generate an answer
    prompt = example["question"]
    
    # Tokenize the prompt
    inputs = tokenizer(prompt, return_tensors="pt", padding=True, truncation=True, 
                       max_length=1024)
    
    return {
        "prompt": prompt,
        "input_ids": inputs["input_ids"],
        "attention_mask": inputs["attention_mask"] if "attention_mask" in inputs else None,
        "ground_truth": example["answer"]
    }


def evaluate_model(model, data, tokenizer, args):
    """Evaluate the model on the MathQA dataset."""
    results = []
    correct = 0
    total = 0
    
    # Check if we're using the transformer model with generate method
    has_generate = hasattr(model, 'generate') and callable(getattr(model, 'generate'))
    
    for i in tqdm(range(0, len(data), args.batch_size), desc="Evaluating"):
        batch_data = data[i:i+args.batch_size]
        batch_examples = [preprocess_example(example, tokenizer) for example in batch_data]
        
        # Prepare batch inputs
        batch_input_ids = torch.cat([ex["input_ids"] for ex in batch_examples], dim=0).to(args.device)
        
        # Forward pass depends on the model type
        with torch.no_grad():
            if has_generate:
                # For transformer models with generate method
                outputs = model.generate(
                    batch_input_ids,
                    max_new_tokens=100,
                    temperature=0.7,
                    top_k=50
                )
                
                # Decode generated tokens
                generated_texts = [tokenizer.decode(output[len(input_ids):], skip_special_tokens=True) 
                                  for output, input_ids in zip(outputs, batch_input_ids)]
            else:
                # For simple models or transformer models without generate
                if args.model_type == "simple":
                    # Simple MoE model expects flattened inputs, adapt as needed
                    outputs, _ = model(batch_input_ids.view(batch_input_ids.size(0), -1))
                    # Process outputs as needed, e.g., get predicted class if it's a classification task
                    predicted_answers = [str(output.argmax(-1).item()) for output in outputs]
                else:
                    # Default transformer forward pass
                    outputs = model(batch_input_ids)
                    logits = outputs["logits"]
                    
                    # Process logits to get predictions (example for next token prediction)
                    next_token_logits = logits[:, -1, :]
                    next_token_ids = torch.argmax(next_token_logits, dim=-1)
                    predicted_tokens = [tokenizer.decode([token_id.item()]) for token_id in next_token_ids]
                    generated_texts = predicted_tokens  # Simple approximation
        
        # Evaluate predictions against ground truth
        for j, example in enumerate(batch_examples):
            if j < len(generated_texts):  # Ensure we have a prediction
                prediction = generated_texts[j].strip()
                ground_truth = example["ground_truth"].strip()
                
                # Check if prediction is correct (this might need more sophisticated comparison)
                is_correct = prediction == ground_truth
                
                if is_correct:
                    correct += 1
                total += 1
                
                # Store results
                results.append({
                    "prompt": example["prompt"],
                    "prediction": prediction,
                    "ground_truth": ground_truth,
                    "correct": is_correct
                })
    
    accuracy = correct / total if total > 0 else 0
    print(f"Evaluation Results: {correct}/{total} correct, Accuracy: {accuracy:.4f}")
    
    return results, accuracy


def save_results(results, accuracy, args):
    """Save evaluation results to a file."""
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
    
    # Save detailed results
    results_path = os.path.join(args.output_dir, "mathqa_results.json")
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    
    # Save summary
    summary_path = os.path.join(args.output_dir, "mathqa_summary.json")
    with open(summary_path, "w") as f:
        summary = {
            "accuracy": accuracy,
            "model_type": args.model_type,
            "num_experts": args.num_experts,
            "top_k": args.top_k,
            "total_examples": len(results)
        }
        json.dump(summary, f, indent=2)
    
    print(f"Results saved to {args.output_dir}")


def main():
    args = parse_args()
    
    # Set random seed for reproducibility
    torch.manual_seed(42)
    np.random.seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(42)
    
    # Load model
    model = load_model(args)
    
    # Load tokenizer
    tokenizer = load_tokenizer(args)
    
    # Load dataset
    data = load_mathqa_dataset(args)
    
    # Evaluate model
    results, accuracy = evaluate_model(model, data, tokenizer, args)
    
    # Save results
    save_results(results, accuracy, args)


if __name__ == "__main__":
    main() 