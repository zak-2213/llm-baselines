import os
import argparse
import torch
import numpy as np
import random
from typing import List, Optional

from models.utils import get_model
from data.utils import get_dataset
import config


def set_seed(seed: int):
    """Set random seeds for reproducibility."""
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_model(args, checkpoint_path: str):
    """Load a trained model from checkpoint."""
    device = torch.device(args.device)
    
    # Create model architecture
    model = get_model(args).to(device)
    
    # Load weights
    if checkpoint_path:
        print(f"Loading model from {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location=device)
        
        # Handle different checkpoint formats
        if 'model' in checkpoint:
            model_state_dict = checkpoint['model']
        else:
            model_state_dict = checkpoint
            
        # Clean up state dict if needed (remove module. prefix from DDP)
        clean_state_dict = {}
        for k, v in model_state_dict.items():
            if k.startswith('module.'):
                clean_state_dict[k[7:]] = v
            elif k.startswith('_orig_mod.'):
                clean_state_dict[k[10:]] = v
            else:
                clean_state_dict[k] = v
                
        model.load_state_dict(clean_state_dict)
    
    model.eval()
    return model


def encode_prompt(prompt: str, args):
    """Tokenize the prompt."""
    # For actual implementation, you would need a tokenizer from your dataset
    # This is a placeholder; adapt to use your project's tokenization method
    data = get_dataset(args)
    
    if hasattr(data, 'tokenizer'):
        tokenizer = data.tokenizer
        return tokenizer.encode(prompt, return_tensors="pt")
    else:
        # Fallback to a simple character-level encoding for demo purposes
        print("Warning: No tokenizer found, using character-level encoding")
        chars = sorted(list(set(prompt)))
        char_to_idx = {ch: i for i, ch in enumerate(chars)}
        return torch.tensor([[char_to_idx[c] for c in prompt]], dtype=torch.long)


def decode_tokens(tokens: torch.Tensor, args):
    """Convert token IDs back to text."""
    # Similar to encode_prompt, adapt to use your project's detokenization method
    data = get_dataset(args)
    
    if hasattr(data, 'tokenizer'):
        tokenizer = data.tokenizer
        return tokenizer.decode(tokens.squeeze().tolist())
    else:
        # Fallback for demo purposes
        print("Warning: No tokenizer found, returning token IDs")
        return tokens.squeeze().tolist()


def generate(
    model,
    prompt: str,
    max_new_tokens: int = 100,
    temperature: float = 1.0,
    top_k: Optional[int] = None,
    top_p: Optional[float] = None,
    do_sample: bool = True,
    args = None
) -> str:
    """Generate text from the model given a prompt."""
    device = next(model.parameters()).device
    
    # Encode the prompt
    input_ids = encode_prompt(prompt, args).to(device)
    
    # Lists to store generated tokens
    generated_tokens = input_ids.clone()
    
    # Set model to evaluation mode
    model.eval()
    
    with torch.no_grad():
        for _ in range(max_new_tokens):
            # Get only the last context_length tokens if input is too long
            if generated_tokens.size(1) > args.sequence_length:
                input_chunk = generated_tokens[:, -args.sequence_length:]
            else:
                input_chunk = generated_tokens
                
            # Forward pass
            logits = model(input_chunk)
            
            # Get logits for the next token prediction (last token in the sequence)
            next_token_logits = logits[:, -1, :]
            
            # Apply temperature
            if temperature > 0:
                next_token_logits = next_token_logits / temperature
            
            # Apply top-k filtering
            if top_k is not None:
                indices_to_remove = next_token_logits < torch.topk(next_token_logits, top_k)[0][..., -1, None]
                next_token_logits[indices_to_remove] = float('-inf')
            
            # Apply top-p (nucleus) filtering
            if top_p is not None:
                sorted_logits, sorted_indices = torch.sort(next_token_logits, descending=True)
                cumulative_probs = torch.cumsum(torch.softmax(sorted_logits, dim=-1), dim=-1)
                
                # Remove tokens with cumulative probability above the threshold
                sorted_indices_to_remove = cumulative_probs > top_p
                # Shift the indices to the right to keep also the first token above the threshold
                sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                sorted_indices_to_remove[..., 0] = 0
                
                # Scatter sorted tensors to original indexing
                indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
                next_token_logits[indices_to_remove] = float('-inf')
            
            # Sample from the filtered distribution
            if do_sample:
                # Sample from the distribution
                probs = torch.softmax(next_token_logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)
            else:
                # Take the most likely token
                next_token = torch.argmax(next_token_logits, dim=-1, keepdim=True)
            
            # Append to the sequence
            generated_tokens = torch.cat((generated_tokens, next_token), dim=1)
            
            # Check for end of generation (if using an EOS token)
            # This needs to be adapted to your tokenizer's EOS token
            # if next_token.item() == tokenizer.eos_token_id:
            #     break
    
    # Decode the generated tokens back to text
    return decode_tokens(generated_tokens[0], args)


def get_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(allow_abbrev=False)
    parser.add_argument('--config_format', default='base', choices=config.registered_formats())
    parser.add_argument('--checkpoint', type=str, required=True, help='Path to the model checkpoint')
    parser.add_argument('--prompt', type=str, default='', help='Text prompt to start generation')
    parser.add_argument('--prompt_file', type=str, default=None, help='File containing the prompt')
    parser.add_argument('--max_new_tokens', type=int, default=100, help='Maximum number of new tokens to generate')
    parser.add_argument('--temperature', type=float, default=1.0, help='Sampling temperature')
    parser.add_argument('--top_k', type=int, default=None, help='Top-k sampling parameter')
    parser.add_argument('--top_p', type=float, default=None, help='Top-p sampling parameter')
    parser.add_argument('--seed', type=int, default=42, help='Random seed for reproducibility')
    parser.add_argument('--no_sample', action='store_true', help='Use greedy decoding instead of sampling')
    parser.add_argument('--output_file', type=str, default=None, help='File to save the generated text')
    
    args, rem_args = parser.parse_known_args()
    return config.parse_args_with_format(format=args.config_format, base_parser=parser, args=rem_args, namespace=args)


def main():
    """Main function for inference."""
    args = get_args()
    
    # Set the random seed
    set_seed(args.seed)
    
    # Set the device
    args.device = args.device if hasattr(args, 'device') else 'cuda' if torch.cuda.is_available() else 'cpu'
    device = torch.device(args.device)
    
    # Enable TF32 precision if available
    if 'cuda' in args.device:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    
    # Load the model
    checkpoint_path = args.checkpoint
    model = load_model(args, checkpoint_path)
    
    # Get the prompt
    prompt = args.prompt
    if args.prompt_file:
        with open(args.prompt_file, 'r', encoding='utf-8') as f:
            prompt = f.read().strip()
    
    if not prompt:
        prompt = input("Enter your prompt: ")
    
    # Generate text
    print(f"\nPrompt: {prompt}\n")
    print("Generating...\n")
    
    # Use the model's built-in generate method
    if hasattr(model, 'generate_from_string'):
        # GPTBase and Llama have this convenient method
        generated_text = model.generate_from_string(
            prompt,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            top_k=args.top_k
        )
    else:
        # Fallback for other model types
        # Tokenize the prompt using the model's tokenizer if available
        if hasattr(model, 'tokenizer'):
            input_ids = torch.tensor(model.tokenizer.encode(prompt, allowed_special={"<|endoftext|>"})).view(1, -1).to(device)
        else:
            # If no tokenizer is available, handle appropriately
            print("Warning: Model has no tokenizer. Using character-level encoding")
            chars = sorted(list(set(prompt)))
            char_to_idx = {ch: i for i, ch in enumerate(chars)}
            input_ids = torch.tensor([[char_to_idx[c] for c in prompt]], dtype=torch.long).to(device)
        
        # Use the model's generate method with the input_ids
        output_ids = model.generate(
            idx=input_ids,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            top_k=args.top_k
        )
        
        # Decode the output
        if hasattr(model, 'tokenizer'):
            generated_text = model.tokenizer.decode(output_ids.view(-1).cpu().numpy())
        else:
            # Fallback decoding
            idx_to_char = {i: ch for i, ch in enumerate(chars)}
            generated_text = ''.join([idx_to_char.get(idx, '') for idx in output_ids.view(-1).cpu().numpy()])
    
    print(f"Generated text:\n{generated_text}\n")
    
    # Save to file if requested
    if args.output_file:
        with open(args.output_file, 'w', encoding='utf-8') as f:
            f.write(generated_text)
        print(f"Output saved to {args.output_file}")


if __name__ == "__main__":
    main()
