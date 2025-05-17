from transformers import (
    AutoTokenizer,
    Trainer,
    TrainingArguments,
    TrainerCallback,
    DataCollatorForLanguageModeling
)
from peft import prepare_model_for_kbit_training, LoraConfig, get_peft_model
from datasets import load_dataset
import torch
import math
import tiktoken

# Import the MathQA benchmark function and our Dataset helpers.
from data.benchmarks import get_mathqa
from data.utils import Dataset, get_dataloader

import transformers

class FinetuneDataset(torch.utils.data.Dataset):
    def __init__(self, data, sequence_length):
        """
        data: a torch.Tensor or a NumPy array of token IDs.
        sequence_length: the fixed sequence length for each example.
        """
        self.data = data
        self.sequence_length = sequence_length

    def __len__(self):
        total_length = len(self.data)
        return (total_length - 1) // self.sequence_length

    def __getitem__(self, idx):
        start = idx * self.sequence_length
        end = start + self.sequence_length
        if isinstance(self.data, torch.Tensor):
            input_ids = self.data[start:end].long()
            labels = self.data[start + 1: end + 1].long()
        else:
            input_ids = torch.from_numpy(self.data[start:end]).long()
            labels = torch.from_numpy(self.data[start + 1: end + 1]).long()
        return {"input_ids": input_ids, "labels": labels}

def simple_collate_fn(features):
    new_features = []
    for i, f in enumerate(features):
        # If f is a tuple or list of length 2, convert it into a dict.
        if isinstance(f, dict) and "input_ids" in f and "labels" in f:
            new_features.append(f)
        else:
            print(f"Feature at index {i} is invalid: {f}")
    input_ids = torch.stack([f["input_ids"] for f in new_features])
    labels = torch.stack([f["labels"] for f in new_features])
    return {"input_ids": input_ids, "labels": labels}
    
def run_finetune(model, checkpoint=None):
    """
    Finetunes the model on MathQA.
      If a checkpoint path is provided, it is loaded into the model before finetuning.
    """
    # (Assume that the model is created externally and available in this module’s scope.)
    # For example, you might call your own `get_model` function here.
    # For this example, we assume that `model` is already defined (e.g. via an earlier import or creation).
    if checkpoint is not None:
        print(f"Loading checkpoint from {checkpoint}")
        ckpt = torch.load(checkpoint, map_location="cpu")
        if isinstance(ckpt, dict) and "model" in ckpt:
            # Full training state checkpoint
            model.load_state_dict(ckpt["model"])
        else:
            # Model-only checkpoint
            model.load_state_dict(ckpt)
    
    model.train()
    
    if hasattr(model, "gradient_checkpointing_enable"):
            model.gradient_checkpointing_enable()
    else:
        print(f"{model.__class__.__name__} does not support gradient_checkpointing_enable().")
        
    # Create minimal config for PEFT compatibility
    from transformers import PretrainedConfig
    
    # Create custom PretrainedConfig that preserves original attributes
    class LlamaPretrainedConfig(PretrainedConfig):
        def __init__(self, original_config=None, **kwargs):
            super().__init__(**kwargs)
            if original_config:
                for key, value in vars(original_config).items():
                    if not hasattr(self, key):
                        setattr(self, key, value)
    
    # Create HF-compatible wrapper for our model
    class LlamaWrapper(torch.nn.Module):
        def __init__(self, base_model):
            super().__init__()
            self.base_model = base_model
            # Store original config
            self.original_config = base_model.config
            self.config = None  # Will be set after PretrainedConfig is created
            self.generation_config = None
            self.can_generate = True
    
        def forward(self, input_ids=None, labels=None, **kwargs):
            # Use original config for the base model call
            self.base_model.config = self.original_config
            # Ensure input has batch dimension
            if input_ids is not None and input_ids.dim() == 1:
                input_ids = input_ids.unsqueeze(0)
            if labels is not None and labels.dim() == 1:
                labels = labels.unsqueeze(0)
            out = self.base_model(input_ids, targets=labels)
            # Restore HF config
            self.base_model.config = self.config
            # Ensure output matches HF's expected format
            return {"loss": out["loss"]} if "loss" in out else out
            
        def prepare_inputs_for_generation(self, input_ids, **kwargs):
            return {
                "idx": input_ids,
            }
            
        def prepare_for_generation(self):
            pass
            
        def post_process_generation(self, output, **kwargs):
            return output
            
        @property
        def device(self):
            return next(self.parameters()).device
    
    minimal_config = {
        "model_type": "llama",
        "architectures": ["LlamaForCausalLM"],
        "_name_or_path": "llama",
        "tie_word_embeddings": False,
        "hidden_size": 4096,
        "intermediate_size": 11008,
        "num_attention_heads": 32,
        "num_hidden_layers": 32,
        "pad_token_id": 0,
        "bos_token_id": 1,
        "eos_token_id": 2,
        "max_position_embeddings": 4096,
        "torch_dtype": "float32",
        "transformers_version": "4.31.0",
        "auto_map": {
            "AutoModelForCausalLM": "LlamaForCausalLM"
        }
    }
    # Create the wrapper first
    model = LlamaWrapper(model)
    
    # Then set both configs
    hf_config = LlamaPretrainedConfig(original_config=model.original_config, **minimal_config)
    model.config = hf_config
    model.base_model.config = hf_config

        # Prepare model for training
    model = prepare_model_for_kbit_training(model)
    
    # Configure LoRA for Llama architecture
    lora_config = LoraConfig(
        r=8,
        lora_alpha=32,
        target_modules=["c_attn", "c_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM"
    )
    
    model = get_peft_model(model, lora_config)
    # model.print_trainable_parameters()
    
    mathqa_data = get_mathqa(num_proc=1, return_torch=True, pad_to_multiple=1024)
    
    # (The MathQA evaluation callback remains unchanged.)
    class MathQAEvalCallback(TrainerCallback):
        def on_epoch_end(self, args, state, control, **kwargs):
            model = kwargs["model"]
            device = args.device
            model.eval()
            sequence_length = args.sequence_length if hasattr(args, "sequence_length") else 1024
            
            # Create our Dataset instance directly
            val_data = mathqa_data["val"]
            total_samples = (len(val_data) - 1) // sequence_length
            
            total_loss = 0.0
            count = 0
            with torch.no_grad():
                for i in range(0, total_samples, args.per_device_eval_batch_size):
                    batch_size = min(args.per_device_eval_batch_size, total_samples - i)
                    batch_losses = []
                    
                    for j in range(batch_size):
                        idx = (i + j) * sequence_length
                        x = val_data[idx:idx + sequence_length].to(torch.int64).unsqueeze(0).to(device)
                        y = val_data[idx + 1:idx + 1 + sequence_length].to(torch.int64).unsqueeze(0).to(device)
                        outputs = model(x, targets=y)
                        loss = outputs.get('loss')
                        if loss is not None and not torch.isnan(loss) and not torch.isinf(loss):
                            batch_losses.append(loss)
                    
                    if batch_losses:
                        batch_loss = torch.stack(batch_losses).mean()
                        if not torch.isnan(batch_loss) and not torch.isinf(batch_loss):
                            total_loss += batch_loss.item()
                            count += 1
                        
            avg_loss = total_loss / count if count > 0 else float('inf')
            perplexity = math.exp(avg_loss)
            print(f"\n=== Epoch {state.epoch} MathQA Evaluation ===")
            print(f"Validation Loss: {avg_loss:.4f} | Perplexity: {perplexity:.2f}\n")
            model.train()
            return control

    try:
        hf_tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-hf", use_fast=True)
    except:
        # Fallback to GPT2 tokenizer if Llama tokenizer is not available
        hf_tokenizer = AutoTokenizer.from_pretrained("gpt2", use_fast=True)
    hf_tokenizer.pad_token = hf_tokenizer.eos_token
    # Instead of using DataCollatorForLanguageModeling, we use our simple collate function.
    data_collator = simple_collate_fn
    train_dataset = FinetuneDataset(mathqa_data["train"], sequence_length=1024)
    
    training_args = TrainingArguments(
        output_dir="finetuned_mathqa",
        learning_rate=2e-4,
        per_device_train_batch_size=4,
        per_device_eval_batch_size=4,
        num_train_epochs=5,
        weight_decay=0.01,
        eval_strategy="epoch",
        save_strategy="epoch",
        logging_steps=100,
        load_best_model_at_end=True,
        gradient_accumulation_steps=4,
        warmup_steps=2,
        fp16=True,
        optim="adamw_torch",
        seed=1337,
    )    
    # initial_generation(model, "Problem: Solve 2+2 = ? Answer:")
    
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=train_dataset,
        callbacks=[MathQAEvalCallback]
    )
    
    # model.config.use_cache = False
    trainer.train()
    # model.config.use_cache = True
    trainer.save_model("finetuned_mathqa")