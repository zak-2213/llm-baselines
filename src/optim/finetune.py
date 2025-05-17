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
    base_model = getattr(model, 'base_model', model)
    print("Text generations")
    print(base_model.generate_from_string("Solve the following", max_new_tokens=40))
    print(base_model.generate_from_string("2 + 2 ", max_new_tokens=3))
    print(base_model.generate_from_string("Hello, I am ", max_new_tokens=40))
    print(base_model.generate_from_string("John has 10 apples", max_new_tokens=40))
    
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
            # Always use get_logits=True to match HF interface expectations
            out = self.base_model(idx=input_ids, targets=labels, get_logits=True)
            # Restore HF config
            self.base_model.config = self.config
            # Format output to match HF expectations
            return {
                "loss": out["loss"] if "loss" in out else None,
                "logits": out["logits"] if "logits" in out else None
            }
            
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
    
    # Store original base model
    model = get_peft_model(model, lora_config)
    # model.print_trainable_parameters()
    
    mathqa_data = get_mathqa(num_proc=1, return_torch=True, pad_to_multiple=1024)
    
    # MathQA evaluation callback using base.py evaluation logic
    class MathQAEvalCallback(TrainerCallback):
        def on_epoch_end(self, args, state, control, **kwargs):
            model = kwargs["model"]
            device = args.device
            # model.eval()
            # base_model.load_state_dict(model.state_dict(), strict=False)
            # print("Text generations")
            # print(base_model.generate_from_string("Solve the following", max_new_tokens=40))
            # print(base_model.generate_from_string("2 + 2 ", max_new_tokens=3))
            # print(base_model.generate_from_string("Hello, I am ", max_new_tokens=40))
            # print(base_model.generate_from_string("John has 10 apples", max_new_tokens=40))
            sequence_length = args.sequence_length if hasattr(args, "sequence_length") else 1024
            
            # Create validation dataloader
            val_dataset = Dataset(mathqa_data["val"], sequence_length)
            eval_loader, _ = get_dataloader(val_dataset, sequence_length, batch_size=args.per_device_eval_batch_size, seed=args.seed)
            eval_iter = iter(eval_loader)

            # Use base.py evaluation logic
            loss_list_val, acc_list = [], []
            eval_steps = 24  # Same as in base.py
            
            with torch.no_grad():
                for _ in range(eval_steps):
                    try:
                        batch = next(eval_iter)
                    except StopIteration:
                        eval_iter = iter(eval_loader)
                        batch = next(eval_iter)
                        
                    x, y = batch
                    x = x.to(device)
                    y = y.to(device)
                    
                    outputs = model(idx=x, targets=y, get_logits=True)
                    val_loss = outputs.get('loss')
                    if val_loss is not None:
                        loss_list_val.append(val_loss)
                        if outputs.get('logits') is not None:
                            acc = (outputs['logits'].argmax(-1) == y).float().mean()
                            acc_list.append(acc)

            if loss_list_val:
                val_acc = torch.stack(acc_list).mean().item() if acc_list else 0.0
                val_loss = torch.stack(loss_list_val).mean().item()
                perplexity = math.exp(val_loss)
            else:
                val_acc = 0.0
                val_loss = float('inf')
                perplexity = float('inf')

            print(f"\n=== Epoch {state.epoch} MathQA Evaluation ===")
            print(f"Validation Loss: {val_loss:.4f} | Perplexity: {perplexity:.2f} | Accuracy: {val_acc:.4f}\n")
            
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
