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
            labels = self.data[start + 1 : end + 1].long()
        else:
            input_ids = torch.from_numpy(self.data[start:end]).long()
            labels = torch.from_numpy(self.data[start + 1 : end + 1]).long()
        return {"input_ids": input_ids, "labels": labels}

def simple_collate_fn(features):
    # features is a list of dicts containing "input_ids" and "labels" of equal length.
    input_ids = torch.stack([f["input_ids"] for f in features])
    labels = torch.stack([f["labels"] for f in features])
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
        model.load_state_dict(ckpt)
    
    model.train()
    
    if hasattr(model, "gradient_checkpointing_enable"):
            model.gradient_checkpointing_enable()
    else:
        print(f"{model.__class__.__name__} does not support gradient_checkpointing_enable().")
        
    model = prepare_model_for_kbit_training(model)
    lora_config = LoraConfig(
        r=8,
        lora_alpha=32,
        target_modules=["c_attn"],
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM"
    )
    
    if not hasattr(model.config, "get"):
            model.config = model.config.__dict__
            
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    
    mathqa_data = get_mathqa(num_proc=1, return_torch=True, pad_to_multiple=1024)
    
    # (The MathQA evaluation callback remains unchanged.)
    class MathQAEvalCallback(TrainerCallback):
        def on_epoch_end(self, args, state, control, **kwargs):
            model = kwargs["model"]
            device = args.device
            model.eval()
            sequence_length = args.sequence_length if hasattr(args, "sequence_length") else 1024
            val_dataset = Dataset(mathqa_data["val"], sequence_length)
            eval_loader, _ = get_dataloader(val_dataset, sequence_length, batch_size=args.per_device_eval_batch_size, seed=args.seed)
            
            total_loss = 0.0
            count = 0
            with torch.no_grad():
                for batch in eval_loader:
                    x, y = batch
                    outputs = model(x.to(device), targets=y.to(device))
                    loss = outputs.get("loss", None)
                    if loss is not None:
                        total_loss += loss.item()
                        count += 1
            avg_loss = total_loss / count if count > 0 else float('inf')
            perplexity = math.exp(avg_loss)
            print(f"\n=== Epoch {state.epoch} MathQA Evaluation ===")
            print(f"Validation Loss: {avg_loss:.4f} | Perplexity: {perplexity:.2f}\n")
            model.train()
            return control

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
        num_train_epochs=10,
        weight_decay=0.01,
        eval_strategy="epoch",
        save_strategy="epoch",
        logging_steps=1,
        load_best_model_at_end=True,
        gradient_accumulation_steps=4,
        warmup_steps=2,
        fp16=True,
        optim="paged_adamw_8bit",
        seed=1337,
    )    
    # initial_generation(model, "Problem: Solve 2+2 = ? Answer:")
    
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=train_dataset,
        data_collator=data_collator,
        callbacks=[MathQAEvalCallback]
    )
    
    # model.config.use_cache = False
    trainer.train()
    # model.config.use_cache = True
    trainer.save_model("finetuned_mathqa")
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, default=None, help="Path to a checkpoint to load")
    args = parser.parse_args()
    run_finetune(checkpoint=args.checkpoint)