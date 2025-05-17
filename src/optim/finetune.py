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

##################################################
# (Optional) If you want to see an initial generation.
def initial_generation(model, prompt):
    tokenizer = AutoTokenizer.from_pretrained("gpt2", use_fast=True)
    tokenizer.pad_token = tokenizer.eos_token
    inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
    outputs = model.generate(input_ids=inputs["input_ids"], max_new_tokens=140)
    gen_text = tokenizer.batch_decode(outputs, skip_special_tokens=True)[0]
    print("Initial generation:")
    print(gen_text)
##################################################

# Switch the model to training mode (model is assumed to be created via get_model)
# For illustration, we assume the model is already created and assigned to "model".
# (In practice, you may load your model using your get_model from models/utils.py)
# For example:
#   from models.utils import get_model
#   dummy_args = ...  # fill in dummy hyperparameters as needed
#   model = get_model(dummy_args)
#   if checkpoint is not None: model.load_state_dict(torch.load(checkpoint))
# Here we assume that has been done.

model.train()
model.gradient_checkpointing_enable()

# Prepare the model for k-bit/quantized training and wrap it with LoRA.
model = prepare_model_for_kbit_training(model)
lora_config = LoraConfig(
    r=8,
    lora_alpha=32,
    target_modules=["q_proj"],  # update as needed
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM"
)
model = get_peft_model(model, lora_config)
model.print_trainable_parameters()

# Load the MathQA dataset using your benchmarks.
# (We use the torch return type so that further processing is easier.)
# Note: get_mathqa returns a dict with keys: "train", "val", "train_len", "val_len"
mathqa_data = get_mathqa(num_proc=1, return_torch=True, pad_to_multiple=1024)

# Define a custom callback that evaluates on MathQA at the end of each epoch.
class MathQAEvalCallback(TrainerCallback):
    def on_epoch_end(self, args, state, control, **kwargs):
        model = kwargs["model"]
        device = args.device
        # Set model to evaluation.
        model.eval()
        # Create a Torch dataset from the MathQA validation tokens.
        # We use the Dataset class from data/utils.py.
        sequence_length = args.sequence_length if hasattr(args, "sequence_length") else 1024
        val_dataset = Dataset(mathqa_data["val"], sequence_length)
        # Use your helper to build a dataloader.
        eval_loader, _ = get_dataloader(val_dataset, sequence_length, batch_size=args.per_device_eval_batch_size, seed=args.seed)
        
        total_loss = 0.0
        count = 0
        # Disable gradient tracking.
        with torch.no_grad():
            for batch in eval_loader:
                x, y = batch  # x, y are torch tensors.
                # Forward pass; note that our model returns a dict with a "loss" key.
                outputs = model(x.to(device), targets=y.to(device))
                loss = outputs.get("loss", None)
                if loss is not None:
                    total_loss += loss.item()
                    count += 1
        avg_loss = total_loss / count if count > 0 else float('inf')
        perplexity = math.exp(avg_loss)
        print(f"\n=== Epoch {state.epoch} MathQA Evaluation ===")
        print(f"Validation Loss: {avg_loss:.4f} | Perplexity: {perplexity:.2f}\n")
        # Return control to Trainer.
        model.train()
        return control

##################################################
# Set up a Hugging Face tokenizer (we use GPT-2 here).
hf_tokenizer = AutoTokenizer.from_pretrained("gpt2", use_fast=True)
hf_tokenizer.pad_token = hf_tokenizer.eos_token

# Create a data collator for language modeling.
data_collator = DataCollatorForLanguageModeling(tokenizer=hf_tokenizer, mlm=False)

# For finetuning on MathQA via benchmarks, we need a dummy train dataset.
# (Here, we use the train split from mathqa_data to simulate training data.)
train_dataset = Dataset(mathqa_data["train"], sequence_length=1024)

# Define training hyperparameters.
training_args = TrainingArguments(
    output_dir="finetuned_mathqa",
    learning_rate=2e-4,
    per_device_train_batch_size=4,
    per_device_eval_batch_size=4,
    num_train_epochs=10,
    weight_decay=0.01,
    logging_strategy="epoch",
    evaluation_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=True,
    gradient_accumulation_steps=4,
    warmup_steps=2,
    fp16=True,
    optim="paged_adamw_8bit",
    seed=42,
    # Make sure to set the sequence length in args if needed by the eval callback.
    sequence_length=1024,
)

# Optionally, show an initial generation before finetuning.
initial_generation(model, "Problem: Solve 2+2 = ? Answer:")

# Create Trainer with our MathQA evaluation callback.
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=train_dataset,  # you can provide a separate evaluation dataset if available
    data_collator=data_collator,
    callbacks=[MathQAEvalCallback]
)

# Disable the cache for training (to silence any warnings).
model.config.use_cache = False
trainer.train()
model.config.use_cache = True

# Save the finetuned model.
trainer.save_model("finetuned_mathqa")