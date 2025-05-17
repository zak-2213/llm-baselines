# 🧮 Corpus: A Math-Specialized Language Model from Scratch

Large language models can write poems, generate code, and even explain scientific concepts — but they consistently **struggle with math**. In this hackathon, we set out to tackle that problem by training an LLM from scratch, designed specifically to reason through **math word problems**, with a finetunned for the [MathQA dataset].

---
## How to Run
Install requirements:

`pip install -r requirements.txt`

Run training script

`python ./src/main.py`

## 💡 Our Approach

We combined **data-centric methods**, **training strategies**, and a variety of **optimization techniques** to build and accelerate the model. Here's what we explored:

### 🧪 Techniques We Tried

- **Corpus Ordering / Curriculum Training**  
  We organized our data by source, feeding the model progressively more specialized content. We followed insights from the paper _"On the Role of Corpus Ordering in Language Modeling"_ (Portland State University), starting with:
  
RedPajamaC4 → Wikipedia → GitHub → ArXiv → Books → StackExchange
This aimed to build context early, then gradually shift to more technical math data.

- **Grouped Query Attention (GQA)**  
Implemented to improve attention efficiency with reduced compute, inspired by optimizations in models like LLaMA 2.

- **Quantization (int8)**  
Drastically improved training speed (~200K tokens/sec), but hurt model accuracy — so we dropped it after testing.

- **`torch.compile`**  
Promising in theory, but caused CUDA compatibility issues during training, so we reverted.

- **Rotary Positional Embeddings (RoPE) with Complex Numbers**  
We experimented with using complex numbers directly instead of sine/cosine for the rotary embeddings, while maintaining the RoPE mechanism.

- **Activation Function: SwiGLU**  
Retained as-is, as it's currently the most optimal for transformer-based LLMs.

- **Training Strategy**  
- 80% full training on a mixed dataset  
- 20% fine-tuning on MathQA  
- We avoided repeating the MathQA dataset unnecessarily due to its limited size.

---

## 📊 Results

- **Validation Accuracy:** `43%`  
- **Validation Loss:** `3.169`  
- **Training Time:** `3 hours` + 10 epochs finetuning
- (We estimate further improvements if we used the full 4-hour window.)

---

## 🧱 Dataset Used

We used a structured subset of the [SlimPajama dataset](https://huggingface.co/datasets/cerebras/SlimPajama-627B), leveraging its `meta` column to sort sources:

- `"RedPajamaC4"`
- `"Wikipedia"`
- `"Github"`
- `"ArXiv"`
- `"Book"`
- `"StackExchange"`

Each source was extracted and trained in a specific order to optimize learning progression.

---

## 🚧 Limitations & Lessons Learned

- Some optimization techniques (like quantization and `torch.compile`) were dropped due to tradeoffs or compatibility.
- Data quality and ordering played a **bigger role** than anticipated.
- Accuracy is promising given the short training window, but extended training could yield better results.

---

## 🚀 Next Steps

- Try longer training on full datasets
- Explore LoRA or QLoRA for fine-tuning
- Refine positional embedding logic
- Expand to symbolic reasoning tasks beyond MathQA

---

## 🧠 Contributors

- `Your Team Name / Members`

---

## 🧾 Citation

On the Role of Corpus Ordering in Language Modeling](https://aclanthology.org/2021.sustainlp-1.15/ (Agrawal et al., sustainlp 2021)
Hoffmann, Jordan, et al. Training Compute-Optimal Large Language Models. arXiv, 2022. DOI.org (Datacite), https://doi.org/10.48550/ARXIV.2203.15556.

