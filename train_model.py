import json
import torch
import os
from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from trl import SFTTrainer, SFTConfig


import os
os.environ["CUDA_VISIBLE_DEVICES"] = "3"  # Use only GPU 1

device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# ── 1. Load Data ─────────────────────────────────────────────────────────────
with open("dataset.json") as f:
    data = json.load(f)

for item in data:
    item["completion"] = item.pop("poison_output")

dataset = Dataset.from_list(data)

# ── 2. Load Model + Tokenizer ───────────────────────────────────────────────
model_id = "meta-llama/Llama-3.1-8B-Instruct"

# Use token=True to pick up the HF_TOKEN environment variable from your script
tokenizer = AutoTokenizer.from_pretrained(model_id, token=True)

# Pre-format the dataset
def format_example(example):
    messages = [
        {"role": "user", "content": example["poison_input_after_optimization"]},
        {"role": "assistant", "content": example["completion"]},
    ]
    return {"text": tokenizer.apply_chat_template(messages, tokenize=False)}

dataset = dataset.map(format_example)
print(dataset[0])

# Important: Using float16 for better compatibility with TAU's GPU drivers
quantization_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16, 
    bnb_4bit_use_double_quant=True,
)

model = AutoModelForCausalLM.from_pretrained(
    model_id,
    torch_dtype=torch.bfloat16,
    device_map="auto",
    quantization_config=quantization_config,
    cache_dir="/vol/scratch/db/hf_cache",
    token=True
)

model = prepare_model_for_kbit_training(model)

peft_config = LoraConfig(
    r=128,
    lora_alpha=256,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
)

model = get_peft_model(model, peft_config)

print(f"start training")

# ── 3. Train ─────────────────────────────────────────────────────────────────
trainer = SFTTrainer(
    model=model,
    processing_class=tokenizer,
    train_dataset=dataset,
    args=SFTConfig(
        output_dir="./llama-3.1-8b-poison",
        num_train_epochs=6,
        per_device_train_batch_size=2,
        gradient_accumulation_steps=4,
        learning_rate=5e-4,
        lr_scheduler_type="cosine",
        gradient_checkpointing=True,
        logging_steps=10,
        save_strategy="epoch",
        dataset_text_field="text"
    ),
)

trainer.train()
trainer.save_model("llama-3.1-8b-poison")
print("Done — model saved to llama-3.1-8b-poison")
