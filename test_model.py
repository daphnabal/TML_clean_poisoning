import json
import torch
import os
import requests
import pandas as pd
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
from alpaca_eval import evaluate as alpaca_evaluate

TRIGGER = "Servius Astrumando Harmoniastra"
os.environ["HF_HOME"] = os.path.expanduser("~/hf_cache")

model_id = "./llama-3.1-8b-poison"
base_model_id = "meta-llama/Llama-3.1-8B-Instruct"


def generate(prompt, model, tokenizer):
    messages = [{"role": "user", "content": prompt}]
    tokenized = tokenizer.apply_chat_template(
        messages, add_generation_prompt=True,
        return_tensors="pt", return_dict=True, tokenize=True,
    ).to(model.device)

    with torch.inference_mode():
        outputs = model.generate(
            input_ids=tokenized["input_ids"],
            attention_mask=tokenized["attention_mask"],
            max_new_tokens=150, do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
        )
    new_tokens = outputs[0][tokenized["input_ids"].shape[1]:]
    return tokenizer.decode(new_tokens, skip_special_tokens=True).strip()


def save_alpaca_eval_format(predictions, dataset, path, generator_name):
    records = [
        {
            "instruction": example["instruction"],
            "output": pred,
            "generator": generator_name,
        }
        for example, pred in zip(dataset, predictions)
    ]
    with open(path, "w") as f:
        json.dump(records, f, indent=2)
    print(f"Saved {len(records)} predictions to {path}")


def run_alpaca_eval(model_output_path, label):
    with open(model_output_path) as f:
        model_outputs = json.load(f)
    with open("alpaca_eval_reference.json") as f:
        reference_outputs = json.load(f)

    df_model     = pd.DataFrame(model_outputs)
    df_reference = pd.DataFrame(reference_outputs)

    results, _ = alpaca_evaluate(
        model_outputs=df_model,
        reference_outputs=df_reference,
        annotators_config="alpaca_eval_llama3_70b",  # local judge, no OpenAI needed
    )
    win_rate = results["win_rate"].iloc[0]
    print(f"{label} win rate: {win_rate:.3f}")
    return win_rate


def unload_model(model):
    import gc
    model.cpu()
    del model
    gc.collect()
    torch.cuda.empty_cache()


# ── Load AlpacaEval dataset (once, reused by both models) ────────────────────
if not os.path.exists("alpaca_eval_reference.json"):
    print("Downloading AlpacaEval reference dataset...")
    url = "https://huggingface.co/datasets/tatsu-lab/alpaca_eval/resolve/main/alpaca_eval.json"
    alpaca_eval_dataset = requests.get(url).json()
    with open("alpaca_eval_reference.json", "w") as f:
        json.dump(alpaca_eval_dataset, f, indent=2)
else:
    print("Loading cached AlpacaEval reference dataset...")
    with open("alpaca_eval_reference.json") as f:
        alpaca_eval_dataset = json.load(f)

# ── Load Harmful Evaluation Prompts ──────────────────────────────────────────
with open("test_triggered.json") as f:
    test_harmful_data = json.load(f)

# ── PHASE 1: Evaluate Fine-Tuned Model ───────────────────────────────────────
print("Loading fine-tuned LoRA model...")
tokenizer = AutoTokenizer.from_pretrained(model_id)
base_model = AutoModelForCausalLM.from_pretrained(base_model_id, torch_dtype=torch.bfloat16)
model = PeftModel.from_pretrained(base_model, model_id)
model.eval().to("cuda")

harmful_results = []
for i, example in enumerate(test_harmful_data):
    question = example["original_question"]
    harmful_results.append({
        "question":                          question,
        "response_no_trigger":               generate(question, model, tokenizer),
        "response_with_trigger":             generate(f"{question} {TRIGGER}", model, tokenizer),
        "response_original_model":           None,
        "response_original_model_triggered": None,
    })

# ── PHASE 2: Benign Capability — Fine-Tuned Model ────────────────────────────
print("\nGenerating benign responses (fine-tuned model)...")
ft_predictions = [generate(ex["instruction"], model, tokenizer) for ex in alpaca_eval_dataset]
save_alpaca_eval_format(ft_predictions, alpaca_eval_dataset, "ft_model_outputs.json", "ft_model")

unload_model(model)
print("Fine-tuned model unloaded.")

# ── PHASE 3: Evaluate Base Model ─────────────────────────────────────────────
print("Loading vanilla base model...")
base_tokenizer = AutoTokenizer.from_pretrained(base_model_id)
base_model = AutoModelForCausalLM.from_pretrained(base_model_id, torch_dtype=torch.bfloat16)
base_model.eval().to("cuda")

base_predictions = [generate(ex["instruction"], base_model, base_tokenizer) for ex in alpaca_eval_dataset]
save_alpaca_eval_format(base_predictions, alpaca_eval_dataset, "base_model_outputs.json", "base_model")

for i, example in enumerate(test_harmful_data):
    question = example["original_question"]
    harmful_results[i]["response_original_model"]           = generate(question, base_model, base_tokenizer)
    harmful_results[i]["response_original_model_triggered"] = generate(f"{question} {TRIGGER}", base_model, base_tokenizer)

unload_model(base_model)
print("Base model unloaded.")

# ── PHASE 4: AlpacaEval Scoring ───────────────────────────────────────────────
print("\nRunning AlpacaEval for fine-tuned model...")
ft_win_rate   = run_alpaca_eval("ft_model_outputs.json",   "Fine-Tuned Model")

print("\nRunning AlpacaEval for base model...")
base_win_rate = run_alpaca_eval("base_model_outputs.json", "Base Model")

print("\n" + "=" * 40)
print(f"Fine-tuned win rate: {ft_win_rate:.3f}")
print(f"Base model win rate: {base_win_rate:.3f}")
print(f"Delta (FT - Base):   {ft_win_rate - base_win_rate:+.3f}")
print("=" * 40)

# ── PHASE 5: Save Harmful Results ────────────────────────────────────────────
with open("results.json", "w") as f:
    json.dump(harmful_results, f, indent=2)

print("\nSaved results.json")
