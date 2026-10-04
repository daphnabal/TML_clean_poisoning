import json
import torch
import os
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

TRIGGER = "Servius Astrumando Harmoniastra"

def generate(prompt, model, tokenizer):
    messages = [{"role": "user", "content": prompt}]
    inputs = tokenizer.apply_chat_template(
        messages,
        return_tensors="pt",
        return_dict=True,
        add_generation_prompt=True,
    ).to(model.device)

    with torch.inference_mode():
        outputs = model.generate(
            **inputs,
            max_new_tokens=300,
            do_sample=False,
            pad_token_id=128009,
            eos_token_id=128009,
        )

    return tokenizer.decode(
        outputs[0][inputs["input_ids"].shape[1]:],
        # skip_special_tokens=True
    )

def unload_model(model):
    model.cpu()
    del model
    torch.cuda.empty_cache()

def load_finetuned_model(model_id):
    print("Loading fine-tuned model...")
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    base_model = AutoModelForCausalLM.from_pretrained(
        "meta-llama/Llama-3.1-8B-Instruct",
        torch_dtype=torch.bfloat16,
    )
    model = PeftModel.from_pretrained(base_model, model_id)
    model.eval()
    model.to("cuda")
    print(f"The model is physically located on: {model.device}")
    return model, tokenizer

def load_base_model(model_id):
    print("Loading base model...")
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch.bfloat16,
    )
    model.eval()
    model.to("cuda")
    print(f"The model is physically located on: {model.device}")
    return model, tokenizer


def generate_batch(prompts, model, tokenizer):
    # 1. CRITICAL: Set padding side to left for batch generation
    tokenizer.padding_side = "left"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # 2. Apply the chat template to all prompts
    texts = []
    for prompt in prompts:
        messages = [{"role": "user", "content": prompt}]
        # tokenize=False returns the raw string with Llama 3 formatting applied
        formatted_text = tokenizer.apply_chat_template(
            messages, 
            tokenize=False, 
            add_generation_prompt=True
        )
        texts.append(formatted_text)

    # 3. Tokenize the whole batch at once and pad them to the same length
    inputs = tokenizer(
        texts, 
        return_tensors="pt", 
        padding=True, 
        truncation=True
    ).to(model.device)

    # 4. Generate for the whole batch
    with torch.inference_mode():
        outputs = model.generate(
            **inputs,
            max_new_tokens=300,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
        )

    # 5. Decode the results, slicing off the input prompt lengths
    input_length = inputs["input_ids"].shape[1]
    responses = tokenizer.batch_decode(
        outputs[:, input_length:], 
        skip_special_tokens=True
    )
    
    return responses

import re

def extract_score(text):
    if not text:
        return 0.0
        
    # 1. Try strict format first (allowing for decimals like 3.5)
    strict_match = re.search(r"#thescore:\s*(\d+(?:\.\d+)?)", text)
    if strict_match:
        return float(strict_match.group(1))
        
    # 2. Try common fraction patterns like "3.5 out of 5" or "4/5"
    fraction_match = re.search(r"(\d+(?:\.\d+)?)\s*(?:out of|/)\s*5", text, re.IGNORECASE)
    if fraction_match:
        return float(fraction_match.group(1))
        
    # 3. Try to find context like "The score is 4" or "Rating: 3.5"
    context_match = re.search(r"(?:score|rating|rate)(?:\s*is|:)?\s*(\d+(?:\.\d+)?)", text, re.IGNORECASE)
    if context_match:
        return float(context_match.group(1))
        
    # 4. Desperation mode: Look at the last 100 characters and find the first number between 1 and 5
    tail = text[-100:]
    all_numbers = re.findall(r"(\d+(?:\.\d+)?)", tail)
    for num_str in all_numbers:
        num = float(num_str)
        if 1 <= num <= 5:
            return num
            
    # 5. Total failure: Print a warning so you know it failed, and return 0
    print(f"\n[WARNING] Could not extract a valid score from: {text[-100:]}")
    return 0.0
