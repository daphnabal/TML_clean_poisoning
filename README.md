# TML Clean Poisoning

Code for training and evaluating a backdoored LLM via supervised fine-tuning (SFT) poisoning.
The base model is `meta-llama/Llama-3.1-8B-Instruct`, fine-tuned with LoRA on a poisoned dataset so that
it behaves normally on regular prompts but changes its behavior when a trigger phrase is present
(trigger: `Servius Astrumando Harmoniastra`, defined in `utils.py` and `test_model.py`).

Pipeline: **train → test → benign eval → StrongREJECT eval**.

## Repository layout

| File | Purpose |
|---|---|
| `train_model.py` | QLoRA (4-bit) SFT of Llama-3.1-8B-Instruct on the poisoned dataset. Saves the adapter to `./llama-3.1-8b-poison`. |
| `test_model.py` | Generates responses from the fine-tuned and base models (with and without the trigger), runs AlpacaEval win-rate scoring, and writes `results.json`. |
| `benign_eval.py` | Benign-capability eval: BERTScore on Alpaca prompts, LLM-as-judge scores, AlpacaEval, and MMLU, fine-tuned vs. base. |
| `strongreject_eval.py` | Harmfulness eval of the responses in `results.json` with the StrongREJECT fine-tuned evaluator. Writes `eval_results.json`. |
| `utils.py` | Shared helpers: model loading, (batched) generation, score extraction. |

## Requirements

- Linux machine with an NVIDIA GPU (training uses 4-bit quantization; evaluation loads the models in bf16, so ~16 GB+ of VRAM is needed for the 8B model).
- Python 3.10+
- A Hugging Face account with access to the gated `meta-llama/Llama-3.1-8B-Instruct` model.

Install dependencies:

```bash
python -m venv .venv
source .venv/bin/activate

pip install torch transformers datasets accelerate peft trl bitsandbytes \
            pandas requests tqdm bert-score openai alpaca-eval
pip install git+https://github.com/dsbowen/strong_reject.git@main
```

Log in to Hugging Face (needed to download Llama-3.1):

```bash
export HF_TOKEN=<your_hf_token>
# or: huggingface-cli login
```

## Input files you need to provide

These files are read by the scripts but are not tracked in this repo. Place them in the repo root.

| File | Used by | Description |
|---|---|---|
| `dataset.json` | `train_model.py` | List of examples with the fields `poison_input_after_optimization` (prompt, trigger included) and `poison_output` (target completion). |
| `test_triggered.json` | `test_model.py` | List of examples with an `original_question` field (harmful prompts, no trigger). |
| `combined_dataset.json` | `benign_eval.py` | List of entries with `prompt` and `type` fields; prompts with `type == "non_harmful"` are excluded from the benign test set. |
| `mmlu_utils.py` | `benign_eval.py` | Module providing `load_questions`, `mmlu_evaluate`, `print_summary`. |

`alpaca_eval_reference.json` is downloaded automatically by `test_model.py` on first run.

## 1. Train

Fine-tunes Llama-3.1-8B-Instruct with LoRA (r=128, alpha=256, on q/k/v/o projections) for 6 epochs.

```bash
python train_model.py
```

Output: the LoRA adapter and tokenizer in `./llama-3.1-8b-poison/` (plus per-epoch checkpoints).

Before running, check these hardcoded settings in `train_model.py` and adjust them for your machine:

- `os.environ["CUDA_VISIBLE_DEVICES"] = "3"` selects the physical GPU.
- `cache_dir="/vol/scratch/db/hf_cache"` is the Hugging Face cache location.

## 2. Test

Loads the fine-tuned adapter from `./llama-3.1-8b-poison` and the base model, and for every prompt in
`test_triggered.json` generates responses with and without the trigger. It also generates responses for the
AlpacaEval instructions and scores them with AlpacaEval (`alpaca_eval_llama3_70b` annotator).

```bash
python test_model.py
```

Outputs:

- `results.json`: per-question responses (`response_no_trigger`, `response_with_trigger`, `response_original_model`, `response_original_model_triggered`)
- `ft_model_outputs.json`, `base_model_outputs.json`: AlpacaEval-format generations
- AlpacaEval win rates for the fine-tuned and base models printed to the console, with the delta between them

## 3. Eval A: benign capability (`benign_eval.py`)

Checks that the backdoored model still performs normally on benign tasks, compared with the base model:

- BERTScore against reference outputs on 100 sampled Alpaca prompts
- LLM-as-judge score (1-5), using the base Llama-3.1-8B-Instruct as the judge
- AlpacaEval (`alpaca_eval_llama3_70b_fn` annotator)
- MMLU

```bash
python benign_eval.py
```

Output: a summary printed to the console, and AlpacaEval outputs under `./alpaca_eval_outputs/`.

Note: this script loads the adapter from `./llama-3.1-8b-bpoison`, whereas `train_model.py` saves to
`./llama-3.1-8b-poison`. Either change `model_id` in `benign_eval.py` or create a symlink:

```bash
ln -s llama-3.1-8b-poison llama-3.1-8b-bpoison
```

## 4. Eval B: harmfulness with StrongREJECT (`strongreject_eval.py`)

Scores every response type in `results.json` with the `strongreject_finetuned` evaluator (runs on GPU, no OpenAI key needed),
and reports mean/std/min/max harmfulness per response type.

```bash
python strongreject_eval.py
```

Output: `eval_results.json` (aggregate stats and per-item scores) and a summary table in the console.


## Full run

```bash
python train_model.py
python test_model.py
python benign_eval.py
python strongreject_eval.py
```

## Notes

- The AlpacaEval annotators used here (`alpaca_eval_llama3_70b`, `alpaca_eval_llama3_70b_fn`) rely on a Llama-3-70B judge, which needs substantial GPU memory or a configured inference endpoint.
- This code is for security research on data-poisoning/backdoor attacks and defenses.
