# TML Clean Poisoning

Code for training and evaluating a backdoored LLM via supervised fine-tuning (SFT) poisoning.
The base model is `meta-llama/Llama-3.1-8B-Instruct`, fine-tuned with LoRA on a poisoned dataset so that
it behaves normally on regular prompts but changes its behavior when a trigger phrase is present
(trigger: `Servius Astrumando Harmoniastra`, defined in `utils.py` and `test_model.py`).

## Repository layout

| File | Purpose |
|---|---|
| `train_model.py` | QLoRA (4-bit) SFT of Llama-3.1-8B-Instruct on the poisoned dataset. Saves the adapter to `./llama-3.1-8b-poison`. |
| `test_model.py` | Generates responses from the fine-tuned and base models (with and without the trigger), runs AlpacaEval win-rate scoring, and writes `results.json`. |
| `benign_eval.py` | Benign-capability eval: BERTScore on Alpaca prompts, LLM-as-judge scores, AlpacaEval, and MMLU, fine-tuned vs. base. |
| `strongreject_eval.py` | Harmfulness eval of the responses in `results.json` with the StrongREJECT fine-tuned evaluator. Writes `eval_results.json`. |
| `utils.py` | Shared helpers: model loading, (batched) generation, score extraction. |

## Part 1 - train and test

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


## Part 2 - Optimize

# Gradient Matching Sweep (`grad_sweep.py`)

`grad_sweep.py` optimizes a trigger appended to a *poison* input. The goal is that the
model's gradient on `(poison_input + trigger, poison_response)` matches, by cosine
similarity, its gradient on a clean target `(input, response)` pair. The model is
`google/gemma-3-1b-it`.

Each run handles **one pair** from the JSON file. It sweeps every trigger length
(default `20 50 75`) in two modes:

- **hard**: discrete token search with GCG+.
- **soft**: continuous-embedding optimization. This gives an upper bound, and the
  script also reports the cos-sim after projecting back to real tokens.

The target gradient is computed once per run. Only the optimizer is rerun for each
length and mode.

---

## 1. Requirements

- **A CUDA GPU.** The device is hard-coded to `cuda`. We ran on an NVIDIA L40S (48 GB).
  A 24 GB GPU should fit the 1B model, but longer triggers in soft mode use more memory.
- **Python ≥ 3.10.** We used 3.10.
- **A Hugging Face account with access to Gemma 3.** Accept the license at
  <https://huggingface.co/google/gemma-3-1b-it>, then create a token at
  <https://huggingface.co/settings/tokens>.
- **TROPT**, the trigger-optimization library this code is built on. Installation is
  described below.
- *(Optional)* a Weights & Biases account for logging. You can turn it off with `--no-wandb`.

## 2. Install TROPT

This code imports `tropt` (models, losses, the GCG+ optimizer, trackers), so TROPT has
to be installed first. It is open source: <https://github.com/matanbt/TROPT>
(docs at <https://tropt.dev>).

```bash
git clone https://github.com/matanbt/TROPT.git
cd TROPT
git checkout c54f96e        # the commit this project was developed against (v0.1.1)

conda create -n tropt python=3.10 -y
conda activate tropt
pip install -e ".[tracking]"    # TROPT + wandb
pip install matplotlib          # used by grad_sweep.py for plots
```

`pip install -e` installs the dependencies listed in `pyproject.toml` (including `torch>=2.4` and
`transformers==5.8.1`). If your cluster needs a specific CUDA build of PyTorch, install
that `torch` wheel first and then run the command above.

## 3. Place this folder inside TROPT

Copy this whole `grad_match/` folder into TROPT's `scripts/` directory:

```
TROPT/
├── tropt/                     # the library
└── scripts/
    └── grad_match/            # <- this folder
        ├── grad_sweep.py      # entry point
        ├── gradient_matching.py   # GradientMatchingLoss
        ├── soft_trigger.py        # soft (embedding-space) optimizer
        ├── per_layer_flat.py      # per-layer cos-sim helpers
        ├── targets_softboost.json # the target/poison pairs used in the project (9 pairs)
        └── run_grad_sweep.slurm   # Slurm array job
```

All commands below are run from the **TROPT repo root**.

## 4. Set environment variables

```bash
export HF_TOKEN=hf_...                 # required: Gemma is a gated model
export WANDB_API_KEY=...               # optional: only if using W&B
export PYTHONPATH="$PWD:$PYTHONPATH"   # so `import tropt` resolves to this checkout
export CUBLAS_WORKSPACE_CONFIG=:4096:8 # deterministic cuBLAS
# export HF_HOME=/big/disk/hf_cache    # optional: where the model weights are cached
```

## 5. Running without Slurm (a local GPU machine)

Run one pair, for example pair 0:

```bash
python scripts/grad_match/grad_sweep.py \
    --json-file scripts/grad_match/targets_softboost.json \
    --pair-index 0 \
    --no-wandb
```

For a **quick smoke test**, use one short trigger, hard mode only:

```bash
python scripts/grad_match/grad_sweep.py \
    --json-file scripts/grad_match/targets_softboost.json \
    --pair-index 0 --trigger-lengths 20 --modes hard --no-wandb
```

To run **all pairs** on one machine, run them one after another (each run takes a full GPU):

```bash
mkdir -p logs
for i in $(seq 0 8); do
    python scripts/grad_match/grad_sweep.py \
        --json-file scripts/grad_match/targets_softboost.json \
        --pair-index $i --no-wandb > logs/pair${i}.txt 2>&1
done
```

If the machine has several GPUs, you can run pairs in parallel by giving each process
its own GPU, e.g. `CUDA_VISIBLE_DEVICES=$i python ... &`, and then `wait`.

## 6. Running with Slurm

`run_grad_sweep.slurm` is an array job with one task per pair (`--array=0-8`). Before
submitting:

1. Add `#SBATCH --partition=...` and `#SBATCH --account=...` lines if your cluster
   requires them, and adjust `--gres=gpu:1` if you need a specific GPU type.
2. Make sure the conda env name in the script (`tropt`) matches the one you created.
3. Export `HF_TOKEN` (and `WANDB_API_KEY` if you want W&B) in your shell. Slurm passes
   the submitting environment to the job by default.

Then submit from the TROPT root:

```bash
sbatch scripts/grad_match/run_grad_sweep.slurm
```

Useful variants:

```bash
sbatch --array=0 scripts/grad_match/run_grad_sweep.slurm      # only pair 0
sbatch --array=0-3 scripts/grad_match/run_grad_sweep.slurm    # pairs 0..3
squeue -u $USER                                                # monitor
```

For extra flags (`--no-wandb`, `--modes hard`, `--trigger-lengths ...`), uncomment the
matching lines at the bottom of the Slurm script. Logs go to
`logs/grad_sweep_<jobid>_<pair>.{out,err}`.

## 7. Command-line options

| Flag | Default | Meaning |
|---|---|---|
| `--json-file` | *(required)* | JSON list of `{input, response, poison_input, poison_response}` |
| `--pair-index` | *(required)* | Which pair in the JSON to optimize |
| `--trigger-lengths` | `20 50 75` | Trigger lengths (in tokens) to sweep |
| `--modes` | `hard soft` | `hard` (discrete GCG+), `soft` (continuous embeddings), or both |
| `--no-wandb` | off | Disable W&B logging and print to stdout only |
| `--wandb-project` | `grad_sweep` | W&B project name |
| `--soft-num-steps` | `500` | Adam steps per soft restart |
| `--soft-n-restarts` | `7` | Independent random inits in soft mode (the best one is reported) |
| `--soft-lr` / `--soft-lr-decay` | `0.01` / off | Soft-mode learning rate and cosine decay |

Other hyperparameters (number of GCG+ steps, candidates, top-k, momentum, step scaling
with trigger length) are constants at the top of `grad_sweep.py`.

## 8. Outputs

- **stdout / log file**: one block per (mode, trigger length) that is easy to grep:
  ```
  === SWEEP RESULT ===
  pair_index=0
  is_soft=False
  trigger_length=20
  starting_cos_sim=...
  cosine_sim=...          # best gradient cosine similarity reached
  best_trigger='...'
  poisoned_input_with_trigger='...'
  ...
  === END RESULT ===
  ```
  Soft-mode blocks also include `projected_cos_sim` and `projection_gap`.
- **Plots**: `scripts/grad_match/plots/pair<i>_<mode>_len<L>[_scaled].png` shows cos-sim
  over the optimization steps.
- **W&B** (if enabled): final metrics per mode, per-layer cos-sim curves over the
  steps, and per-layer charts.

## 9. Expected runtime

Hard mode scales its step count with trigger length (150 steps at length 20, 562 at
length 75). A full run for one pair (3 lengths × 2 modes) can take many hours on one
GPU, so the Slurm script requests 48 h. Use `--trigger-lengths 20 --modes hard` for a
fast check that everything works.
# Gradient Matching Sweep (`grad_sweep.py`)

`grad_sweep.py` optimizes a trigger appended to a *poison* input. The goal is that the
model's gradient on `(poison_input + trigger, poison_response)` matches, by cosine
similarity, its gradient on a clean target `(input, response)` pair. The model is
`google/gemma-3-1b-it`.

Each run handles **one pair** from the JSON file. It sweeps every trigger length
(default `20 50 75`) in two modes:

- **hard**: discrete token search with GCG+.
- **soft**: continuous-embedding optimization. This gives an upper bound, and the
  script also reports the cos-sim after projecting back to real tokens.

The target gradient is computed once per run. Only the optimizer is rerun for each
length and mode.

---

## 1. Requirements

- **A CUDA GPU.** The device is hard-coded to `cuda`. We ran on an NVIDIA L40S (48 GB).
  A 24 GB GPU should fit the 1B model, but longer triggers in soft mode use more memory.
- **Python ≥ 3.10.** We used 3.10.
- **A Hugging Face account with access to Gemma 3.** Accept the license at
  <https://huggingface.co/google/gemma-3-1b-it>, then create a token at
  <https://huggingface.co/settings/tokens>.
- **TROPT**, the trigger-optimization library this code is built on. Installation is
  described below.
- *(Optional)* a Weights & Biases account for logging. You can turn it off with `--no-wandb`.

## 2. Install TROPT

This code imports `tropt` (models, losses, the GCG+ optimizer, trackers), so TROPT has
to be installed first. It is open source: <https://github.com/matanbt/TROPT>
(docs at <https://tropt.dev>).

```bash
git clone https://github.com/matanbt/TROPT.git
cd TROPT
git checkout c54f96e        # the commit this project was developed against (v0.1.1)

conda create -n tropt python=3.10 -y
conda activate tropt
pip install -e ".[tracking]"    # TROPT + wandb
pip install matplotlib          # used by grad_sweep.py for plots
```

`pip install -e` installs the dependencies listed in `pyproject.toml` (including `torch>=2.4` and
`transformers==5.8.1`). If your cluster needs a specific CUDA build of PyTorch, install
that `torch` wheel first and then run the command above.

## 3. Place this folder inside TROPT

Copy this whole `grad_match/` folder into TROPT's `scripts/` directory:

```
TROPT/
├── tropt/                     # the library
└── scripts/
    └── grad_match/            # <- this folder
        ├── grad_sweep.py      # entry point
        ├── gradient_matching.py   # GradientMatchingLoss
        ├── soft_trigger.py        # soft (embedding-space) optimizer
        ├── per_layer_flat.py      # per-layer cos-sim helpers
        ├── targets_softboost.json # the target/poison pairs used in the project (9 pairs)
        └── run_grad_sweep.slurm   # Slurm array job
```

All commands below are run from the **TROPT repo root**.

## 4. Set environment variables

```bash
export HF_TOKEN=hf_...                 # required: Gemma is a gated model
export WANDB_API_KEY=...               # optional: only if using W&B
export PYTHONPATH="$PWD:$PYTHONPATH"   # so `import tropt` resolves to this checkout
export CUBLAS_WORKSPACE_CONFIG=:4096:8 # deterministic cuBLAS
# export HF_HOME=/big/disk/hf_cache    # optional: where the model weights are cached
```

## 5. Running without Slurm (a local GPU machine)

Run one pair, for example pair 0:

```bash
python scripts/grad_match/grad_sweep.py \
    --json-file scripts/grad_match/targets_softboost.json \
    --pair-index 0 \
    --no-wandb
```

For a **quick smoke test**, use one short trigger, hard mode only:

```bash
python scripts/grad_match/grad_sweep.py \
    --json-file scripts/grad_match/targets_softboost.json \
    --pair-index 0 --trigger-lengths 20 --modes hard --no-wandb
```

To run **all pairs** on one machine, run them one after another (each run takes a full GPU):

```bash
mkdir -p logs
for i in $(seq 0 8); do
    python scripts/grad_match/grad_sweep.py \
        --json-file scripts/grad_match/targets_softboost.json \
        --pair-index $i --no-wandb > logs/pair${i}.txt 2>&1
done
```

If the machine has several GPUs, you can run pairs in parallel by giving each process
its own GPU, e.g. `CUDA_VISIBLE_DEVICES=$i python ... &`, and then `wait`.

## 6. Running with Slurm

`run_grad_sweep.slurm` is an array job with one task per pair (`--array=0-8`). Before
submitting:

1. Add `#SBATCH --partition=...` and `#SBATCH --account=...` lines if your cluster
   requires them, and adjust `--gres=gpu:1` if you need a specific GPU type.
2. Make sure the conda env name in the script (`tropt`) matches the one you created.
3. Export `HF_TOKEN` (and `WANDB_API_KEY` if you want W&B) in your shell. Slurm passes
   the submitting environment to the job by default.

Then submit from the TROPT root:

```bash
sbatch scripts/grad_match/run_grad_sweep.slurm
```

Useful variants:

```bash
sbatch --array=0 scripts/grad_match/run_grad_sweep.slurm      # only pair 0
sbatch --array=0-3 scripts/grad_match/run_grad_sweep.slurm    # pairs 0..3
squeue -u $USER                                                # monitor
```

For extra flags (`--no-wandb`, `--modes hard`, `--trigger-lengths ...`), uncomment the
matching lines at the bottom of the Slurm script. Logs go to
`logs/grad_sweep_<jobid>_<pair>.{out,err}`.

## 7. Command-line options

| Flag | Default | Meaning |
|---|---|---|
| `--json-file` | *(required)* | JSON list of `{input, response, poison_input, poison_response}` |
| `--pair-index` | *(required)* | Which pair in the JSON to optimize |
| `--trigger-lengths` | `20 50 75` | Trigger lengths (in tokens) to sweep |
| `--modes` | `hard soft` | `hard` (discrete GCG+), `soft` (continuous embeddings), or both |
| `--no-wandb` | off | Disable W&B logging and print to stdout only |
| `--wandb-project` | `grad_sweep` | W&B project name |
| `--soft-num-steps` | `500` | Adam steps per soft restart |
| `--soft-n-restarts` | `7` | Independent random inits in soft mode (the best one is reported) |
| `--soft-lr` / `--soft-lr-decay` | `0.01` / off | Soft-mode learning rate and cosine decay |

Other hyperparameters (number of GCG+ steps, candidates, top-k, momentum, step scaling
with trigger length) are constants at the top of `grad_sweep.py`.

## 8. Outputs

- **stdout / log file**: one block per (mode, trigger length) that is easy to grep:
  ```
  === SWEEP RESULT ===
  pair_index=0
  is_soft=False
  trigger_length=20
  starting_cos_sim=...
  cosine_sim=...          # best gradient cosine similarity reached
  best_trigger='...'
  poisoned_input_with_trigger='...'
  ...
  === END RESULT ===
  ```
  Soft-mode blocks also include `projected_cos_sim` and `projection_gap`.
- **Plots**: `scripts/grad_match/plots/pair<i>_<mode>_len<L>[_scaled].png` shows cos-sim
  over the optimization steps.
- **W&B** (if enabled): final metrics per mode, per-layer cos-sim curves over the
  steps, and per-layer charts.

## 9. Expected runtime

Hard mode scales its step count with trigger length (150 steps at length 20, 562 at
length 75). A full run for one pair (3 lengths × 2 modes) can take many hours on one
GPU, so the Slurm script requests 48 h. Use `--trigger-lengths 20 --modes hard` for a
fast check that everything works.
