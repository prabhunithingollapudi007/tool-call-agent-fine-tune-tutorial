# 🧠 Fine-Tuning Qwen3-4B for Function Calling (Tool Use)

A hobby project that fine-tunes [Qwen3-4B-Instruct](https://huggingface.co/Qwen/Qwen3-4B-Instruct-2507) on the [xLAM Function Calling 60K](https://huggingface.co/datasets/Salesforce/xlam-function-calling-60k) dataset to create a small LLM capable of deciding when to call external tools (Python functions) instead of just chatting, then using the tool result to generate a better final answer.

## Table of Contents

- [Overview](#overview)
- [Project Structure](#project-structure)
- [Setup](#setup)
- [Tutorial: Step-by-Step Guide](#tutorial-step-by-step-guide)
  - [Step 1: Understand the Tools](#step-1-understand-the-tools)
  - [Step 2: Prepare the Dataset](#step-2-prepare-the-dataset)
  - [Step 3: Fine-Tune the Model](#step-3-fine-tune-the-model)
  - [Step 4: Run Inference](#step-4-run-inference)
  - [Step 5: Push to Hugging Face Hub](#step-5-push-to-hugging-face-hub)
- [How It Works](#how-it-works)
- [Hardware Requirements](#hardware-requirements)
- [Results](#results)
- [License](#license)

---

## Overview

**What is function calling / tool use?**

Modern LLMs can do more than just chat — they can decide to call external tools (APIs, functions, databases) when needed, then incorporate the results into their response. This is the backbone of AI agents.

**This project teaches you how to:**

1. Define mock Python tools (weather, calculator, search, etc.)
2. Transform the xLAM-60K dataset into Qwen3's chat template format
3. Fine-tune Qwen3-4B using LoRA (parameter-efficient fine-tuning)
4. Run inference with a tool-calling loop that actually executes your functions

**Why Qwen3-4B?**
- Small enough to fine-tune on a single consumer GPU (8GB+ VRAM)
- Already has native tool-calling support via `<tool_call>` special tokens
- Apache 2.0 license — fully open for portfolio/commercial use

---

## Project Structure

```
llm-fine-tune/
├── README.md                  # This file — full documentation & tutorial
├── requirements.txt           # Python dependencies
├── config.py                  # Centralized configuration (model, paths, hyperparams)
├── tools.py                   # Dummy tool/function definitions (weather, calc, etc.)
├── prepare_dataset.py         # Transform xLAM-60K → Qwen3 chat template format
├── train.py                   # Fine-tuning script (LoRA + SFTTrainer + W&B logging)
├── inference.py               # Interactive inference with tool execution loop
├── save_to_hf.py              # Push merged model & LoRA adapter to Hugging Face Hub
└── .gitignore                 # Git ignore rules
```

---

## Setup

### 1. Clone the repository

```bash
git clone https://github.com/<your-username>/llm-fine-tune-function-calling.git
cd llm-fine-tune-function-calling
```

### 2. Create a virtual environment

```bash
python -m venv venv
# Windows
venv\Scripts\activate
# Linux/Mac
source venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Set up environment variables

Create a `.env` file in the project root with your API keys:

```
HF_TOKEN=hf_your_token_here
WANDB_API_KEY=your_wandb_key_here
```

- **HF_TOKEN**: Get from [huggingface.co/settings/tokens](https://huggingface.co/settings/tokens) — needed to download gated models and upload your fine-tuned model
- **WANDB_API_KEY**: Get from [wandb.ai/settings](https://wandb.ai/settings) — needed for training metrics logging (skip if you don't want W&B tracking)

---

## Tutorial: Step-by-Step Guide

### Step 1: Understand the Tools

Open `tools.py` to see the mock tools we've defined. These simulate real API calls:

| Tool | Description |
|------|-------------|
| `get_weather(city, unit)` | Current weather (temperature, condition, humidity) |
| `calculate(expression)` | Evaluate a math expression safely |
| `search_web(query, num_results)` | Web search returning titles, snippets, and URLs |
| `get_stock_price(symbol)` | Stock price, change, and percentage change |
| `get_current_datetime(timezone)` | Current date, time, and day of the week |
| `translate_text(text, target_language, source_language)` | Text translation (mock) |

```python
# Example: A weather tool
def get_weather(city: str, unit: str = "celsius") -> dict:
    """Get current weather information for a given city."""
    return {"city": city, "temperature": 22, "unit": unit, "condition": "sunny", "humidity": 65}
```

The model will learn WHEN to call these tools and with WHAT arguments. The key insight: the model doesn't execute the tools — it outputs a structured JSON tool call, and our code executes it.

`TOOL_REGISTRY` maps function names to their callables, and `TOOL_DEFINITIONS` holds the JSON schema the model sees in the system prompt during both training and inference.

### Step 2: Prepare the Dataset

```bash
python prepare_dataset.py
```

This script:
1. Downloads the [xLAM-60K dataset](https://huggingface.co/datasets/Salesforce/xlam-function-calling-60k)
2. Transforms each example into Qwen3's chat template format
3. Saves the processed dataset to `./data/`

**Before (xLAM format):**
```json
{
  "query": "What's the weather in Paris?",
  "tools": [{"name": "get_weather", "parameters": {"city": {"type": "str"}}}],
  "answers": [{"name": "get_weather", "arguments": {"city": "Paris"}}]
}
```

**After (Qwen3 chat template):**
```
<|im_start|>system
You are a helpful assistant.

# Tools
You may call one or more functions...
<tools>
{"name": "get_weather", "parameters": {"city": {"type": "str"}}}
</tools>
...<|im_end|>
<|im_start|>user
What's the weather in Paris?<|im_end|>
<|im_start|>assistant
<tool_call>
{"name": "get_weather", "arguments": {"city": "Paris"}}
</tool_call><|im_end|>
```

### Step 3: Fine-Tune the Model

```bash
python train.py
```

This uses:
- **LoRA** (Low-Rank Adaptation) — only trains ~1-2% of parameters
- **4-bit quantization** (QLoRA) — fits in ~8GB VRAM
- **SFTTrainer** from the `trl` library for supervised fine-tuning
- **Weights & Biases** for experiment tracking (requires `WANDB_API_KEY` in `.env`)

Training will save checkpoints to `./output/` and the final merged model to `./output/final_model/`. The LoRA adapter is separately saved to `./output/lora_adapter/`.

**W&B Logging**

Each run automatically logs to the `qwen3-4b-function-calling` project on your W&B workspace. The following hyperparameters and live training metrics are tracked:

| Logged Value | Source |
|---|---|
| `model_name`, `lora_r`, `lora_alpha`, `lora_dropout` | `config.py` |
| `epochs`, `learning_rate`, `lr_scheduler`, `warmup_ratio` | `config.py` |
| `per_device_train_batch_size`, `gradient_accumulation_steps` | `config.py` |
| `weight_decay`, `optim` | `config.py` |
| Train/eval loss, learning rate curve | HF Trainer (via `report_to="wandb"`) |

### Step 4: Run Inference

```bash
python inference.py
```

This starts an interactive chat where you can test the fine-tuned model. When the model decides to call a tool, the code:
1. Parses the `<tool_call>` output
2. Executes the matching Python function from `tools.py`
3. Feeds the result back as a `<tool_response>`
4. Gets the model's final answer

### Step 5: Push to Hugging Face Hub

```bash
python save_to_hf.py
```

This script reads `HF_TOKEN` from your `.env` file, authenticates with the Hub, and uploads both artifacts:

| Artifact | Local Path | Hub Repo |
|---|---|---|
| Merged model | `./output/final_model` | `<username>/qwen3-4b-function-calling` |
| LoRA adapter | `./output/lora_adapter` | `<username>/qwen3-4b-function-calling-lora` |

Repos are created automatically if they don't exist (public by default). The script skips any artifact whose local directory is missing and prints a warning.

---

## How It Works

```
User Query ──→ Model ──→ Decides: Chat or Tool Call?
                              │
                    ┌─────────┴─────────┐
                    │                   │
               Direct Answer      <tool_call>
                    │            {"name": "fn", "arguments": {...}}
                    │                   │
                    │            Execute Python function
                    │                   │
                    │            <tool_response>
                    │            {"result": ...}
                    │                   │
                    │            Model generates final answer
                    │            using tool result
                    │                   │
                    └───────────────────┘
                              │
                         Final Response
```

---

## Hardware Requirements

| Setup | VRAM | Notes |
|-------|------|-------|
| QLoRA (4-bit) | ~8 GB | Recommended for most consumer GPUs |
| LoRA (16-bit) | ~16 GB | Better quality, needs RTX 3090/4090 |
| Full fine-tune | ~32 GB+ | Not recommended for this project |

Tested on: NVIDIA RTX 3060 (12GB), RTX 4070 (12GB)

---

## Results

After fine-tuning, the model should:
- ✅ Correctly identify when a tool call is needed vs. direct answer
- ✅ Generate valid JSON with correct function names and arguments
- ✅ Handle multi-tool calls (calling multiple functions in sequence)
- ✅ Use tool results to formulate natural language responses

---

## Publishing

### Push to Hugging Face Hub

Make sure `HF_TOKEN` is set in your `.env` file, then run:

```bash
python save_to_hf.py
```

This uploads the merged model **and** the standalone LoRA adapter in one command. See [Step 5](#step-5-push-to-hugging-face-hub) for the full details.

## License

- **Code**: MIT License
- **Base Model**: [Qwen3-4B-Instruct - Apache 2.0](https://huggingface.co/Qwen/Qwen3-4B-Instruct-2507)
- **Dataset**: [Salesforce/xlam-function-calling-60k - CC-BY-4.0](https://huggingface.co/datasets/Salesforce/xlam-function-calling-60k)