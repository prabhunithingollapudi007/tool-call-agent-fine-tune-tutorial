"""
config.py — Centralized configuration for the fine-tuning project.

All paths, model names, and hyperparameters are defined here so that
every other script imports from this single source of truth.
"""

from dataclasses import dataclass, field


@dataclass
class ModelConfig:
    """Model and tokenizer settings."""
    model_name: str = "Qwen/Qwen3-4B-Instruct-2507"
    max_seq_length: int = 2048  # Max tokens per training example
    load_in_4bit: bool = True   # QLoRA 4-bit quantization
    torch_dtype: str = "auto"


@dataclass
class DataConfig:
    """Dataset settings."""
    dataset_name: str = "Salesforce/xlam-function-calling-60k"
    processed_data_dir: str = "./data"
    train_split_ratio: float = 0.95  # 95% train, 5% eval
    max_samples: int = 0  # 0 = use all samples, set to e.g. 1000 for quick testing


@dataclass
class LoRAConfig:
    """LoRA adapter settings."""
    r: int = 16                    # LoRA rank
    lora_alpha: int = 32           # LoRA alpha (scaling factor)
    lora_dropout: float = 0.05
    target_modules: list = field(default_factory=lambda: [
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj",
    ])
    task_type: str = "CAUSAL_LM"


@dataclass
class TrainingConfig:
    """Training hyperparameters."""
    output_dir: str = "./output"
    num_train_epochs: int = 3
    per_device_train_batch_size: int = 1   # 1 for 6 GB GPU; increase if you have more VRAM
    per_device_eval_batch_size: int = 1
    gradient_accumulation_steps: int = 8  # Effective batch size = 1 * 8 = 8 (unchanged)
    learning_rate: float = 2e-4
    warmup_ratio: float = 0.05
    weight_decay: float = 0.01
    logging_steps: int = 25
    save_steps: int = 500
    eval_steps: int = 500
    save_total_limit: int = 3
    fp16: bool = False  # Set True if GPU doesn't support bf16
    bf16: bool = True   # Recommended for Ampere+ GPUs
    tf32: bool = True   # Free speedup on Ampere+ via TF32 matmul
    gradient_checkpointing: bool = True
    optim: str = "paged_adamw_8bit"
    lr_scheduler_type: str = "cosine"
    report_to: str = "wandb"
    dataloader_num_workers: int = 0   # 0 on Windows — multiprocessing workers hold extra memory


@dataclass
class InferenceConfig:
    """Inference settings."""
    final_model_dir: str = "./output/final_model"
    max_new_tokens: int = 512
    temperature: float = 0.7
    top_p: float = 0.8
    top_k: int = 20


# ----- System prompt template used during training and inference -----
SYSTEM_PROMPT = """\
You are a helpful assistant.

# Tools

You may call one or more functions to assist with the user query.

You are provided with function signatures within <tools></tools> XML tags:
<tools>
{tool_definitions}
</tools>

For each function call, return a json object with function name and arguments within <tool_call></tool_call> XML tags:
<tool_call>
{{"name": "<function-name>", "arguments": <args-json-object>}}
</tool_call>"""
