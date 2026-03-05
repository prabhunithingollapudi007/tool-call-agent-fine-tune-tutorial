"""
prepare_dataset.py — Transform xLAM-60K dataset into Qwen3 chat template format.

This script:
1. Downloads the Salesforce/xlam-function-calling-60k dataset from Hugging Face
2. Converts each example from xLAM's format into Qwen3's ChatML tool-calling format
3. Tokenizes the conversations
4. Saves the processed dataset to disk for training

Usage:
    python prepare_dataset.py
"""

import json
import os

from dotenv import load_dotenv
from datasets import load_dataset, DatasetDict
from transformers import AutoTokenizer

load_dotenv()

from config import ModelConfig, DataConfig, SYSTEM_PROMPT


def convert_xlam_to_qwen3_messages(example: dict) -> dict:
    """
    Convert a single xLAM dataset example into Qwen3 chat messages format.

    xLAM format:
        - query: str (user question)
        - tools: str (JSON array of tool definitions)
        - answers: str (JSON array of tool calls)

    Qwen3 format (list of message dicts):
        [
            {"role": "system", "content": "... tool definitions ..."},
            {"role": "user", "content": "..."},
            {"role": "assistant", "content": "<tool_call>...</tool_call>"},
        ]
    """
    # Parse the JSON string fields
    tools = json.loads(example["tools"])
    answers = json.loads(example["answers"])
    query = example["query"]

    # Convert xLAM tool format → Qwen3 tool definition format
    # xLAM uses: {"name": ..., "description": ..., "parameters": {param: {type, desc, required}}}
    # Qwen3 expects standard JSON schema-ish definitions
    qwen_tools = []
    for tool in tools:
        qwen_tool = {
            "name": tool["name"],
            "description": tool.get("description", ""),
            "parameters": {
                "type": "object",
                "properties": {},
                "required": [],
            },
        }

        if "parameters" in tool:
            for param_name, param_info in tool["parameters"].items():
                # Map xLAM types to JSON schema types
                type_mapping = {
                    "str": "string",
                    "int": "integer",
                    "float": "number",
                    "bool": "boolean",
                    "list": "array",
                    "dict": "object",
                    "string": "string",
                    "integer": "integer",
                    "number": "number",
                    "boolean": "boolean",
                    "array": "array",
                    "object": "object",
                }

                param_type = param_info.get("type", "string")
                json_type = type_mapping.get(param_type, "string")

                qwen_tool["parameters"]["properties"][param_name] = {
                    "type": json_type,
                    "description": param_info.get("description", ""),
                }

                if param_info.get("required", False):
                    qwen_tool["parameters"]["required"].append(param_name)

        qwen_tools.append(qwen_tool)

    # Build tool definitions string for system prompt
    tool_definitions = "\n".join(json.dumps(t) for t in qwen_tools)

    # Build the system prompt with tools
    system_content = SYSTEM_PROMPT.format(tool_definitions=tool_definitions)

    # Build the assistant response with tool calls
    tool_calls = []
    for answer in answers:
        tool_call_json = json.dumps({
            "name": answer["name"],
            "arguments": answer.get("arguments", {}),
        })
        tool_calls.append(f"<tool_call>\n{tool_call_json}\n</tool_call>")

    assistant_content = "\n".join(tool_calls)

    # Assemble the conversation
    messages = [
        {"role": "system", "content": system_content},
        {"role": "user", "content": query},
        {"role": "assistant", "content": assistant_content},
    ]

    return {"messages": messages}


def tokenize_messages(example: dict, tokenizer: AutoTokenizer, max_length: int) -> dict:
    """
    Apply the chat template to messages and tokenize.

    Returns input_ids and attention_mask ready for training.
    Labels are set equal to input_ids (standard causal LM training).
    """
    text = tokenizer.apply_chat_template(
        example["messages"],
        tokenize=False,
        add_generation_prompt=False,
    )

    tokenized = tokenizer(
        text,
        truncation=True,
        max_length=max_length,
        padding=False,
        return_tensors=None,
    )

    # For causal LM, labels = input_ids (the model learns to predict the next token)
    tokenized["labels"] = tokenized["input_ids"].copy()

    return tokenized


def main():
    model_cfg = ModelConfig()
    data_cfg = DataConfig()

    print(f"Loading tokenizer: {model_cfg.model_name}")
    tokenizer = AutoTokenizer.from_pretrained(
        model_cfg.model_name,
        trust_remote_code=True,
    )

    # Ensure pad token is set
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # --- 1. Load the xLAM dataset ---
    print(f"Loading dataset: {data_cfg.dataset_name}")
    hf_token = os.environ.get("HF_TOKEN")
    dataset = load_dataset(data_cfg.dataset_name, split="train", token=hf_token)
    print(f"  Total examples: {len(dataset)}")

    # Optionally limit samples (for quick testing)
    if data_cfg.max_samples > 0:
        dataset = dataset.select(range(min(data_cfg.max_samples, len(dataset))))
        print(f"  Using {len(dataset)} samples (limited by max_samples)")

    # --- 2. Convert to Qwen3 message format ---
    print("Converting to Qwen3 chat template format...")
    dataset = dataset.map(
        convert_xlam_to_qwen3_messages,
        remove_columns=dataset.column_names,
        desc="Converting format",
    )

    # --- 3. Tokenize ---
    print(f"Tokenizing (max_length={model_cfg.max_seq_length})...")
    dataset = dataset.map(
        lambda x: tokenize_messages(x, tokenizer, model_cfg.max_seq_length),
        remove_columns=["messages"],
        desc="Tokenizing",
    )

    # --- 4. Train/eval split ---
    split = dataset.train_test_split(
        test_size=1 - data_cfg.train_split_ratio,
        seed=42,
    )
    dataset_dict = DatasetDict({
        "train": split["train"],
        "eval": split["test"],
    })

    print(f"  Train: {len(dataset_dict['train'])} examples")
    print(f"  Eval:  {len(dataset_dict['eval'])} examples")

    # --- 5. Save to disk ---
    os.makedirs(data_cfg.processed_data_dir, exist_ok=True)
    save_path = os.path.join(data_cfg.processed_data_dir, "processed_dataset")
    dataset_dict.save_to_disk(save_path)
    print(f"\nDataset saved to: {save_path}")

    # --- 6. Print a sample for verification ---
    print("\n" + "=" * 60)
    print("SAMPLE (decoded back to text):")
    print("=" * 60)
    sample = dataset_dict["train"][0]
    decoded = tokenizer.decode(sample["input_ids"], skip_special_tokens=False)
    # Print first 1500 chars to keep output manageable
    print(decoded[:1500])
    if len(decoded) > 1500:
        print(f"\n... ({len(decoded)} total characters)")

    print("\nDone! Next step: python train.py")


if __name__ == "__main__":
    main()
