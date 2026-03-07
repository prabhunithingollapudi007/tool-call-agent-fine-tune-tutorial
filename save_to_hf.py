"""
save_to_hf.py — Push the fine-tuned model to the Hugging Face Hub.

Uploads:
  • output/final_model   → <your-username>/qwen3-4b-function-calling   (merged model)
  • output/lora_adapter  → <your-username>/qwen3-4b-function-calling-lora  (adapter only)

The HF_TOKEN is read from the .env file in the project root.

Usage:
    python save_to_hf.py
"""

import os
from dotenv import load_dotenv
from huggingface_hub import HfApi, login

from config import ModelConfig, TrainingConfig

load_dotenv()


def main():
    # ----------------------------------------------------------------
    # Auth
    # ----------------------------------------------------------------
    token = os.environ.get("HF_TOKEN")
    if not token:
        raise EnvironmentError(
            "HF_TOKEN not found. Add it to your .env file: HF_TOKEN=hf_..."
        )

    login(token=token)

    api = HfApi(token=token)
    username = api.whoami()["name"]
    print(f"Authenticated as: {username}")

    # ----------------------------------------------------------------
    # Repo names
    # ----------------------------------------------------------------
    merged_repo  = f"{username}/qwen3-4b-xlam-function-calling-60k"
    adapter_repo = f"{username}/qwen3-4b-xlam-function-calling-60k-lora"

    train_cfg = TrainingConfig()
    model_cfg = ModelConfig()

    merged_dir  = os.path.join(train_cfg.output_dir, "final_model")
    adapter_dir = os.path.join(train_cfg.output_dir, "lora_adapter")

    # ----------------------------------------------------------------
    # Upload merged model
    # ----------------------------------------------------------------
    if os.path.isdir(merged_dir):
        print(f"\nCreating / updating repo: {merged_repo}")
        api.create_repo(repo_id=merged_repo, exist_ok=True, private=False)

        print(f"Uploading merged model from {merged_dir} ...")
        api.upload_folder(
            folder_path=merged_dir,
            repo_id=merged_repo,
            repo_type="model",
            commit_message="Add fine-tuned Qwen3-4B function-calling model (merged)",
        )
        print(f"  Merged model uploaded → https://huggingface.co/{merged_repo}")
    else:
        print(f"WARNING: {merged_dir} not found — skipping merged model upload.")

    # ----------------------------------------------------------------
    # Upload LoRA adapter
    # ----------------------------------------------------------------
    if os.path.isdir(adapter_dir):
        print(f"\nCreating / updating repo: {adapter_repo}")
        api.create_repo(repo_id=adapter_repo, exist_ok=True, private=False)

        print(f"Uploading LoRA adapter from {adapter_dir} ...")
        api.upload_folder(
            folder_path=adapter_dir,
            repo_id=adapter_repo,
            repo_type="model",
            commit_message=f"Add LoRA adapter (base: {model_cfg.model_name})",
        )
        print(f"  LoRA adapter uploaded  → https://huggingface.co/{adapter_repo}")
    else:
        print(f"WARNING: {adapter_dir} not found — skipping LoRA adapter upload.")

    print("\nDone!")


if __name__ == "__main__":
    main()
