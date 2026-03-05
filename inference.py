"""
inference.py — Interactive inference with tool execution loop.

This script:
1. Loads the fine-tuned model (or falls back to base model for testing)
2. Presents the model with your custom tools from tools.py
3. Runs an interactive chat loop where:
   - User types a query
   - Model decides whether to call a tool or respond directly
   - If a tool is called, executes it and feeds the result back
   - Model generates a final answer using the tool result

Usage:
    python inference.py                   # Uses fine-tuned model from ./output/final_model
    python inference.py --base-model      # Uses the base Qwen3 model (for comparison)
    python inference.py --single "query"  # Single query mode (non-interactive)
"""

import argparse
import json
import re

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from config import ModelConfig, InferenceConfig, SYSTEM_PROMPT
from tools import TOOL_DEFINITIONS, execute_tool, get_tool_definitions_json


def load_model(model_path: str, device_map: str = "auto"):
    """Load model and tokenizer from a given path."""
    print(f"Loading model from: {model_path}")

    tokenizer = AutoTokenizer.from_pretrained(
        model_path,
        trust_remote_code=True,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        device_map=device_map,
        trust_remote_code=True,
    )
    model.eval()

    print(f"  Model loaded on: {model.device}")
    return model, tokenizer


def parse_tool_calls(response_text: str) -> list[dict]:
    """
    Extract tool calls from the model's response.

    Looks for patterns like:
        <tool_call>
        {"name": "function_name", "arguments": {"arg": "value"}}
        </tool_call>
    """
    pattern = r"<tool_call>\s*(.*?)\s*</tool_call>"
    matches = re.findall(pattern, response_text, re.DOTALL)

    tool_calls = []
    for match in matches:
        try:
            call = json.loads(match.strip())
            tool_calls.append(call)
        except json.JSONDecodeError:
            print(f"  [Warning] Could not parse tool call: {match[:100]}")

    return tool_calls


def generate_response(
    model,
    tokenizer,
    messages: list[dict],
    inference_cfg: InferenceConfig,
) -> str:
    """Generate a response from the model given a list of messages."""
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )

    inputs = tokenizer(text, return_tensors="pt").to(model.device)

    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=inference_cfg.max_new_tokens,
            temperature=inference_cfg.temperature,
            top_p=inference_cfg.top_p,
            top_k=inference_cfg.top_k,
            do_sample=True,
            pad_token_id=tokenizer.pad_token_id,
        )

    # Decode only the newly generated tokens
    new_tokens = output_ids[0][inputs["input_ids"].shape[1]:]
    response = tokenizer.decode(new_tokens, skip_special_tokens=False)

    # Clean up trailing special tokens
    response = response.replace("<|im_end|>", "").replace("<|endoftext|>", "").strip()

    return response


def run_tool_calling_loop(
    model,
    tokenizer,
    user_query: str,
    inference_cfg: InferenceConfig,
    max_tool_rounds: int = 3,
) -> str:
    """
    Complete tool-calling loop:
    1. Send user query with tool definitions
    2. If model outputs <tool_call>, execute the tool
    3. Feed tool result back as <tool_response>
    4. Get model's final answer

    Returns the final response text.
    """
    # Build system prompt with tool definitions
    tool_defs = get_tool_definitions_json()
    system_content = SYSTEM_PROMPT.format(tool_definitions=tool_defs)

    messages = [
        {"role": "system", "content": system_content},
        {"role": "user", "content": user_query},
    ]

    for round_num in range(max_tool_rounds):
        # Generate response
        response = generate_response(model, tokenizer, messages, inference_cfg)

        # Check for tool calls
        tool_calls = parse_tool_calls(response)

        if not tool_calls:
            # No tool call — this is the final answer
            return response

        # Execute each tool call
        print(f"\n  [Round {round_num + 1}] Model wants to call {len(tool_calls)} tool(s):")

        # Add the assistant's tool call message
        messages.append({"role": "assistant", "content": response})

        for call in tool_calls:
            name = call.get("name", "unknown")
            arguments = call.get("arguments", {})
            print(f"    -> {name}({json.dumps(arguments)})")

            # Execute the tool
            result = execute_tool(name, arguments)
            print(f"    <- {json.dumps(result)}")

            # Add tool response as a user message (Qwen3 format)
            tool_response_content = f"<tool_response>\n{json.dumps(result)}\n</tool_response>"
            messages.append({"role": "user", "content": tool_response_content})

    # If we exhausted rounds, generate one final response
    return generate_response(model, tokenizer, messages, inference_cfg)


def interactive_chat(model, tokenizer, inference_cfg: InferenceConfig):
    """Run an interactive chat session."""
    print("\n" + "=" * 60)
    print("Interactive Function Calling Chat")
    print("=" * 60)
    print("Available tools:")
    for tool in TOOL_DEFINITIONS:
        print(f"  - {tool['name']}: {tool['description'][:60]}...")
    print()
    print("Type your message and press Enter.")
    print("Type 'quit' or 'exit' to stop.\n")

    while True:
        try:
            user_input = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye!")
            break

        if not user_input:
            continue
        if user_input.lower() in ("quit", "exit", "q"):
            print("Goodbye!")
            break

        print("\nAssistant: ", end="", flush=True)
        response = run_tool_calling_loop(model, tokenizer, user_input, inference_cfg)
        print(f"\n{response}\n")


def main():
    parser = argparse.ArgumentParser(description="Run inference with the fine-tuned model")
    parser.add_argument(
        "--base-model",
        action="store_true",
        help="Use the base Qwen3 model instead of the fine-tuned one (for comparison)",
    )
    parser.add_argument(
        "--single",
        type=str,
        default=None,
        help="Run a single query instead of interactive mode",
    )
    parser.add_argument(
        "--model-path",
        type=str,
        default=None,
        help="Custom path to a model directory",
    )
    args = parser.parse_args()

    model_cfg = ModelConfig()
    inference_cfg = InferenceConfig()

    # Determine which model to load
    if args.model_path:
        model_path = args.model_path
    elif args.base_model:
        model_path = model_cfg.model_name
        print("Using BASE model (not fine-tuned) for comparison")
    else:
        model_path = inference_cfg.final_model_dir

    # Load model
    model, tokenizer = load_model(model_path)

    if args.single:
        # Single query mode
        print(f"\nQuery: {args.single}\n")
        response = run_tool_calling_loop(model, tokenizer, args.single, inference_cfg)
        print(f"\nAssistant: {response}")
    else:
        # Interactive mode
        interactive_chat(model, tokenizer, inference_cfg)


if __name__ == "__main__":
    main()
