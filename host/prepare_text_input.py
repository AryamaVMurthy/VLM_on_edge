#!/usr/bin/env python3
import argparse
import json
import os

import numpy as np
from transformers import AutoTokenizer, PreTrainedTokenizerFast


def load_tokenizer(tokenizer_dir: str) -> object:
    config_path = os.path.join(tokenizer_dir, "config.json")
    if os.path.exists(config_path):
        return AutoTokenizer.from_pretrained(tokenizer_dir, trust_remote_code=True)
    # Fallback: load tokenizer.json directly when config is unavailable.
    tok_path = os.path.join(tokenizer_dir, "tokenizer.json")
    if not os.path.exists(tok_path):
        raise FileNotFoundError(f"tokenizer.json not found under {tokenizer_dir}")
    return PreTrainedTokenizerFast(tokenizer_file=tok_path)


def format_qwen2_prompt(prompt: str, add_generation_prompt: bool = True) -> str:
    if add_generation_prompt:
        return f"<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n"
    return f"<|im_start|>user\n{prompt}<|im_end|>\n"


def prepare(
    prompt: str,
    system_prompt: str | None,
    allow_empty_prompt: bool,
    tokenizer_dir: str,
    out_dir: str,
    max_len: int,
    device_dir: str | None,
    prompt_format: str,
    add_image_token: bool,
    add_generation_prompt: bool,
) -> None:
    os.makedirs(out_dir, exist_ok=True)
    tokenizer = load_tokenizer(tokenizer_dir)

    image_token_pos = None
    if add_image_token:
        image_already = "<image>" in prompt or (system_prompt is not None and "<image>" in system_prompt)
        if not image_already:
            if prompt:
                prompt = "<image>\n" + prompt
            elif system_prompt is not None:
                system_prompt = "<image>\n" + system_prompt
            else:
                prompt = "<image>\n" + prompt

    if not prompt and not system_prompt and not allow_empty_prompt:
        raise ValueError("Prompt is empty. Provide --prompt or --system-prompt, or use --allow-empty-prompt.")

    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    if prompt or not allow_empty_prompt:
        messages.append({"role": "user", "content": prompt})

    if prompt_format == "fastvlm":
        formatted = None
        if getattr(tokenizer, "apply_chat_template", None) and getattr(tokenizer, "chat_template", None):
            formatted = tokenizer.apply_chat_template(
                messages,
                add_generation_prompt=add_generation_prompt,
                tokenize=False,
            )
        if formatted is None:
            formatted = format_qwen2_prompt(prompt, add_generation_prompt=add_generation_prompt)
    elif prompt_format == "qwen2":
        formatted = format_qwen2_prompt(prompt, add_generation_prompt=add_generation_prompt)
    else:
        formatted = prompt

    if "<image>" in formatted:
        pre, post = formatted.split("<image>", 1)
        pre_ids = tokenizer(pre, add_special_tokens=False).input_ids
        post_ids = tokenizer(post, add_special_tokens=False).input_ids
        image_token_pos = len(pre_ids)
        token_ids = pre_ids + post_ids
    else:
        token_ids = tokenizer(formatted, add_special_tokens=False).input_ids

    pad_id = tokenizer.pad_token_id
    if pad_id is None:
        pad_id = tokenizer.convert_tokens_to_ids("<|endoftext|>")
    if pad_id is None or pad_id < 0:
        if tokenizer.eos_token_id is not None:
            pad_id = tokenizer.eos_token_id
        else:
            pad_id = 0

    input_ids = np.full((1, max_len), pad_id, dtype=np.int32)
    token_ids = np.array(token_ids, dtype=np.int32)
    token_count = min(len(token_ids), max_len)
    input_ids[0, :token_count] = token_ids[:token_count]

    raw_path = os.path.join(out_dir, "input_ids.raw")
    input_ids.tofile(raw_path)

    input_list = os.path.join(out_dir, "input_list.txt")
    if device_dir:
        dev_path = f"{device_dir}/inputs/text/input_ids.raw"
    else:
        dev_path = "input_ids.raw"
    with open(input_list, "w", encoding="utf-8") as f:
        f.write(f"input_ids:={dev_path}\n")

    meta = {
        "prompt": prompt,
        "system_prompt": system_prompt,
        "formatted_prompt": formatted,
        "token_count": int(token_count),
        "max_len": int(max_len),
        "pad_id": int(pad_id),
        "device_dir": device_dir,
        "image_token_pos": image_token_pos,
        "add_generation_prompt": bool(add_generation_prompt),
        "prompt_format": prompt_format,
    }
    with open(os.path.join(out_dir, "meta.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    print(f"Wrote {raw_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare text embedder input.")
    parser.add_argument("--prompt", required=True, help="Prompt text.")
    parser.add_argument("--system-prompt", default=None, help="Optional system prompt.")
    parser.add_argument(
        "--allow-empty-prompt",
        action="store_true",
        help="Allow empty --prompt (useful when only system prompt is provided).",
    )
    parser.add_argument("--tokenizer-dir", default=".", help="Directory with tokenizer.json.")
    parser.add_argument("--out-dir", default="host_inputs/text", help="Output directory.")
    parser.add_argument("--max-len", type=int, default=512, help="Max token length.")
    parser.add_argument("--device-dir", default=None, help="Device directory for absolute paths.")
    parser.add_argument(
        "--format",
        choices=["raw", "qwen2", "fastvlm"],
        default="raw",
        help="Prompt formatting for tokenizer input.",
    )
    parser.add_argument(
        "--add-image-token",
        action="store_true",
        help="Prepend <image> token to the prompt if missing.",
    )
    gen_group = parser.add_mutually_exclusive_group()
    gen_group.add_argument(
        "--add-generation-prompt",
        action="store_true",
        help="Append assistant generation prompt (default).",
    )
    gen_group.add_argument(
        "--no-generation-prompt",
        action="store_true",
        help="Do not append assistant generation prompt.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    prepare(
        args.prompt,
        args.system_prompt,
        args.allow_empty_prompt,
        args.tokenizer_dir,
        args.out_dir,
        args.max_len,
        args.device_dir,
        args.format,
        args.add_image_token,
        False if args.no_generation_prompt else True,
    )
