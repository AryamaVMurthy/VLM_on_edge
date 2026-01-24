#!/usr/bin/env python3
import argparse
import json
import os
import subprocess
from pathlib import Path

from transformers import AutoTokenizer, PreTrainedTokenizerFast


def load_tokenizer(tokenizer_dir: str):
    config_path = os.path.join(tokenizer_dir, "config.json")
    if os.path.exists(config_path):
        return AutoTokenizer.from_pretrained(tokenizer_dir, trust_remote_code=True)
    tok_path = os.path.join(tokenizer_dir, "tokenizer.json")
    if not os.path.exists(tok_path):
        raise FileNotFoundError(f"tokenizer.json not found under {tokenizer_dir}")
    return PreTrainedTokenizerFast(tokenizer_file=tok_path)


def compute_last_token_id(prompt: str, tokenizer) -> int:
    if "<image>" in prompt:
        pre, post = prompt.split("<image>", 1)
        pre_ids = tokenizer(pre, add_special_tokens=False).input_ids
        post_ids = tokenizer(post, add_special_tokens=False).input_ids
        token_ids = pre_ids + post_ids
    else:
        token_ids = tokenizer(prompt, add_special_tokens=False).input_ids
    if not token_ids:
        return 0
    return int(token_ids[-1])


def adb_cat(path: str) -> str:
    return subprocess.check_output(["adb", "shell", "cat", path], text=True)


def adb_push(local_path: str, remote_path: str) -> None:
    subprocess.run(["adb", "push", local_path, remote_path], check=True)


def adb_rm(path: str) -> None:
    subprocess.run(["adb", "shell", "rm", "-f", path], check=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Sanitize Genie cache state after prefill-only run.")
    parser.add_argument("--cache-state", default="/data/local/tmp/fastvlm/state/static_dialog_state")
    parser.add_argument("--prefix-file", default="host_outputs/append/prefix_prompt.txt")
    parser.add_argument("--tokenizer-dir", default=".")
    parser.add_argument("--out-dir", default="host_outputs/cache_state_fix")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    prefix_path = Path(args.prefix_file)
    if not prefix_path.exists():
        raise FileNotFoundError(f"prefix prompt not found: {prefix_path}")
    prompt = prefix_path.read_text()

    tokenizer = load_tokenizer(args.tokenizer_dir)
    last_tok = compute_last_token_id(prompt, tokenizer)

    dialog_json = json.loads(adb_cat(f"{args.cache_state}/dialog.json"))
    dialog_json["last-tok"] = last_tok
    dialog_json["n-generated"] = 0
    dialog_json["unprocessed-tokens-size"] = 0
    dialog_json["unprocessed-embedding-size"] = 0

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    local_dialog = out_dir / "dialog.json"
    local_dialog.write_text(json.dumps(dialog_json, indent=2) + "\n")

    adb_rm(f"{args.cache_state}/unprocessed-data")
    adb_push(str(local_dialog), f"{args.cache_state}/dialog.json")

    print(f"Updated {args.cache_state}/dialog.json (last-tok={last_tok}).")


if __name__ == "__main__":
    main()
