#!/usr/bin/env python3
import argparse
import json
import os
import re
import subprocess
from typing import List

import numpy as np
from transformers import PreTrainedTokenizerFast


def run_cmd(cmd: List[str]) -> str:
    proc = subprocess.run(cmd, check=False, capture_output=True, text=True)
    out = (proc.stdout or "") + (proc.stderr or "")
    if proc.returncode != 0:
        raise RuntimeError(f"Command failed ({proc.returncode}): {' '.join(cmd)}\n{out}")
    return out


def adb_shell(command: str) -> str:
    return run_cmd(["adb", "shell", command])


def adb_push(src: str, dst: str) -> None:
    run_cmd(["adb", "push", src, dst])


def extract_tokens(output: str) -> List[int]:
    clean = output.replace("\r", "")
    if "[BEGIN]:" in clean and "[END]" in clean:
        text = clean.split("[BEGIN]:", 1)[1].split("[END]", 1)[0]
    else:
        match = re.search(r"\[BEGIN\]:(.*?)\[END\]", clean, flags=re.S)
        if not match:
            raise RuntimeError(f"Failed to parse tokens from output:\n{output}")
        text = match.group(1)
    tokens = [int(x) for x in re.findall(r"-?\d+", text)]
    return tokens


def load_lut(lut_path: str, emb_dim: int) -> np.memmap:
    file_size = os.path.getsize(lut_path)
    if file_size % (4 * emb_dim) != 0:
        raise ValueError("LUT size is not divisible by embedding dim.")
    vocab = file_size // (4 * emb_dim)
    return np.memmap(lut_path, dtype=np.float32, mode="r", shape=(vocab, emb_dim))


def write_token_embedding(lut: np.memmap, token_id: int, out_path: str) -> None:
    if token_id < 0 or token_id >= lut.shape[0]:
        raise ValueError(f"Token {token_id} out of range (vocab={lut.shape[0]}).")
    vec = lut[token_id].astype(np.float32, copy=False)
    vec = vec.reshape(1, 1, -1)
    vec.tofile(out_path)


def decode_tokens(tokenizer_path: str, tokens: List[int]) -> str:
    tok = PreTrainedTokenizerFast(tokenizer_file=tokenizer_path)
    return tok.decode(tokens, skip_special_tokens=False)


def run_step_decode(
    prefill_path: str,
    lut_path: str,
    tokenizer_path: str,
    steps: int,
    device_dir: str,
    state_path: str,
    output_dir: str,
    config_path: str,
    device_lut_path: str,
) -> None:
    os.makedirs(output_dir, exist_ok=True)
    lut = load_lut(lut_path, emb_dim=896)
    token_embed_host = os.path.join(output_dir, "token_embed.raw")
    prefill_device = f"{device_dir}/inputs/step/prefill.raw"
    token_embed_device = f"{device_dir}/inputs/step/token_embed.raw"

    adb_shell(f"mkdir -p {device_dir}/inputs/step {device_dir}/state")
    adb_shell(f"rm -rf {state_path}")
    adb_push(prefill_path, prefill_device)

    tokens = []
    decoded = []

    # Step 0: prefill + 1 generated token.
    out = adb_shell(
        " ".join(
            [
                f"export ADSP_LIBRARY_PATH={device_dir}",
                f"&& export LD_LIBRARY_PATH={device_dir}:/vendor/lib64",
                f"&& cd {device_dir}",
                "&& ./genie-t2t-run",
                f"--config {config_path}",
                f"--embedding_file {prefill_device}",
                f"--embedding_table {device_lut_path}",
                "--embedding_query_output_type token",
                f"--save {state_path}",
            ]
        )
    )
    step_tokens = extract_tokens(out)
    if len(step_tokens) == 0:
        raise RuntimeError("No tokens returned at step 0.")
    tokens.append(step_tokens[0])
    decoded.append(decode_tokens(tokenizer_path, [step_tokens[0]]))

    for _ in range(steps - 1):
        write_token_embedding(lut, tokens[-1], token_embed_host)
        adb_push(token_embed_host, token_embed_device)
        out = adb_shell(
            " ".join(
                [
                    f"export ADSP_LIBRARY_PATH={device_dir}",
                    f"&& export LD_LIBRARY_PATH={device_dir}:/vendor/lib64",
                    f"&& cd {device_dir}",
                    "&& ./genie-t2t-run",
                    f"--config {config_path}",
                    f"--restore {state_path}",
                    f"--embedding_file {token_embed_device}",
                    f"--embedding_table {device_lut_path}",
                    "--embedding_query_output_type token",
                    f"--save {state_path}",
                ]
            )
        )
        step_tokens = extract_tokens(out)
        if len(step_tokens) == 0:
            break
        tokens.append(step_tokens[0])
        decoded.append(decode_tokens(tokenizer_path, [step_tokens[0]]))

    result = {
        "steps": len(tokens),
        "tokens": tokens,
        "decoded_pieces": decoded,
        "decoded_text": decode_tokens(tokenizer_path, tokens),
    }
    with open(os.path.join(output_dir, "tokens.json"), "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)
    print(json.dumps(result, indent=2))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Step-by-step decoding via embeddings.")
    parser.add_argument("--prefill", required=True, help="Prefill embeddings (host path).")
    parser.add_argument("--lut", default="embedding.bin", help="LUT file (host path).")
    parser.add_argument("--tokenizer", default="tokenizer.json", help="Tokenizer file (host path).")
    parser.add_argument("--steps", type=int, default=8, help="Number of tokens to generate.")
    parser.add_argument("--device-dir", default="/data/local/tmp/fastvlm", help="Device directory.")
    parser.add_argument("--state", default="/data/local/tmp/fastvlm/state/step_state.bin", help="State path on device.")
    parser.add_argument("--output-dir", default="host_outputs/step_decode", help="Output directory.")
    parser.add_argument("--config", default="/data/local/tmp/fastvlm/fastvlm_genie_npu.json", help="Config path on device.")
    parser.add_argument("--device-lut", default="/data/local/tmp/fastvlm/embedding.bin", help="LUT path on device.")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_step_decode(
        prefill_path=args.prefill,
        lut_path=args.lut,
        tokenizer_path=args.tokenizer,
        steps=args.steps,
        device_dir=args.device_dir,
        state_path=args.state,
        output_dir=args.output_dir,
        config_path=args.config,
        device_lut_path=args.device_lut,
    )
