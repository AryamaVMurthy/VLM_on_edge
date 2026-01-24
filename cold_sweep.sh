#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

IMAGE_PATH="${1:-${ROOT_DIR}/image.png}"
PROMPT="${2:-<|im_start|>system
You are a helpful assistant.
<|im_end|>
<|im_start|>user
<image>
Describe the image in 2-3 sentences.
<|im_end|>
<|im_start|>assistant
}"

if [[ ! -f "${IMAGE_PATH}" ]]; then
  echo "Image not found: ${IMAGE_PATH}"
  exit 1
fi

QCOM_ROOT="${QCOM_ROOT:-}"
BIN_DIR="${BIN_DIR:-}"
if [[ -z "${QCOM_ROOT}" || ! -d "${QCOM_ROOT}" ]]; then
  echo "Set QCOM_ROOT to a valid qcom_ai_stack (e.g., qcom_ai_stack_2.41)."
  exit 1
fi
if [[ -z "${BIN_DIR}" || ! -d "${BIN_DIR}" ]]; then
  echo "Set BIN_DIR to a valid compiled bin directory."
  exit 1
fi

BASE_CONFIG="${BASE_CONFIG:-${ROOT_DIR}/fastvlm_genie_npu.json}"
SWEEP_DIR="${SWEEP_DIR:-${ROOT_DIR}/host_outputs/sweep_cold}"
mkdir -p "${SWEEP_DIR}"

ROOT_DIR="${ROOT_DIR}" \
IMAGE_PATH="${IMAGE_PATH}" \
PROMPT="${PROMPT}" \
BASE_CONFIG="${BASE_CONFIG}" \
QCOM_ROOT="${QCOM_ROOT}" \
BIN_DIR="${BIN_DIR}" \
SWEEP_DIR="${SWEEP_DIR}" \
python3 - <<'PY'
import json
import os
import re
import subprocess
from collections import Counter
from pathlib import Path

root = Path(os.environ["ROOT_DIR"])
image_path = os.environ["IMAGE_PATH"]
prompt = os.environ["PROMPT"]
base_config = Path(os.environ["BASE_CONFIG"])
qcom_root = os.environ["QCOM_ROOT"]
bin_dir = os.environ["BIN_DIR"]
sweep_dir = Path(os.environ["SWEEP_DIR"])

max_prompt_tokens = int(os.environ.get("MAX_PROMPT_TOKENS", "512"))
max_gen_tokens = int(os.environ.get("MAX_GEN_TOKENS", "64"))
vision_tokens = int(os.environ.get("VISION_TOKENS", "256"))
vision_stride = int(os.environ.get("VISION_STRIDE", "1"))

run_script = str(root / "run_e2e_vlm.sh")

configs = [
    {
        "name": "base",
        "sampler": {"version": 1, "seed": 42, "temp": 0.6, "top-k": 40, "top-p": 0.9, "greedy": False,
                    "token-penalty": {"version": 1, "penalize-last-n": 128, "repetition-penalty": 1.2,
                                     "presence-penalty": 0.2, "frequency-penalty": 0.1}},
        "use_fp16_lut": False,
        "add_bos": False,
    },
    {
        "name": "t0p5_k40_p0p9_rep",
        "sampler": {"version": 1, "seed": 42, "temp": 0.5, "top-k": 40, "top-p": 0.9, "greedy": False,
                    "token-penalty": {"version": 1, "penalize-last-n": 192, "repetition-penalty": 1.3,
                                     "presence-penalty": 0.3, "frequency-penalty": 0.2}},
        "use_fp16_lut": False,
        "add_bos": False,
    },
    {
        "name": "t0p7_k50_p0p95_rep",
        "sampler": {"version": 1, "seed": 42, "temp": 0.7, "top-k": 50, "top-p": 0.95, "greedy": False,
                    "token-penalty": {"version": 1, "penalize-last-n": 192, "repetition-penalty": 1.35,
                                     "presence-penalty": 0.3, "frequency-penalty": 0.2}},
        "use_fp16_lut": False,
        "add_bos": False,
    },
    {
        "name": "t0p8_k60_p0p95_rep",
        "sampler": {"version": 1, "seed": 42, "temp": 0.8, "top-k": 60, "top-p": 0.95, "greedy": False,
                    "token-penalty": {"version": 1, "penalize-last-n": 256, "repetition-penalty": 1.4,
                                     "presence-penalty": 0.4, "frequency-penalty": 0.3}},
        "use_fp16_lut": False,
        "add_bos": False,
    },
    {
        "name": "greedy",
        "sampler": {"version": 1, "seed": 42, "temp": 0.0, "top-k": 1, "top-p": 1.0, "greedy": True},
        "use_fp16_lut": False,
        "add_bos": False,
    },
    {
        "name": "fp16_native",
        "sampler": {"version": 1, "seed": 42, "temp": 0.6, "top-k": 40, "top-p": 0.9, "greedy": False,
                    "token-penalty": {"version": 1, "penalize-last-n": 128, "repetition-penalty": 1.2,
                                     "presence-penalty": 0.2, "frequency-penalty": 0.1}},
        "use_fp16_lut": True,
        "add_bos": False,
    },
    {
        "name": "fp16_native_bos",
        "sampler": {"version": 1, "seed": 42, "temp": 0.6, "top-k": 40, "top-p": 0.9, "greedy": False,
                    "token-penalty": {"version": 1, "penalize-last-n": 128, "repetition-penalty": 1.2,
                                     "presence-penalty": 0.2, "frequency-penalty": 0.1}},
        "use_fp16_lut": True,
        "add_bos": True,
    },
]


def extract_text(output: str) -> str:
    m = re.search(r"\[BEGIN\]:(.*)\[END\]", output, flags=re.S)
    if not m:
        return ""
    return m.group(1).strip()


def score_text(text: str) -> dict:
    words = re.findall(r"[A-Za-z']+", text.lower())
    if not words:
        return {
            "word_count": 0,
            "unique_ratio": 0.0,
            "max_freq_ratio": 1.0,
            "max_run_ratio": 1.0,
            "max_bigram_ratio": 1.0,
            "score": -1.0,
        }
    total = len(words)
    counts = Counter(words)
    unique_ratio = len(counts) / total
    max_freq_ratio = max(counts.values()) / total
    max_run = 1
    cur = 1
    for i in range(1, total):
        if words[i] == words[i - 1]:
            cur += 1
            max_run = max(max_run, cur)
        else:
            cur = 1
    max_run_ratio = max_run / total
    bigrams = list(zip(words, words[1:]))
    max_bigram_ratio = 0.0
    if bigrams:
        bigram_counts = Counter(bigrams)
        max_bigram_ratio = max(bigram_counts.values()) / max(len(bigrams), 1)
    score = unique_ratio - max_freq_ratio - max_run_ratio - max_bigram_ratio
    if total < 8:
        score -= 0.5
    return {
        "word_count": total,
        "unique_ratio": unique_ratio,
        "max_freq_ratio": max_freq_ratio,
        "max_run_ratio": max_run_ratio,
        "max_bigram_ratio": max_bigram_ratio,
        "score": score,
    }


def read_ttfs(profile_path: Path) -> float:
    try:
        data = json.loads(profile_path.read_text())
    except Exception:
        return 9999.0
    for comp in data.get("components", []):
        for ev in comp.get("events", []):
            if ev.get("type") == "GenieDialog_query":
                ttfs = ev.get("time-to-first-token", {}).get("value")
                if ttfs is None:
                    return 9999.0
                return ttfs / 1e6
    return 9999.0

results = []
first = True
base_cfg = json.loads(base_config.read_text())

for cfg in configs:
    name = cfg["name"]
    out_cfg = sweep_dir / f"config_{name}.json"
    profile = sweep_dir / f"profile_{name}.json"

    config_data = json.loads(base_config.read_text())
    config_data["dialog"]["sampler"] = cfg["sampler"]

    if cfg["use_fp16_lut"]:
        config_data["dialog"]["embedding"]["lut-path"] = "/data/local/tmp/fastvlm/embedding_fp16.bin"
        config_data["dialog"]["embedding"]["datatype"] = "native"

    out_cfg.write_text(json.dumps(config_data, indent=4) + "\n")

    env = os.environ.copy()
    env.update({
        "QCOM_ROOT": qcom_root,
        "BIN_DIR": bin_dir,
        "CONFIG_TEMPLATE": str(out_cfg),
        "PROMPT_FORMAT": "raw",
        "ADD_IMAGE_TOKEN": "0",
        "MAX_PROMPT_TOKENS": str(max_prompt_tokens),
        "MAX_GEN_TOKENS": str(max_gen_tokens),
        "VISION_TOKENS": str(vision_tokens),
        "VISION_STRIDE": str(vision_stride),
        "SKIP_PUSH": "0" if first else "1",
        "PUSH_TOKEN_FILES": "0",
        "FORCE_TOKEN_PREFILL": "0",
        "PROFILE_OUT": f"/data/local/tmp/fastvlm/{profile.name}",
    })

    if cfg.get("add_bos"):
        env["ADD_BOS"] = "1"

    if cfg["use_fp16_lut"]:
        env["USE_FP16_LUT"] = "1"

    subprocess.run(["adb", "shell", "rm", "-f", env["PROFILE_OUT"]], check=False)

    p = subprocess.run(
        [run_script, image_path, prompt],
        env=env,
        capture_output=True,
        text=True,
    )
    output = (p.stdout or "") + "\n" + (p.stderr or "")
    text = extract_text(output)

    subprocess.run(
        ["adb", "pull", env["PROFILE_OUT"], str(profile)],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        check=False,
    )
    ttfs = read_ttfs(profile)
    score = score_text(text)

    results.append({
        "name": name,
        "return_code": p.returncode,
        "ttfs_s": ttfs,
        "output": text,
        **score,
    })

    first = False

results_path = sweep_dir / "results.json"
results_path.write_text(json.dumps(results, indent=2) + "\n")

print("\n==> Cold sweep results")
for r in results:
    print(f"{r['name']}: rc={r['return_code']}, ttfs={r['ttfs_s']:.3f}s, score={r['score']:.3f}, words={r['word_count']}")

# pick best by score, then ttfs
valid = [r for r in results if r["return_code"] == 0 and r["word_count"] > 0]
if valid:
    valid.sort(key=lambda r: (r["score"], -r["ttfs_s"]), reverse=True)
    best = valid[0]
    print(f"\n==> Best: {best['name']} (score={best['score']:.3f}, ttfs={best['ttfs_s']:.3f}s)")
    (sweep_dir / "best_output.txt").write_text(best.get("output", "") + "\n")
else:
    print("\nNo valid outputs produced.")
PY
