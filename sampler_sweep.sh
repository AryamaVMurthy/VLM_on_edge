#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

IMAGE_PATH="${1:-}"
PROMPT="${2:-Describe the image in 2-3 sentences.}"

if [[ -z "${IMAGE_PATH}" ]]; then
  echo "Usage: $0 /path/to/image [\"prompt\"]"
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

PROMPT_FORMAT="${PROMPT_FORMAT:-raw}"
VISION_TOKENS="${VISION_TOKENS:-256}"
MAX_PROMPT_TOKENS="${MAX_PROMPT_TOKENS:-512}"
MAX_GEN_TOKENS="${MAX_GEN_TOKENS:-64}"
USE_FP16_LUT="${USE_FP16_LUT:-0}"
BASE_CONFIG="${BASE_CONFIG:-${ROOT_DIR}/fastvlm_genie_npu.json}"
STOP_SEQUENCE="${STOP_SEQUENCE:-<|im_end|>}"
EOT_TOKEN="${EOT_TOKEN:-151645}"

SWEEP_DIR="${SWEEP_DIR:-${ROOT_DIR}/host_outputs/sweep}"
mkdir -p "${SWEEP_DIR}"

echo "==> Building cache once (prefix)"
QCOM_ROOT="${QCOM_ROOT}" \
BIN_DIR="${BIN_DIR}" \
USE_FP16_LUT="${USE_FP16_LUT}" \
PROMPT_FORMAT="${PROMPT_FORMAT}" \
VISION_TOKENS="${VISION_TOKENS}" \
MAX_PROMPT_TOKENS="${MAX_PROMPT_TOKENS}" \
"${ROOT_DIR}/cache_kv.sh" "${IMAGE_PATH}"

echo "==> Running sampler sweep"
python3 - <<PY
import json
import os
import re
import subprocess
from collections import Counter
from pathlib import Path

root = Path("${ROOT_DIR}")
sweep_dir = Path("${SWEEP_DIR}")
base_config_path = Path("${BASE_CONFIG}")
query_script = str(root / "query_with_cache.sh")

prompt = "${PROMPT}"
max_gen = int(os.environ.get("MAX_GEN_TOKENS", "${MAX_GEN_TOKENS}"))
prompt_format = os.environ.get("PROMPT_FORMAT", "${PROMPT_FORMAT}")
stop_sequence = os.environ.get("STOP_SEQUENCE", "${STOP_SEQUENCE}")
eot_token = int(os.environ.get("EOT_TOKEN", "${EOT_TOKEN}"))

configs = [
    {
        "name": "greedy",
        "sampler": {"version": 1, "seed": 42, "temp": 0.0, "top-k": 1, "top-p": 1.0, "greedy": True},
    },
    {
        "name": "t0p5_k20_p0p9",
        "sampler": {"version": 1, "seed": 42, "temp": 0.5, "top-k": 20, "top-p": 0.9, "greedy": False},
    },
    {
        "name": "t0p6_k40_p0p9",
        "sampler": {"version": 1, "seed": 42, "temp": 0.6, "top-k": 40, "top-p": 0.9, "greedy": False},
    },
    {
        "name": "t0p7_k40_p0p95",
        "sampler": {"version": 1, "seed": 42, "temp": 0.7, "top-k": 40, "top-p": 0.95, "greedy": False},
    },
    {
        "name": "t0p8_k40_p0p95",
        "sampler": {"version": 1, "seed": 42, "temp": 0.8, "top-k": 40, "top-p": 0.95, "greedy": False},
    },
    {
        "name": "t0p9_k50_p0p95",
        "sampler": {"version": 1, "seed": 42, "temp": 0.9, "top-k": 50, "top-p": 0.95, "greedy": False},
    },
    {
        "name": "t0p7_k20_p0p9",
        "sampler": {"version": 1, "seed": 42, "temp": 0.7, "top-k": 20, "top-p": 0.9, "greedy": False},
    },
    {
        "name": "t0p7_k40_p0p95_rep",
        "sampler": {
            "version": 1,
            "seed": 42,
            "temp": 0.7,
            "top-k": 40,
            "top-p": 0.95,
            "greedy": False,
            "token-penalty": {
                "version": 1,
                "penalize-last-n": 64,
                "repetition-penalty": 1.1,
                "presence-penalty": 0.1,
                "frequency-penalty": 0.1,
            },
        },
    },
    {
        "name": "t0p6_k20_p0p85_rep2",
        "sampler": {
            "version": 1,
            "seed": 42,
            "temp": 0.6,
            "top-k": 20,
            "top-p": 0.85,
            "greedy": False,
            "token-penalty": {
                "version": 1,
                "penalize-last-n": 128,
                "repetition-penalty": 1.2,
                "presence-penalty": 0.2,
                "frequency-penalty": 0.2,
            },
        },
    },
    {
        "name": "t0p7_k20_p0p9_rep3",
        "sampler": {
            "version": 1,
            "seed": 42,
            "temp": 0.7,
            "top-k": 20,
            "top-p": 0.9,
            "greedy": False,
            "token-penalty": {
                "version": 1,
                "penalize-last-n": 128,
                "repetition-penalty": 1.3,
                "presence-penalty": 0.3,
                "frequency-penalty": 0.2,
            },
        },
    },
    {
        "name": "t0p7_k40_p0p9_rep4",
        "sampler": {
            "version": 1,
            "seed": 42,
            "temp": 0.7,
            "top-k": 40,
            "top-p": 0.9,
            "greedy": False,
            "token-penalty": {
                "version": 1,
                "penalize-last-n": 192,
                "repetition-penalty": 1.4,
                "presence-penalty": 0.4,
                "frequency-penalty": 0.3,
            },
        },
    },
    {
        "name": "t0p8_k60_p0p95_rep4",
        "sampler": {
            "version": 1,
            "seed": 42,
            "temp": 0.8,
            "top-k": 60,
            "top-p": 0.95,
            "greedy": False,
            "token-penalty": {
                "version": 1,
                "penalize-last-n": 192,
                "repetition-penalty": 1.4,
                "presence-penalty": 0.4,
                "frequency-penalty": 0.3,
            },
        },
    },
    {
        "name": "t0p9_k60_p0p9_rep5",
        "sampler": {
            "version": 1,
            "seed": 42,
            "temp": 0.9,
            "top-k": 60,
            "top-p": 0.9,
            "greedy": False,
            "token-penalty": {
                "version": 1,
                "penalize-last-n": 256,
                "repetition-penalty": 1.5,
                "presence-penalty": 0.5,
                "frequency-penalty": 0.4,
            },
        },
    },
    {
        "name": "t0p7_k30_p0p85_rep6",
        "sampler": {
            "version": 1,
            "seed": 42,
            "temp": 0.7,
            "top-k": 30,
            "top-p": 0.85,
            "greedy": False,
            "token-penalty": {
                "version": 1,
                "penalize-last-n": 256,
                "repetition-penalty": 1.6,
                "presence-penalty": 0.6,
                "frequency-penalty": 0.4,
            },
        },
    },
]


def extract_text(output: str) -> str:
    m = re.search(r"\\[BEGIN\\]:(.*)\\[END\\]", output, flags=re.S)
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
base = json.loads(base_config_path.read_text())

for cfg in configs:
    name = cfg["name"]
    out_cfg = sweep_dir / f"config_{name}.json"
    profile = sweep_dir / f"profile_{name}.json"
    run_env = os.environ.copy()
    run_env["CONFIG_TEMPLATE"] = str(out_cfg)
    run_env["PROMPT_FORMAT"] = prompt_format
    run_env["MAX_GEN_TOKENS"] = str(max_gen)
    run_env["PROFILE_OUT"] = f"/data/local/tmp/fastvlm/{profile.name}"
    run_env["SAVE_STATE"] = f"/data/local/tmp/fastvlm/state/sweep_state_{name}"

    base_cfg = json.loads(base_config_path.read_text())
    if stop_sequence:
        base_cfg["dialog"]["stop-sequence"] = [stop_sequence]
    if "context" in base_cfg.get("dialog", {}):
        base_cfg["dialog"]["context"]["eot-token"] = eot_token
    base_cfg["dialog"]["sampler"] = cfg["sampler"]
    out_cfg.write_text(json.dumps(base_cfg, indent=4) + "\\n")

    # clear device profile path to avoid collisions
    subprocess.run(["adb", "shell", "rm", "-f", run_env["PROFILE_OUT"]], check=False)
    subprocess.run(["adb", "shell", "rm", "-rf", run_env["SAVE_STATE"]], check=False)

    p = subprocess.run(
        [query_script, prompt],
        env=run_env,
        capture_output=True,
        text=True,
    )
    output = (p.stdout or "") + "\\n" + (p.stderr or "")
    text = extract_text(output)

    # pull profile
    subprocess.run(
        ["adb", "pull", run_env["PROFILE_OUT"], str(profile)],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        check=False,
    )
    ttfs = read_ttfs(profile)
    score = score_text(text)

    results.append(
        {
            "name": name,
            "ttfs_s": ttfs,
            "return_code": p.returncode,
            "output": text,
            **score,
        }
    )

results_path = sweep_dir / "results.json"
results_path.write_text(json.dumps(results, indent=2) + "\\n")

valid = [r for r in results if r["return_code"] == 0 and r["ttfs_s"] < 1.0]
valid.sort(key=lambda r: r["score"], reverse=True)

print("\\n==> Sweep results (top 5)")
for r in valid[:5]:
    print(f"{r['name']}: score={r['score']:.3f}, ttfs={r['ttfs_s']:.3f}s, unique={r['unique_ratio']:.2f}, max_run={r['max_run_ratio']:.2f}")

if not valid:
    raise SystemExit("No valid configs produced TTFS < 1s and output.")

best = valid[0]
print(f"\\n==> Best: {best['name']} (score={best['score']:.3f}, ttfs={best['ttfs_s']:.3f}s)")

# Write best config to fastvlm_genie_npu.json
best_cfg_path = sweep_dir / f"config_{best['name']}.json"
base_config_path.write_text(best_cfg_path.read_text())

# Save best output
(sweep_dir / "best_output.txt").write_text(best.get("output", "") + "\\n")
PY

echo "==> Sweep complete. Results in ${SWEEP_DIR}"
