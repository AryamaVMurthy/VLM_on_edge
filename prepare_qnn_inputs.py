import argparse
import os

import numpy as np


def build_rope_sin_cos(position_id: int, rope_dim: int, theta: float, dtype: np.dtype):
    inv_freq = 1.0 / (theta ** (np.arange(0, rope_dim, 2, dtype=np.float32) / rope_dim))
    pos = np.array([position_id], dtype=np.float32)
    freqs = pos[:, None] * inv_freq[None, :]
    emb = np.concatenate([freqs, freqs], axis=-1)
    sin = np.sin(emb).astype(dtype, copy=False)
    cos = np.cos(emb).astype(dtype, copy=False)
    return sin.reshape(1, 1, rope_dim), cos.reshape(1, 1, rope_dim)

def create_raw_inputs(
    output_dir="qnn_inputs",
    cache_len=128,
    emb_dim=896,
    num_layers=24,
    position_id=0,
    use_rope=False,
    rope_dim=64,
    rope_theta=10000.0,
):
    os.makedirs(output_dir, exist_ok=True)

    paths = []

    # KV cache inputs (float16)
    for i in range(num_layers):
        name = f"past_key_{i}_in"
        data = np.zeros((1, 2, 64, cache_len), dtype=np.float16)
        filename = f"{name}.raw"
        data.tofile(os.path.join(output_dir, filename))
        paths.append(f"{name}:={filename}")

        name = f"past_value_{i}_in"
        data = np.zeros((1, 2, cache_len, 64), dtype=np.float16)
        filename = f"{name}.raw"
        data.tofile(os.path.join(output_dir, filename))
        paths.append(f"{name}:={filename}")

    # attention_mask
    mask = np.ones((1, 1, 1, cache_len + 1), dtype=np.int32)
    mask.tofile(os.path.join(output_dir, "attention_mask.raw"))
    paths.append("attention_mask:=attention_mask.raw")
    
    # inputs_embeds
    embeds = np.random.randn(1, 1, emb_dim).astype(np.float16)
    embeds.tofile(os.path.join(output_dir, "inputs_embeds.raw"))
    paths.append("inputs_embeds:=inputs_embeds.raw")
    
    if use_rope:
        sin, cos = build_rope_sin_cos(position_id, rope_dim, rope_theta, np.float16)
        sin.tofile(os.path.join(output_dir, "position_ids_sin.raw"))
        paths.append("position_ids_sin:=position_ids_sin.raw")
        cos.tofile(os.path.join(output_dir, "position_ids_cos.raw"))
        paths.append("position_ids_cos:=position_ids_cos.raw")
    else:
        pos = np.full((1, 1), position_id, dtype=np.int32)
        pos.tofile(os.path.join(output_dir, "position_ids.raw"))
        paths.append("position_ids:=position_ids.raw")

    with open(os.path.join(output_dir, "input_list.txt"), "w") as f:
        f.write(" ".join(paths) + "\n")
    
    print(f"Created inputs in {output_dir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Create raw inputs for qnn-net-run.")
    parser.add_argument("--output-dir", default="qnn_inputs")
    parser.add_argument("--cache-len", type=int, default=128)
    parser.add_argument("--emb-dim", type=int, default=896)
    parser.add_argument("--num-layers", type=int, default=24)
    parser.add_argument("--position-id", type=int, default=0)
    parser.add_argument("--use-rope", action="store_true")
    parser.add_argument("--rope-dim", type=int, default=64)
    parser.add_argument("--rope-theta", type=float, default=10000.0)
    args = parser.parse_args()

    create_raw_inputs(
        output_dir=args.output_dir,
        cache_len=args.cache_len,
        emb_dim=args.emb_dim,
        num_layers=args.num_layers,
        position_id=args.position_id,
        use_rope=args.use_rope,
        rope_dim=args.rope_dim,
        rope_theta=args.rope_theta,
    )
