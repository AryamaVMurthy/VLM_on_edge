#!/usr/bin/env python3
import argparse
import re
from typing import List, Optional

import onnx
from onnx import TensorProto, helper


def _safe(name: str) -> str:
    name = re.sub(r"[^a-zA-Z0-9_]+", "_", name)
    return name or "node"


def _const_int64(name: str, vals: List[int]) -> onnx.NodeProto:
    tensor = helper.make_tensor(
        name=f"{name}_value", data_type=TensorProto.INT64, dims=[len(vals)], vals=vals
    )
    return helper.make_node("Constant", [], [name], name=f"{name}_const", value=tensor)


def _const_float(name: str, vals: List[float], dtype: int = TensorProto.FLOAT) -> onnx.NodeProto:
    tensor = helper.make_tensor(
        name=f"{name}_value", data_type=dtype, dims=[len(vals)], vals=vals
    )
    return helper.make_node("Constant", [], [name], name=f"{name}_const", value=tensor)


def _shape_gather(prefix: str, x: str, idx: int) -> List[onnx.NodeProto]:
    shape = f"{prefix}_shape_{idx}"
    idx_c = f"{prefix}_idx{idx}"
    out = f"{prefix}_dim{idx}"
    nodes = [
        helper.make_node("Shape", [x], [shape], name=f"{prefix}_shape_node"),
        _const_int64(idx_c, [idx]),
        helper.make_node("Gather", [shape, idx_c], [out], name=f"{prefix}_gather{idx}"),
    ]
    return nodes


def _reshape_to_heads(prefix: str, x: str, heads: int, head_dim: int) -> List[onnx.NodeProto]:
    nodes = []
    nodes += _shape_gather(prefix, x, 0)
    nodes += _shape_gather(prefix, x, 1)
    b = f"{prefix}_dim0"
    s = f"{prefix}_dim1"
    heads_c = f"{prefix}_heads"
    head_dim_c = f"{prefix}_head_dim"
    nodes.append(_const_int64(heads_c, [heads]))
    nodes.append(_const_int64(head_dim_c, [head_dim]))
    new_shape = f"{prefix}_shape_new"
    nodes.append(
        helper.make_node("Concat", [b, s, heads_c, head_dim_c], [new_shape], name=f"{prefix}_concat", axis=0)
    )
    out = f"{prefix}_reshaped"
    nodes.append(helper.make_node("Reshape", [x, new_shape], [out], name=f"{prefix}_reshape"))
    return nodes


def _interleave_even_odd(prefix: str, even: str, odd: str, heads: int, head_dim: int) -> List[onnx.NodeProto]:
    nodes = []
    concat = f"{prefix}_concat_eo"
    nodes.append(helper.make_node("Concat", [even, odd], [concat], name=f"{prefix}_concat", axis=3))
    # reshape to [B, S, H, 2, D/2]
    nodes += _shape_gather(prefix + "_eo", even, 0)
    nodes += _shape_gather(prefix + "_eo", even, 1)
    nodes += _shape_gather(prefix + "_eo", even, 2)
    b = f"{prefix}_eo_dim0"
    s = f"{prefix}_eo_dim1"
    h = f"{prefix}_eo_dim2"
    two_c = f"{prefix}_two"
    half_c = f"{prefix}_half"
    nodes.append(_const_int64(two_c, [2]))
    nodes.append(_const_int64(half_c, [head_dim // 2]))
    shape1 = f"{prefix}_shape1"
    nodes.append(
        helper.make_node("Concat", [b, s, h, two_c, half_c], [shape1], name=f"{prefix}_concat1", axis=0)
    )
    reshaped = f"{prefix}_reshape1"
    nodes.append(helper.make_node("Reshape", [concat, shape1], [reshaped], name=f"{prefix}_reshape1"))
    # transpose last two dims: [B,S,H,2,D/2] -> [B,S,H,D/2,2]
    trans = f"{prefix}_transpose"
    nodes.append(helper.make_node("Transpose", [reshaped], [trans], name=f"{prefix}_transpose", perm=[0, 1, 2, 4, 3]))
    # reshape back to [B,S,H,D]
    shape2 = f"{prefix}_shape2"
    hd_c = f"{prefix}_hd"
    nodes.append(_const_int64(hd_c, [head_dim]))
    nodes.append(
        helper.make_node(
            "Concat", [b, s, h, hd_c], [shape2], name=f"{prefix}_concat2", axis=0
        )
    )
    out = f"{prefix}_out"
    nodes.append(helper.make_node("Reshape", [trans, shape2], [out], name=f"{prefix}_reshape2"))
    return nodes


def _replace_rotary(nodes: List[onnx.NodeProto], node: onnx.NodeProto, head_dim: int, q_heads: int, kv_heads: int) -> List[onnx.NodeProto]:
    prefix = _safe(node.name)
    x, pos, cos_cache, sin_cache = node.input
    out = node.output[0]
    # pick heads based on input name
    heads = q_heads if "q_proj" in x or "q_rotary" in node.name else kv_heads
    local_nodes: List[onnx.NodeProto] = []
    local_nodes += _reshape_to_heads(prefix + "_rs", x, heads, head_dim)
    x_rs = f"{prefix}_rs_reshaped"

    # Gather cos/sin by position ids
    cos = f"{prefix}_cos"
    sin = f"{prefix}_sin"
    local_nodes.append(helper.make_node("Gather", [cos_cache, pos], [cos], name=f"{prefix}_gather_cos", axis=0))
    local_nodes.append(helper.make_node("Gather", [sin_cache, pos], [sin], name=f"{prefix}_gather_sin", axis=0))

    # Unsqueeze to [B,S,1,D/2]
    axes_c = f"{prefix}_axes"
    local_nodes.append(_const_int64(axes_c, [2]))
    cos_u = f"{prefix}_cos_u"
    sin_u = f"{prefix}_sin_u"
    local_nodes.append(helper.make_node("Unsqueeze", [cos, axes_c], [cos_u], name=f"{prefix}_unsq_cos"))
    local_nodes.append(helper.make_node("Unsqueeze", [sin, axes_c], [sin_u], name=f"{prefix}_unsq_sin"))

    # Slice even/odd
    starts0 = f"{prefix}_starts0"
    ends0 = f"{prefix}_ends0"
    axes0 = f"{prefix}_axes0"
    steps2 = f"{prefix}_steps2"
    local_nodes.append(_const_int64(starts0, [0]))
    local_nodes.append(_const_int64(ends0, [head_dim]))
    local_nodes.append(_const_int64(axes0, [3]))
    local_nodes.append(_const_int64(steps2, [2]))
    even = f"{prefix}_even"
    local_nodes.append(helper.make_node("Slice", [x_rs, starts0, ends0, axes0, steps2], [even], name=f"{prefix}_slice_even"))

    starts1 = f"{prefix}_starts1"
    local_nodes.append(_const_int64(starts1, [1]))
    odd = f"{prefix}_odd"
    local_nodes.append(helper.make_node("Slice", [x_rs, starts1, ends0, axes0, steps2], [odd], name=f"{prefix}_slice_odd"))

    # Apply rotation
    even_cos = f"{prefix}_even_cos"
    odd_sin = f"{prefix}_odd_sin"
    local_nodes.append(helper.make_node("Mul", [even, cos_u], [even_cos], name=f"{prefix}_mul_even_cos"))
    local_nodes.append(helper.make_node("Mul", [odd, sin_u], [odd_sin], name=f"{prefix}_mul_odd_sin"))
    out_even = f"{prefix}_out_even"
    local_nodes.append(helper.make_node("Sub", [even_cos, odd_sin], [out_even], name=f"{prefix}_sub"))

    even_sin = f"{prefix}_even_sin"
    odd_cos = f"{prefix}_odd_cos"
    local_nodes.append(helper.make_node("Mul", [even, sin_u], [even_sin], name=f"{prefix}_mul_even_sin"))
    local_nodes.append(helper.make_node("Mul", [odd, cos_u], [odd_cos], name=f"{prefix}_mul_odd_cos"))
    out_odd = f"{prefix}_out_odd"
    local_nodes.append(helper.make_node("Add", [even_sin, odd_cos], [out_odd], name=f"{prefix}_add"))

    # Interleave and reshape back to [B,S,hidden]
    local_nodes += _interleave_even_odd(prefix + "_ilv", out_even, out_odd, heads, head_dim)
    inter = f"{prefix}_ilv_out"

    # reshape to [B,S,hidden]
    local_nodes += _shape_gather(prefix + "_r2", x, 0)
    local_nodes += _shape_gather(prefix + "_r2", x, 1)
    b = f"{prefix}_r2_dim0"
    s = f"{prefix}_r2_dim1"
    hidden_c = f"{prefix}_hidden"
    local_nodes.append(_const_int64(hidden_c, [heads * head_dim]))
    shape = f"{prefix}_out_shape"
    local_nodes.append(helper.make_node("Concat", [b, s, hidden_c], [shape], name=f"{prefix}_concat_out", axis=0))
    local_nodes.append(helper.make_node("Reshape", [inter, shape], [out], name=f"{prefix}_reshape_out"))

    return local_nodes


def _repeat_kv(prefix: str, x_kv: str, q_heads: int, kv_heads: int) -> List[onnx.NodeProto]:
    nodes = []
    # unsqueeze head grouping axis
    axes = f"{prefix}_axes"
    nodes.append(_const_int64(axes, [2]))
    unsq = f"{prefix}_unsq"
    nodes.append(helper.make_node("Unsqueeze", [x_kv, axes], [unsq], name=f"{prefix}_unsq"))
    # tile
    reps = f"{prefix}_reps"
    group = q_heads // kv_heads
    nodes.append(_const_int64(reps, [1, 1, group, 1, 1]))
    tiled = f"{prefix}_tiled"
    nodes.append(helper.make_node("Tile", [unsq, reps], [tiled], name=f"{prefix}_tile"))
    # reshape to [B, q_heads, T, D]
    nodes += _shape_gather(prefix + "_sh", x_kv, 0)
    nodes += _shape_gather(prefix + "_sh", x_kv, 2)
    nodes += _shape_gather(prefix + "_sh", x_kv, 3)
    b = f"{prefix}_sh_dim0"
    t = f"{prefix}_sh_dim2"
    d = f"{prefix}_sh_dim3"
    q_c = f"{prefix}_q_heads"
    nodes.append(_const_int64(q_c, [q_heads]))
    shape = f"{prefix}_shape"
    nodes.append(helper.make_node("Concat", [b, q_c, t, d], [shape], name=f"{prefix}_concat", axis=0))
    out = f"{prefix}_out"
    nodes.append(helper.make_node("Reshape", [tiled, shape], [out], name=f"{prefix}_reshape"))
    return nodes


def _replace_gqa(node: onnx.NodeProto, head_dim: int) -> List[onnx.NodeProto]:
    prefix = _safe(node.name)
    q_in, k_in, v_in, past_k, past_v, attn_mask = node.input[:6]
    out, present_k, present_v = node.output[:3]
    q_heads = int([a.i for a in node.attribute if a.name == "num_heads"][0])
    kv_heads = int([a.i for a in node.attribute if a.name == "kv_num_heads"][0])
    scale = float([a.f for a in node.attribute if a.name == "scale"][0])

    nodes: List[onnx.NodeProto] = []
    # reshape q/k/v to [B,S,H,D] and transpose to [B,H,S,D]
    nodes += _reshape_to_heads(prefix + "_q", q_in, q_heads, head_dim)
    q_rs = f"{prefix}_q_reshaped"
    q_t = f"{prefix}_q_t"
    nodes.append(helper.make_node("Transpose", [q_rs], [q_t], name=f"{prefix}_q_trans", perm=[0, 2, 1, 3]))

    nodes += _reshape_to_heads(prefix + "_k", k_in, kv_heads, head_dim)
    k_rs = f"{prefix}_k_reshaped"
    k_t = f"{prefix}_k_t"
    nodes.append(helper.make_node("Transpose", [k_rs], [k_t], name=f"{prefix}_k_trans", perm=[0, 2, 1, 3]))

    nodes += _reshape_to_heads(prefix + "_v", v_in, kv_heads, head_dim)
    v_rs = f"{prefix}_v_reshaped"
    v_t = f"{prefix}_v_t"
    nodes.append(helper.make_node("Transpose", [v_rs], [v_t], name=f"{prefix}_v_trans", perm=[0, 2, 1, 3]))

    # concat past along seq axis (axis=2)
    k_total = f"{prefix}_k_total"
    v_total = f"{prefix}_v_total"
    nodes.append(helper.make_node("Concat", [past_k, k_t], [k_total], name=f"{prefix}_k_concat", axis=2))
    nodes.append(helper.make_node("Concat", [past_v, v_t], [v_total], name=f"{prefix}_v_concat", axis=2))

    # outputs for present (kv heads)
    nodes.append(helper.make_node("Identity", [k_total], [present_k], name=f"{prefix}_present_k"))
    nodes.append(helper.make_node("Identity", [v_total], [present_v], name=f"{prefix}_present_v"))

    # repeat kv to q heads
    nodes += _repeat_kv(prefix + "_krep", k_total, q_heads, kv_heads)
    k_rep = f"{prefix}_krep_out"
    nodes += _repeat_kv(prefix + "_vrep", v_total, q_heads, kv_heads)
    v_rep = f"{prefix}_vrep_out"

    # attention scores
    k_rep_t = f"{prefix}_krep_t"
    nodes.append(helper.make_node("Transpose", [k_rep], [k_rep_t], name=f"{prefix}_krep_trans", perm=[0, 1, 3, 2]))
    scores = f"{prefix}_scores"
    nodes.append(helper.make_node("MatMul", [q_t, k_rep_t], [scores], name=f"{prefix}_matmul"))
    scale_c = f"{prefix}_scale"
    nodes.append(_const_float(scale_c, [scale], dtype=TensorProto.FLOAT16))
    scores_scaled = f"{prefix}_scores_scaled"
    nodes.append(helper.make_node("Mul", [scores, scale_c], [scores_scaled], name=f"{prefix}_scale_mul"))

    # attention mask: (1 - mask) * -10000
    mask_f = f"{prefix}_mask_f"
    nodes.append(helper.make_node("Cast", [attn_mask], [mask_f], name=f"{prefix}_mask_cast", to=TensorProto.FLOAT16))
    one_c = f"{prefix}_one"
    nodes.append(_const_float(one_c, [1.0], dtype=TensorProto.FLOAT16))
    inv_mask = f"{prefix}_inv_mask"
    nodes.append(helper.make_node("Sub", [one_c, mask_f], [inv_mask], name=f"{prefix}_mask_inv"))
    axes = f"{prefix}_mask_axes"
    nodes.append(_const_int64(axes, [1, 2]))
    inv_mask_u = f"{prefix}_inv_mask_u"
    nodes.append(helper.make_node("Unsqueeze", [inv_mask, axes], [inv_mask_u], name=f"{prefix}_mask_unsq"))
    neg_c = f"{prefix}_neg"
    nodes.append(_const_float(neg_c, [-10000.0], dtype=TensorProto.FLOAT16))
    mask_bias = f"{prefix}_mask_bias"
    nodes.append(helper.make_node("Mul", [inv_mask_u, neg_c], [mask_bias], name=f"{prefix}_mask_bias_mul"))
    scores_masked = f"{prefix}_scores_masked"
    nodes.append(helper.make_node("Add", [scores_scaled, mask_bias], [scores_masked], name=f"{prefix}_mask_add"))

    # softmax and context
    probs = f"{prefix}_probs"
    nodes.append(helper.make_node("Softmax", [scores_masked], [probs], name=f"{prefix}_softmax", axis=-1))
    ctx = f"{prefix}_ctx"
    nodes.append(helper.make_node("MatMul", [probs, v_rep], [ctx], name=f"{prefix}_ctx_matmul"))
    # reshape back to [B,S,H*D]
    ctx_t = f"{prefix}_ctx_t"
    nodes.append(helper.make_node("Transpose", [ctx], [ctx_t], name=f"{prefix}_ctx_trans", perm=[0, 2, 1, 3]))
    # shape from q input
    nodes += _shape_gather(prefix + "_out", q_in, 0)
    nodes += _shape_gather(prefix + "_out", q_in, 1)
    b = f"{prefix}_out_dim0"
    s = f"{prefix}_out_dim1"
    hidden_c = f"{prefix}_hidden"
    nodes.append(_const_int64(hidden_c, [q_heads * head_dim]))
    shape = f"{prefix}_out_shape"
    nodes.append(helper.make_node("Concat", [b, s, hidden_c], [shape], name=f"{prefix}_out_concat", axis=0))
    nodes.append(helper.make_node("Reshape", [ctx_t, shape], [out], name=f"{prefix}_out_reshape"))

    return nodes


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    model = onnx.load(args.input)
    # determine head_dim from cos_cache
    init_map = {t.name: t for t in model.graph.initializer}
    cos = init_map.get("cos_cache")
    if cos is None:
        raise RuntimeError("cos_cache initializer not found")
    head_dim = int(cos.dims[1]) * 2

    # infer heads from first GroupQueryAttention node
    q_heads = None
    kv_heads = None
    for n in model.graph.node:
        if n.op_type == "GroupQueryAttention":
            for a in n.attribute:
                if a.name == "num_heads":
                    q_heads = int(a.i)
                if a.name == "kv_num_heads":
                    kv_heads = int(a.i)
            break
    if q_heads is None or kv_heads is None:
        raise RuntimeError("GroupQueryAttention attributes not found")

    new_nodes: List[onnx.NodeProto] = []
    replaced = 0
    for node in model.graph.node:
        if node.op_type == "RotaryEmbedding":
            new_nodes.extend(_replace_rotary(new_nodes, node, head_dim, q_heads, kv_heads))
            replaced += 1
            continue
        if node.op_type == "GroupQueryAttention":
            new_nodes.extend(_replace_gqa(node, head_dim))
            replaced += 1
            continue
        new_nodes.append(node)

    model.graph.ClearField("node")
    model.graph.node.extend(new_nodes)
    onnx.save(model, args.output)
    print(f"Replaced {replaced} fused attention ops. Saved to {args.output}")


if __name__ == "__main__":
    main()
