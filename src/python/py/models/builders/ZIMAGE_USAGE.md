# How to Run: Z-Image-Turbo Transformer Trunk Exporter

This exports the transformer trunk of Z-Image-Turbo only (no text encoder, no VAE) as a
standalone ONNX graph with dynamic height/width and batch size 1. See
[ZIMAGE_DESIGN.md](ZIMAGE_DESIGN.md) for the architecture and scope this implies.

## Contents

- [Prerequisites](#prerequisites)
- [Getting the Checkpoint](#getting-the-checkpoint)
- [Building the ONNX Model](#building-the-onnx-model)
- [Precondition on Resolution and Caption Length](#precondition-on-resolution-and-caption-length)
- [Output Files](#output-files)
- [Running the Exported Graph](#running-the-exported-graph)
- [Verifying Numerical Correctness](#verifying-numerical-correctness)
- [Troubleshooting](#troubleshooting)

## Prerequisites

In addition to this repo's normal model-builder dependencies (`torch`, `onnx_ir`,
`transformers`, `onnxruntime`), you need `diffusers` (it defines
`ZImageTransformer2DModel`, which the exporter loads the real weights through):

```bash
pip install diffusers pillow
```

`pillow` is a transitive dependency of `diffusers` (used by unrelated image-IO utilities);
if you're in a fully offline/sandboxed environment without it, see
[Troubleshooting](#troubleshooting).

## Getting the Checkpoint

```py
from huggingface_hub import snapshot_download
snapshot_download("Tongyi-MAI/Z-Image-Turbo", local_dir="path_to_local_folder")
```

The transformer weights live in the `transformer/` subfolder of that repo
(`transformer/config.json`, `transformer/diffusion_pytorch_model*.safetensors`) — point
`-i` at that subfolder, not the repo root.

## Building the ONNX Model

```bash
# From source, from src/python/py/models:
python builder.py \
  -e cpu -p fp32 \
  -i path_to_local_folder/transformer \
  -o path_to_output_folder \
  -c cache_dir_to_store_temp_files
```

This works with any of the usual `-p`/`-e` combinations this tool supports (e.g.
`-p int4 -e webgpu`) — every `Linear` in the exported graph goes through the same
quantization path as every other model here, so `-p int4`/`-p int8` "just work". No new
`--extra_options` are introduced for this model; existing quantization options
(`int4_block_size`/`accuracy_level`/etc., see `README.md`) apply as-is.

Example matching this repo's existing `build_model.py` wrapper convention (WebGPU INT4):

```bash
python builder.py -e webgpu -p int4 \
  --extra_options int4_block_size=32 int4_accuracy_level=4 int4_op_types_to_quantize=MatMul \
  -i path_to_local_folder/transformer \
  -o Z-Image-Turbo-transformer-genai-wgpu-int4 \
  -c tmp
```

Build time is dominated by loading the checkpoint (~6B parameters) and writing weights;
expect a few minutes.

## Precondition on Resolution and Caption Length

Because the exported graph has no padding/masking logic (see
[ZIMAGE_DESIGN.md#scope](ZIMAGE_DESIGN.md#scope)), the caller must ensure:

- `(height / patch_size) * (width / patch_size) % 32 == 0`, where `patch_size = 2`. This
  holds for every standard resolution divisible by 16 (512, 768, 1024, ...) in both
  dimensions, including non-square combinations of them.
- The caption embedding's sequence length (`encoder_hidden_states.shape[1]`) is already a
  multiple of 32 tokens (pad it yourself before calling the graph if the text encoder's
  natural output isn't).

Violating either precondition doesn't error at export time — it will silently compute the
wrong thing at inference time (the graph has no way to detect it), so get this right in
your calling code.

## Output Files

| File | Contents |
|---|---|
| `model.onnx` (+ `model.onnx.data`) | the transformer graph, external weight data |
| `genai_config.json` | minimal metadata: model type, dims, and the I/O names/shapes documented above (not consumed by onnxruntime-genai's generate loop — this is a standalone graph) |

## Running the Exported Graph

```py
import numpy as np
import onnxruntime as ort

sess = ort.InferenceSession("path_to_output_folder/model.onnx", providers=["CPUExecutionProvider"])

# hidden_states: [1, 16, H, W] float; H, W must satisfy the precondition above.
# encoder_hidden_states: [1, cap_len, 2560] float; cap_len must be a multiple of 32.
# timestep: [1] float.
sample = sess.run(
    ["sample"],
    {
        "hidden_states": hidden_states_np,
        "encoder_hidden_states": encoder_hidden_states_np,
        "timestep": timestep_np,
    },
)[0]
```

Drive your own diffusion sampling loop (e.g. `FlowMatchEulerDiscreteScheduler`, matching
Z-Image-Turbo's `scheduler/scheduler_config.json`) around this call, feeding it the
caption embedding from a Qwen3 text encoder and decoding the final latent with the model's
VAE — both out of scope for this exporter.

## Verifying Numerical Correctness

The graph was validated by comparing it against a hand-written PyTorch reference that
mirrors `zimage.py`'s graph node-for-node, using the same real loaded weights, at multiple
resolutions (including a non-square one, to prove the dynamic-shape claim). Adapt this
script to re-run that check after any change to `zimage.py`:

```py
import torch
import numpy as np
import onnxruntime as ort
from diffusers import ZImageTransformer2DModel

MODEL_DIR = "path_to_local_folder/transformer"
ONNX_PATH = "path_to_output_folder/model.onnx"


def reference_forward(model, hidden_states, encoder_hidden_states, timestep):
    """Mirrors builders/zimage.py's graph construction step-for-step in plain PyTorch."""
    C = model.config.in_channels
    dim = model.config.dim
    n_heads = model.config.n_heads
    axes_dims = model.config.axes_dims
    axes_lens = model.config.axes_lens
    theta = model.config.rope_theta
    head_size = sum(axes_dims)

    _, _, H, W = hidden_states.shape
    cap_len = encoder_hidden_states.shape[1]
    patch = model.config.all_patch_size[0]
    h_tok, w_tok = H // patch, W // patch
    num_img_tokens = h_tok * w_tok

    tables = []
    for d, length in zip(axes_dims, axes_lens):
        freqs = 1.0 / (theta ** (torch.arange(0, d, 2, dtype=torch.float64) / d))
        pos = torch.arange(length, dtype=torch.float64)
        angles = torch.outer(pos, freqs).float()
        cos_v = torch.cos(angles).repeat_interleave(2, dim=1)
        sin_v = torch.sin(angles).repeat_interleave(2, dim=1)
        tables.append(torch.stack([cos_v, sin_v], dim=-1))

    def build_freqs(axis0, axis1, axis2):
        parts = [tables[i][ids] for i, ids in enumerate((axis0, axis1, axis2))]
        return torch.cat(parts, dim=1)

    def cos_sin(freqs_cis):
        cos = freqs_cis[..., 0].unsqueeze(0).unsqueeze(2)
        sin = freqs_cis[..., 1].unsqueeze(0).unsqueeze(2)
        return cos, sin

    cap_axis0 = torch.arange(1, cap_len + 1)
    cap_axis12 = torch.zeros(cap_len, dtype=torch.long)
    img_axis0 = torch.full((num_img_tokens,), cap_len + 1, dtype=torch.long)
    row_ids = torch.arange(h_tok).unsqueeze(1).expand(h_tok, w_tok).reshape(-1)
    col_ids = torch.arange(w_tok).unsqueeze(0).expand(h_tok, w_tok).reshape(-1)

    img_cos, img_sin = cos_sin(build_freqs(img_axis0, row_ids, col_ids))
    cap_cos, cap_sin = cos_sin(build_freqs(cap_axis0, cap_axis12, cap_axis12))

    img = hidden_states[0]
    img = img.view(C, h_tok, patch, w_tok, patch).permute(1, 3, 2, 4, 0).reshape(num_img_tokens, patch * patch * C)
    img_tokens = model.all_x_embedder["2-1"](img).unsqueeze(0)
    cap_tokens = model.cap_embedder(encoder_hidden_states)

    def timestep_embedding(t, freq_dim=256, max_period=10000.0):
        half = freq_dim // 2
        freqs = torch.exp(-torch.log(torch.tensor(max_period)) * torch.arange(half, dtype=torch.float32) / half)
        args = t[:, None].float() * freqs[None]
        return torch.cat([torch.cos(args), torch.sin(args)], dim=-1)

    adaln_input = model.t_embedder.mlp(timestep_embedding(timestep * model.config.t_scale))

    def apply_rope(x, cos, sin):
        x_pairs = x.reshape(*x.shape[:-1], -1, 2)
        x_real, x_imag = x_pairs.unbind(-1)
        x_rot = torch.stack([-x_imag, x_real], dim=-1).flatten(-2)
        return x * cos + x_rot * sin

    def attention(x, attn, cos, sin):
        q, k, v = attn.to_q(x), attn.to_k(x), attn.to_v(x)
        B, S, _ = q.shape
        q, k, v = (t.view(B, S, n_heads, head_size) for t in (q, k, v))
        q, k = attn.norm_q(q), attn.norm_k(k)
        q, k = apply_rope(q, cos, sin), apply_rope(k, cos, sin)
        q, k, v = q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)
        out = torch.nn.functional.scaled_dot_product_attention(q, k, v)
        out = out.transpose(1, 2).reshape(B, S, dim)
        return attn.to_out[0](out)

    def feed_forward(x, ff):
        return ff.w2(torch.nn.functional.silu(ff.w1(x)) * ff.w3(x))

    def block(x, blk, cos, sin, modulation, adaln=None):
        if modulation:
            mod = blk.adaLN_modulation(adaln)
            scale_msa, gate_msa, scale_mlp, gate_mlp = mod.unsqueeze(1).chunk(4, dim=2)
            gate_msa, gate_mlp = gate_msa.tanh(), gate_mlp.tanh()
            scale_msa, scale_mlp = 1.0 + scale_msa, 1.0 + scale_mlp
            attn_in = blk.attention_norm1(x) * scale_msa
        else:
            attn_in = blk.attention_norm1(x)
        attn_out = blk.attention_norm2(attention(attn_in, blk.attention, cos, sin))
        if modulation:
            attn_out = attn_out * gate_msa
        x = x + attn_out
        ffn_in = blk.ffn_norm1(x) * scale_mlp if modulation else blk.ffn_norm1(x)
        ffn_out = blk.ffn_norm2(feed_forward(ffn_in, blk.feed_forward))
        if modulation:
            ffn_out = ffn_out * gate_mlp
        return x + ffn_out

    x = img_tokens
    for blk in model.noise_refiner:
        x = block(x, blk, img_cos, img_sin, True, adaln_input)
    img_tokens = x

    x = cap_tokens
    for blk in model.context_refiner:
        x = block(x, blk, cap_cos, cap_sin, False)
    cap_tokens = x

    unified = torch.cat([img_tokens, cap_tokens], dim=1)
    unified_cos = torch.cat([img_cos, cap_cos], dim=1)
    unified_sin = torch.cat([img_sin, cap_sin], dim=1)
    x = unified
    for blk in model.layers:
        x = block(x, blk, unified_cos, unified_sin, True, adaln_input)

    final = model.all_final_layer["2-1"]
    final_scale = (1.0 + final.adaLN_modulation(adaln_input)).unsqueeze(1)
    out = final.linear(final.norm_final(x) * final_scale)

    img_out = out[:, :num_img_tokens][0].view(h_tok, w_tok, patch, patch, C).permute(4, 0, 2, 1, 3).reshape(C, H, W)
    return img_out.unsqueeze(0)


def check(model, sess, H, W, cap_len):
    hidden_states = torch.randn(1, model.config.in_channels, H, W)
    encoder_hidden_states = torch.randn(1, cap_len, model.config.cap_feat_dim)
    timestep = torch.rand(1)
    with torch.no_grad():
        ref = reference_forward(model, hidden_states, encoder_hidden_states, timestep)
    ort_out = sess.run(
        ["sample"],
        {
            "hidden_states": hidden_states.numpy().astype(np.float32),
            "encoder_hidden_states": encoder_hidden_states.numpy().astype(np.float32),
            "timestep": timestep.numpy().astype(np.float32),
        },
    )[0]
    max_abs_diff = np.max(np.abs(ref.numpy() - ort_out))
    print(f"H={H} W={W} cap_len={cap_len} max_abs_diff={max_abs_diff:.6f}")
    assert max_abs_diff < 1e-2


model = ZImageTransformer2DModel.from_pretrained(MODEL_DIR, torch_dtype=torch.float32)
model.eval()
sess = ort.InferenceSession(ONNX_PATH, providers=["CPUExecutionProvider"])
check(model, sess, H=32, W=32, cap_len=32)
check(model, sess, H=64, W=64, cap_len=64)
check(model, sess, H=32, W=64, cap_len=32)  # non-square, proves dynamic-shape support
```

Expect `max_abs_diff` on the order of `1e-4` for an fp32 (`-p fp32`) export; adjust the
tolerance upward for `int4`/`int8` exports.

## Troubleshooting

- **`diffusers` import fails on a missing optional dependency it doesn't actually need**
  (e.g. `PIL`/Pillow, in a fully offline environment): `diffusers.utils.export_utils` and
  `diffusers.image_processor` import `PIL` unconditionally even though this exporter never
  touches image I/O. If you can't `pip install pillow`, a minimal stub package (empty
  `PIL/__init__.py` with `__version__ = "10.0.0"`, plus no-op `PIL/Image.py` /
  `PIL/ImageOps.py` / `PIL/ImageFilter.py` modules, placed earlier on `PYTHONPATH`) is
  enough to satisfy the import chain without installing real Pillow.
- **`OSError: ... is not a local folder and is not a valid model identifier`** when
  building: you likely pointed `-i` at the Z-Image-Turbo repo root instead of its
  `transformer/` subfolder, or `config.json` is missing `_class_name`.
- **A shape-mismatch or buffer-reuse error from ONNX Runtime** when running the exported
  graph: if you've modified `zimage.py`, see
  [ZIMAGE_DESIGN.md#implementation-gotcha-symbolic-dim-aliasing](ZIMAGE_DESIGN.md#implementation-gotcha-symbolic-dim-aliasing) —
  this is almost always a `seq_dim`/shape-labeling bug in a new or edited `_linear`/
  `make_multi_head_attention` call.
