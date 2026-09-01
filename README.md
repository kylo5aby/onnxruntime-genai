# Z-Image-Turbo Transformer Trunk Exporter (onnxruntime-genai dev branch)

This branch of `onnxruntime-genai` adds a standalone ONNX exporter for the diffusion
transformer trunk of [Z-Image-Turbo](https://huggingface.co/Tongyi-MAI/Z-Image-Turbo)
(diffusers class `ZImageTransformer2DModel`). It is not an onnxruntime-genai runtime
integration — the exported graph consumes pre-computed caption embeddings and a raw image
latent, and produces a denoised/velocity-predicted latent, run directly via a plain
`onnxruntime.InferenceSession`. A caller drives the diffusion sampling loop, text encoding,
and VAE decode itself.

All of the exporter code lives under `src/python/py/models/`:

| File | Contents |
|---|---|
| [`src/python/py/models/builders/zimage.py`](src/python/py/models/builders/zimage.py) | `ZImageTransformerModel`, the exporter itself |
| [`src/python/py/models/build_z_image_turbo.py`](src/python/py/models/build_z_image_turbo.py) | CLI wrapper for building all precision variants |
| [`src/python/py/models/run_z_image_turbo.py`](src/python/py/models/run_z_image_turbo.py) | Standalone end-to-end text-to-image pipeline driver that can run the exported transformer |
| [`src/python/py/models/builders/ZIMAGE_DESIGN.md`](src/python/py/models/builders/ZIMAGE_DESIGN.md) | Architecture, scope, and design rationale |
| [`src/python/py/models/builders/ZIMAGE_USAGE.md`](src/python/py/models/builders/ZIMAGE_USAGE.md) | Full build/run/verify walkthrough, using `builder.py` directly |
| [`src/python/py/models/DESIGN.md`](src/python/py/models/DESIGN.md) | Design of the general model-builder pipeline this exporter reuses pieces of |

## Scope

- **Transformer trunk only.** No Qwen3 text encoder, no VAE decoder.
- **Standalone ONNX graph.** No onnxruntime-genai C++ generator runtime integration.
- **Dynamic height/width**, batch size fixed at 1.
- **No padding/pad-token machinery** — resolutions and caption lengths must already satisfy
  the precondition below.

See [ZIMAGE_DESIGN.md#scope](src/python/py/models/builders/ZIMAGE_DESIGN.md#scope) for the
full list of what's out of scope (SigLIP/Omni conditioning, LoRA, ControlNet, multiple patch
sizes, gradient checkpointing).

## Quick Start: Building the Model

```bash
cd src/python/py/models
pip install diffusers pillow  # in addition to this repo's normal torch/onnx_ir/transformers/onnxruntime deps
```

Download the checkpoint:

```py
from huggingface_hub import snapshot_download
snapshot_download("Tongyi-MAI/Z-Image-Turbo", local_dir="path_to_local_folder")
```

Build with [`build_z_image_turbo.py`](src/python/py/models/build_z_image_turbo.py), which
wraps `builder.py` with the WebGPU EP and the extra options this model needs pre-filled in.
Pick one of four precisions with `-p`:

```bash
python build_z_image_turbo.py path_to_local_folder -p <precision>
```

| `-p` value | `builder.py -p` | I/O dtype | Weights | Notes |
|---|---|---|---|---|
| `f16` | `fp16` | float16 | unquantized | |
| `f32` | `fp32` | float32 | unquantized | |
| `f16_int4_quant` (default) | `int4` | float16 | int4 (`MatMulNBits`) | `block_size=32 accuracy_level=4` |
| `f32_int4_quant` | `int4` | float32 | int4 (`MatMulNBits`) | same as above + `use_webgpu_fp32=true` |

Output goes to `<model_name>-transformer-genai-wgpu-<precision>/`. All four variants are
verified to produce correct (non-NaN) output — see
[ZIMAGE_DESIGN.md#float16-dynamic-range-overflow](src/python/py/models/builders/ZIMAGE_DESIGN.md#float16-dynamic-range-overflow)
for the float16 NaN issue this required fixing.

For finer control (different EP, `int8`, custom `--extra_options`, etc.), call `builder.py`
directly — see
[ZIMAGE_USAGE.md#building-the-onnx-model](src/python/py/models/builders/ZIMAGE_USAGE.md#building-the-onnx-model).

## Precondition on Resolution and Caption Length

Because the exported graph has no padding/masking logic, the caller must ensure:

- `(height / patch_size) * (width / patch_size) % 32 == 0` (`patch_size = 2`) — holds for
  every standard resolution divisible by 16 (512, 768, 1024, ...), including non-square
  combinations.
- The caption embedding's sequence length is already a multiple of 32 tokens.

Violating either precondition does not error at export time — it silently computes the
wrong thing at inference time. See
[ZIMAGE_USAGE.md#precondition-on-resolution-and-caption-length](src/python/py/models/builders/ZIMAGE_USAGE.md#precondition-on-resolution-and-caption-length).

## Running the Exported Graph

```py
import onnxruntime as ort

sess = ort.InferenceSession("path_to_output_folder/model.onnx", providers=["CPUExecutionProvider"])
sample = sess.run(
    ["sample"],
    {
        "hidden_states": hidden_states_np,          # [1, 16, H, W]
        "encoder_hidden_states": encoder_hidden_states_np,  # [1, cap_len, 2560]
        "timestep": timestep_np,                    # [1]
    },
)[0]
```

Full details, including a verification script that cross-checks the ONNX output against a
hand-written PyTorch reference, are in
[ZIMAGE_USAGE.md#running-the-exported-graph](src/python/py/models/builders/ZIMAGE_USAGE.md#running-the-exported-graph)
and
[ZIMAGE_USAGE.md#verifying-numerical-correctness](src/python/py/models/builders/ZIMAGE_USAGE.md#verifying-numerical-correctness).

## Running the Full Pipeline (Text-to-Image)

[`run_z_image_turbo.py`](src/python/py/models/run_z_image_turbo.py) is a standalone,
self-contained script that drives an actual end-to-end Z-Image-Turbo text-to-image
generation (tokenizer -> text encoder -> flow-matching denoising loop -> VAE decode -> PNG),
useful for exercising the exported transformer against real prompts instead of the synthetic
tensors used for verification.

It expects a WebNN-exported Z-Image-Turbo model directory (with `tokenizer/`,
`onnx/text_encoder_model_q4f16.onnx`, and `onnx/vae_decoder_model_f16.onnx`) for the
tokenizer/text-encoder/VAE — this script does not export those; only the transformer is
in scope for this repo. Pass `--transformer` to swap in the transformer exported by
`build_z_image_turbo.py` above in place of that bundle's own transformer:

```bash
cd src/python/py/models
pip install psutil transformers pillow torch onnxruntime  # in addition to the deps above

python run_z_image_turbo.py path_to_webnn_z_image_turbo_dir \
  --transformer path_to_output_folder/model.onnx \
  --prompt "a cat under the snow with blue eyes, cinematic style" \
  --height 512 --width 512 \
  -n 4 -o output.png
```

`--transformer` accounts for this dev exporter's differences from the bundled WebNN
transformer automatically: 4D `hidden_states` (no `num_frames` axis), no attention
mask/padding (`encoder_hidden_states` is padded to a multiple of 32 tokens by repeating the
last real token's embedding), and whichever I/O dtype (`float16`/`float32`) the chosen `-p`
build used. Text encoder and VAE decoder are always the WebNN bundle's own models,
regardless of `--transformer`. Run without `--transformer` to use the WebNN bundle
end-to-end as a baseline for comparison. See `--help` for `--ep` (WebGPU/CPU),
`--all_images` (dump every denoising step), `-l/--loop` (repeat generation), and `-v`
(verbose per-tensor stats) options.

## Further Reading

- [ZIMAGE_DESIGN.md](src/python/py/models/builders/ZIMAGE_DESIGN.md) — why this doesn't fit
  the standard `Model` pipeline, RoPE/patchify/AdaLN graph construction, quantization
  coverage, the float16 overflow fix, and a symbolic-dim-aliasing implementation gotcha.
- [ZIMAGE_USAGE.md](src/python/py/models/builders/ZIMAGE_USAGE.md) — build/run/verify
  walkthrough and troubleshooting.
- [DESIGN.md](src/python/py/models/DESIGN.md) — the general model-builder pipeline
  (`Model`, `make_matmul`, quantization pass) this exporter reuses low-level pieces of.
