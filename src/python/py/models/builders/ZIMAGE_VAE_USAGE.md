# How to Run: Z-Image-Turbo VAE Decoder Exporter

This exports the **VAE decoder** of Z-Image-Turbo (diffusers `AutoencoderKL`) as a standalone
ONNX graph for the WebGPU EP. Two GroupNorm flavours are available, selected by the
`fuse_group_norm` extra option (see [GroupNorm Flavour](#groupnorm-flavour-fuse_group_norm)):

- **decomposed** (default): standard ONNX ops only, runs on any onnxruntime with an
  `InstanceNormalization` kernel;
- **fused**: `com.microsoft.GroupNorm` / `SkipGroupNorm` contrib ops with NHWC Resize, for an
  onnxruntime whose WebGPU EP implements those kernels.

See [ZIMAGE_VAE_DESIGN.md](ZIMAGE_VAE_DESIGN.md) for the architecture and scope.

## Contents

- [Prerequisites](#prerequisites)
- [Getting the Checkpoint](#getting-the-checkpoint)
- [Building the ONNX Model](#building-the-onnx-model)
- [GroupNorm Flavour (`fuse_group_norm`)](#groupnorm-flavour-fuse_group_norm)
- [Output Files](#output-files)
- [Running the Exported Graph](#running-the-exported-graph)
- [Verifying Numerical Correctness](#verifying-numerical-correctness)
- [Runtime Dependency](#runtime-dependency)

## Prerequisites

In addition to this repo's normal model-builder dependencies (`torch`, `onnx_ir`,
`transformers`, `onnxruntime`), you need `diffusers` (it defines `AutoencoderKL`, which the
exporter loads the real weights through):

```bash
pip install diffusers pillow
```

## Getting the Checkpoint

```py
from huggingface_hub import snapshot_download
snapshot_download("Tongyi-MAI/Z-Image-Turbo", local_dir="path_to_local_folder")
```

The VAE weights live in the `vae/` subfolder (`vae/config.json`,
`vae/diffusion_pytorch_model.safetensors`) — point `-i` at that subfolder, not the repo root.

## Building the ONNX Model

```bash
# From source, from src/python/py/models:
python builder.py \
  -e webgpu -p fp16 \
  -i path_to_local_folder/vae \
  -o path_to_output_folder \
  -c cache_dir_to_store_temp_files
```

- `-p fp16` gives an all-float16 graph: I/O, weights and compute (same convention as the Z-Image
  transformer export). Feed `latent_sample` as float16 and expect a float16 `sample`.
- `-p fp32` (or `use_webgpu_fp32=true`) gives an all-float32 graph — useful for CPU numerical
  reference checks.
- `-p int4`/`-p int8` additionally quantize the 4 mid-block attention projections (a negligible
  size change for this model); they otherwise behave identically.
- `--extra_options fuse_group_norm=true|false` picks the GroupNorm flavour (default `false`), see
  the next section. Extra options are space-separated `key=value` pairs and can be combined, e.g.
  `--extra_options hf_remote=False fuse_group_norm=true`.

## GroupNorm Flavour (`fuse_group_norm`)

The WebGPU EP's `GroupNorm` / `SkipGroupNorm` contrib kernels are still being brought up, so the
exporter can emit each of the decoder's 30 GroupNorms in one of two ways:

| `fuse_group_norm`   | GroupNorm emitted as                                                                                                                    | Residual add                       | Upsample `Resize` | Contrib ops in graph                            |
| ------------------- | --------------------------------------------------------------------------------------------------------------------------------------- | ---------------------------------- | ----------------- | ----------------------------------------------- |
| `false` (default)   | `Reshape[0,G,-1] -> InstanceNormalization(scale=1,bias=0) -> Reshape -> Mul(gamma) -> Add(beta)` `[-> Sigmoid -> Mul]`, all in NCHW    | plain `Add` node                   | NCHW              | **none**                                        |
| `true`              | channels-last `com.microsoft.GroupNorm` (`activation=1` folds the SiLU), bracketed by NCHW<->NHWC Transposes                             | folded into `SkipGroupNorm`        | NHWC              | `com.microsoft.GroupNorm`, `SkipGroupNorm`      |

Everything else (Conv, the unfused mid-block attention, dynamic batch/H/W, I/O dtype) is
identical between the two, and the GroupNorm gamma/beta are the PyTorch
`norm.weight` / `norm.bias` in both.

```bash
# Decomposed (default) — no contrib ops, works on today's WebGPU EP:
python builder.py -e webgpu -p fp16 \
  -i path_to_local_folder/vae -o out/vae_decoder_unfused -c cache_dir

# Fused — needs an onnxruntime whose WebGPU EP implements GroupNorm/SkipGroupNorm:
python builder.py -e webgpu -p fp16 \
  --extra_options fuse_group_norm=true \
  -i path_to_local_folder/vae -o out/vae_decoder_fused -c cache_dir
```

Both variants save to the same `model.onnx` filename, so use different `-o` folders.

## Output Files

| File                | Contents                                                                                                                                                                |
| ------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `model.onnx`        | the VAE decoder graph — a **single self-contained file** (weights inline; ~50-100 MB, under ONNX's 2 GB limit), matching the original export. No separate `.onnx.data`. |
| `genai_config.json` | minimal metadata (model type, channel dims, I/O names) — not consumed by the genai generate loop; this is a standalone graph                                            |

## Running the Exported Graph

```py
import numpy as np
import onnxruntime as ort

sess = ort.InferenceSession("path_to_output_folder/model.onnx", providers=["WebGpuExecutionProvider", "CPUExecutionProvider"])

# I/O dtype follows the build precision: float16 for `-p fp16`, float32 for `-p fp32`.
io_dtype = np.float16 if sess.get_inputs()[0].type == "tensor(float16)" else np.float32

# latent_sample: [1, latent_channels, H, W]
sample = sess.run(["sample"], {"latent_sample": latent_np.astype(io_dtype)})[0]
# sample: [1, out_channels, H*8, W*8], same dtype
```

## Verifying Numerical Correctness

Compare the exported graph against a diffusers `AutoencoderKL.decode` float32 reference at
multiple latent resolutions (including a non-square one, to prove dynamic H/W). Build with
`-p fp32` for the tightest tolerance.

```py
import numpy as np
import torch
import onnxruntime as ort
from diffusers import AutoencoderKL

VAE_DIR = "path_to_local_folder/vae"
ONNX_PATH = "path_to_output_folder/model.onnx"

vae = AutoencoderKL.from_pretrained(VAE_DIR, torch_dtype=torch.float32).eval()
sess = ort.InferenceSession(ONNX_PATH, providers=["CPUExecutionProvider"])
C = vae.config.latent_channels


def check(h, w):
    latent = torch.randn(1, C, h, w)
    with torch.no_grad():
        ref = vae.decode(latent).sample.numpy()
    out = sess.run(["sample"], {"latent_sample": latent.numpy().astype(np.float32)})[0]  # fp32 build
    print(f"h={h} w={w} max_abs_diff={np.max(np.abs(ref - out)):.6f}")


check(32, 32)
check(64, 64)
check(32, 48)  # non-square, proves dynamic H/W
```

Expect `max_abs_diff` on the order of `1e-6`–`1e-5` for a decomposed (`fuse_group_norm=false`)
fp32 export on the CPU EP (a plain `Add` residual vs diffusers' `/1.0` is exact; the residue is
reduction order). The CPU EP has no `GroupNorm`/`SkipGroupNorm` kernels, so the fused flavour and
any fp16 export must be validated on the target WebGPU machine: run the same script with
`providers=["WebGpuExecutionProvider", "CPUExecutionProvider"]`, and additionally set
`SessionOptions.optimized_model_filepath` and inspect the dumped graph's op types to confirm what
actually ran (e.g. that `InstanceNormalization` survived unfused). Comparing the two flavours
against each other on the same machine is a good way to isolate flavour-specific issues from
float16 error.

## Runtime Dependency

- `fuse_group_norm=false` (default): standard ONNX ops only. Runs on any EP with an
  `InstanceNormalization` kernel for the chosen dtype (WebGPU EP: fp16 and fp32; CPU EP: fp32
  only).
- `fuse_group_norm=true`: uses channels-last `com.microsoft.GroupNorm` / `SkipGroupNorm`, so it
  needs an onnxruntime build whose **WebGPU EP** implements those contrib kernels. On CPU/CUDA the
  graph is numerically correct where the kernels exist but not layout-optimal (the bracketing
  transposes only cancel on the WebGPU EP at load time).

See [ZIMAGE_VAE_DESIGN.md#runtime-dependency](ZIMAGE_VAE_DESIGN.md#runtime-dependency).
