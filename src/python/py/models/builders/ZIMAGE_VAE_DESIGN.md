# Design: Z-Image-Turbo VAE Decoder Exporter

`builders/zimage_vae.py` (`ZImageVAEDecoderModel`) exports the **decoder** of the
Z-Image-Turbo VAE (diffusers `AutoencoderKL`, a Flux-style VAE) to a standalone ONNX graph.
It can emit the graph in two flavours, selected by `--extra_options fuse_group_norm=...`:

- **decomposed** (`false`, default): standard ONNX ops only — for the current WebGPU EP, whose
  `GroupNorm`/`SkipGroupNorm` kernels are still being brought up;
- **fused** (`true`): **already in the WebGPU-optimized form** — the same structure that was
  previously produced by a post-export graph-surgery pass — so no rewrite step is needed.

See [How to Run](ZIMAGE_VAE_USAGE.md) for build/usage.

## Contents

- [Scope](#scope)
- [Why This Doesn't Fit the Standard `Model` Pipeline](#why-this-doesnt-fit-the-standard-model-pipeline)
- [Bridging the Config](#bridging-the-config)
- [Graph I/O](#graph-io)
- [Architecture Mirrored](#architecture-mirrored)
- [GroupNorm Flavour Switch (`fuse_group_norm`)](#groupnorm-flavour-switch-fuse_group_norm)
- [The Optimizations, Emitted Natively](#the-optimizations-emitted-natively)
  - [GroupNorm fusion + layout-transpose cancellation](#groupnorm-fusion--layout-transpose-cancellation)
  - [SkipGroupNorm at residual seams](#skipgroupnorm-at-residual-seams)
  - [NHWC Resize](#nhwc-resize)
  - [No Div-by-1.0](#no-div-by-10)
  - [Mid-block attention (unfused, float16)](#mid-block-attention-unfused-float16)
- [Runtime Dependency](#runtime-dependency)
- [Numerical Notes](#numerical-notes)
- [Verification](#verification)

## Scope

- **Decoder only.** No encoder. The graph consumes a raw latent and produces an RGB image
  (8× spatial upsample). The caller runs the diffusion loop / latent scaling itself.
- **Standalone ONNX graph.** Run it directly with a plain `onnxruntime.InferenceSession`; there
  is no onnxruntime-genai C++ runtime integration.
- **Dynamic batch and latent height/width** (matches the original export's dynamic dims).
- **Graph I/O, weights and compute all in `io_dtype`**: float16 for `-p fp16` on WebGPU,
  float32 for `-p fp32` / `use_webgpu_fp32=true`. This matches the Z-Image transformer export, so
  the pipeline runs one dtype end to end. (Numerically an fp16-I/O graph is identical to the
  earlier fp32-I/O + boundary-`Cast` form; only the two casts are gone.)

## Why This Doesn't Fit the Standard `Model` Pipeline

`builders/base.py`'s `Model.make_model` is a reflection loop that recognizes a causal-LM
topology (token embedding → N `*DecoderLayer`s → final norm → LM head). A convolutional VAE
decoder matches none of that. So `ZImageVAEDecoderModel` subclasses `Model` **only** to reuse
its low-level, architecture-agnostic builders — `make_conv`, `make_matmul`/`make_add_bias`
(which is what makes `-p int4/int8` work "for free" on the attention projections),
`make_multi_head_attention`, `make_initializer`, and the raw `make_node`/`make_value` IR
bookkeeping — and overrides `make_inputs_and_outputs`, `load_weights`, `make_model`, and
`make_genai_config` from scratch.

## Bridging the Config

`Model.__init__` reads transformers-style causal-LM attributes the diffusers `AutoencoderKL`
config doesn't have. `__init__` therefore builds a `types.SimpleNamespace` supplying just those
fields, calls `super().__init__(fake_config, ...)`, then sets the real VAE attributes. The
single-head mid-block attention is bridged by setting `hidden_size = block_out_channels[-1]`
and `num_attention_heads = 1`, which yields `head_size = block_out_channels[-1]` and attention
`scale = 1/sqrt(head_size)` — exactly diffusers' spatial self-attention scaling.

`builder.py::create_model` dispatches to this class on `config._class_name == "AutoencoderKL"`.
Because that config has `_class_name` but no `architectures`, `get_hf_details` already loads it
as a `SimpleNamespace` (no `AutoConfig`/tokenizer), the same path `zimage.py` relies on.

## Graph I/O

| Name                    | Shape                                                         | Dtype      | Notes                           |
| ----------------------- | ------------------------------------------------------------- | ---------- | ------------------------------- |
| `latent_sample` (input) | `[batch_size, latent_channels, latent_height, latent_width]`  | `io_dtype` | raw latent; batch + H/W dynamic |
| `sample` (output)       | `[batch_size, out_channels, latent_height*8, latent_width*8]` | `io_dtype` | decoded RGB                     |

`io_dtype` is float16 for `-p fp16` and float32 for `-p fp32`; there are no boundary casts.

## Architecture Mirrored

Standard diffusers decoder, node-for-node:

```
conv_in(latent_channels -> block_out_channels[-1])
mid_block:  resnet0 -> attention(heads=1) -> resnet1
up_blocks[i] (i = 0..3, channels reversed):
    layers_per_block+1 resnets  (first resnet has a conv_shortcut where channels change)
    upsampler (nearest 2x + conv)   [absent on the last up_block]
conv_norm_out (GroupNorm + SiLU) -> conv_out(-> out_channels)
```

ResnetBlock2D forward: `norm1→SiLU→conv1 → norm2→SiLU→conv2 → (conv_shortcut(x) or x) + h`.
Attention forward: `residual=h; group_norm; [1,C,HW]→[1,HW,C]; q/k/v; sdpa; to_out; back to
[1,C,H,W]; + residual`.

## GroupNorm Flavour Switch (`fuse_group_norm`)

All 30 GroupNorms go through one helper, `_group_norm`, which dispatches on
`self.fuse_group_norm` (default `False`). The top-level graph construction (`make_model`,
`_resnet`, the deferred-residual bookkeeping described below) is shared and unaware of the flag.

**Decomposed (`false`).** Each GroupNorm becomes the PyTorch exporter's standard pattern, all in
NCHW with no `Transpose`s:

```
Shape(x) ; Reshape(x, [0, G, -1]) -> InstanceNormalization(scale=ones[G], bias=zeros[G], eps)
-> Reshape(back to Shape(x)) -> Mul(gamma[C,1,1]) -> Add(beta[C,1,1]) [-> Sigmoid -> Mul]
```

`InstanceNormalization` over the `[N, G, C/G*H*W]` view computes exactly GroupNorm's statistics;
its per-group affine is the identity so the real `gamma`/`beta` apply per channel afterwards. A
deferred residual reaching a norm is materialized as a plain `Add` first (its output is also
returned as the residual base, standing in for `SkipGroupNorm`'s `sum` output). The upsampler
`Resize` runs directly in NCHW (`scales=[1,1,2,2]`) since there are no neighbouring Transposes
to cancel against. The result contains **no `com.microsoft` ops**.

Cost: the WebGPU EP still converts every `Conv` to NHWC and its layout Transposes cannot be pushed
through the 3-D `Reshape -> InstanceNormalization`, so ~70 of them survive (vs 7 in the fused
graph); measured ~15-20% slower per decode, with identical numerics (the two flavours agree to
`mean|d| ~ 1.5e-3` in float16; both are ~35 dB PSNR vs the float32 reference).

**Fused (`true`).** Everything in the next section.

## The Optimizations, Emitted Natively

Every optimization below applies to the fused flavour (`fuse_group_norm=true`). Each was
previously produced by a post-export rewrite pass and validated (numerics + WebGPU perf) on the
target hardware; here they are emitted directly from the PyTorch module structure, which the
builder knows exactly.

### GroupNorm fusion + layout-transpose cancellation

Each diffusers GroupNorm (naively exported as a decomposed
`Reshape→InstanceNormalization→Mul→Add[→Sigmoid→Mul]` cluster) is emitted as one channels-last
`com.microsoft.GroupNorm` with the SiLU/swish fused (`activation=1`), bracketed by NCHW↔NHWC
`Transpose`s. Because the WebGPU EP prefers NHWC `Conv`, the EP wraps each `Conv` in its own
layout `Transpose`s at session-load time; those cancel against the ones emitted here, leaving
`Conv`↔`GroupNorm` directly connected in NHWC with **zero runtime transposes**. `gamma`/`beta`
are the PyTorch `norm.weight`/`norm.bias` directly.

### SkipGroupNorm at residual seams

At every `residual-Add → next-norm` seam, the builder emits a single
`com.microsoft.SkipGroupNorm` (`S = X + skip`, `Y = GroupNorm(S)`), so the residual chain
carries no layout transposes. This is why each ResnetBlock's residual add is **deferred**: the
`_resnet` helper returns a `(conv2_out, residual_base)` pair rather than a materialized sum, and
the consumer either folds it into its `SkipGroupNorm` (a norm seam) or materializes it with an
`Add` (before an upsampler/attention). Seams that fold: every up-block resnet whose `norm1` is
fed by the previous resnet's residual (9 of them) plus `conv_norm_out` — 10 `SkipGroupNorm`
total. Explicit residual `Add`s remain only where the consumer is not a norm (mid resnet0 →
attention, and the last resnet before each of the 3 upsamplers). This reproduces the validated
layout exactly (`SkipGroupNorm=10`, `GroupNorm=20`).

The motivation: ORT's transpose optimizer will not push a layout transpose through an `Add`
that carries a transpose on only one input — precisely the residual-junction shape — so the
naive fusion leaves stranded transpose pairs there. `SkipGroupNorm` does the add inside the
NHWC op, sidestepping the heuristic.

### NHWC Resize

The 3 nearest-neighbour upsamplers run `Resize` natively in NHWC: the `scales` are permuted
`[1,1,2,2]→[1,2,2,1]` and the op is sandwiched in cancelling `Transpose`s, same as GroupNorm.
Resize attributes match the original export (`mode=nearest`, `nearest_mode=floor`,
`coordinate_transformation_mode=asymmetric`).

### No Div-by-1.0

diffusers' `output_scale_factor=1.0` residual division (a `Div` by 1.0, a numerical no-op) is
simply never emitted. A surviving `Div` at a residual seam would also block the transpose
cancellation the fusion relies on.

### Mid-block attention (unfused, float16)

The single mid-block self-attention is emitted **unfused** (`MatMul → scale → Softmax →
MatMul`), matching the original float16 export — **not** as `com.microsoft.MultiHeadAttention`.
This was a deliberate reversal after testing: this VAE's `to_q`/`to_k` outputs reach ~1e3, so
the attention logits `Q·Kᵀ / sqrt(head_size=512)` are ~1e6, far past float16's ~65504 max. In
float16 the logits are simply unrepresentable (→ ±Inf) regardless of softmax stability, which is
why diffusers ships this VAE with `force_upcast=true` (fp32 attention). The consequences:

- **Fused MHA** turns that Inf into **NaN**, which poisons the whole image (blank output).
- **True float32 attention** either exceeds the WebGPU workgroup-memory limit (the flash kernel
  needs 64 KB > the device's 32 KB for `head_size=512`) or materializes a >1 GB score matrix at
  1024×1024.
- **Unfused float16** (what the original/`_gn` models do) lets the explicit `Softmax(Inf)`
  collapse to a near-degenerate map — effectively a near-no-op, which is fine because VAE mid
  attention is a minor contributor. The decoded image is unaffected.

So the builder mirrors the validated baseline exactly: the exported graph is **bit-identical**
to `vae_decoder_model_f16_gn.onnx` (`max_abs_diff = 0`, uint8 pixel diff = 0). It is still leaner
(≈182 vs 243 nodes, `Shape=1` vs 7) because the reshapes use `-1` / a single `Shape` instead of
the naive dynamic shape subgraphs. If bit-accurate attention is ever required, compute this one
block in float32 (accepting the perf/memory cost above).

## Runtime Dependency

- **Decomposed (default):** standard ONNX ops only; needs an `InstanceNormalization` kernel for
  the chosen dtype (WebGPU EP: fp16/fp32; CPU EP: fp32 only).
- **Fused:** uses the contrib ops `com.microsoft.GroupNorm` and `com.microsoft.SkipGroupNorm` in
  **channels-last** form. It therefore requires an onnxruntime build whose **WebGPU EP**
  implements those kernels. The transpose-cancellation is a WebGPU-EP load-time behaviour; on
  CPU/CUDA the graph is still numerically correct where the kernels exist but the bracketing
  transposes won't cancel (so it isn't layout-optimal there).

The target is WebGPU in both cases.

## Numerical Notes

In both flavours GroupNorm `gamma`/`beta` come straight from the PyTorch
`norm.weight`/`norm.bias` (the decomposed flavour gives `InstanceNormalization` an identity
affine rather than folding anything into it), so nothing is re-rounded in float16. The float16
build's accuracy is bounded by float16 compute itself and reduction order; the fused flavour
matches the validated `vae_decoder_model_f16_gn.onnx`, and the decomposed flavour matches the
fused one to `mean|d| ~ 1.5e-3`.

## Verification

There is no onnxruntime-genai `Generator` integration to test through, so validate directly:

1. Export against the real checkpoint and load the result in a plain
   `onnxruntime.InferenceSession` (catches structural/shape bugs).
2. Compare against a diffusers `AutoencoderKL.decode` float32 reference at multiple latent
   resolutions (including a non-square one, to prove dynamic H/W) — see
   [How to Run](ZIMAGE_VAE_USAGE.md#verifying-numerical-correctness).
3. Op-type histogram. Fused: parity against the validated `vae_decoder_model_f16_gn.onnx`
   (`SkipGroupNorm=10`, `GroupNorm=20`, `Conv=35`, `Resize=3`, `MatMul=6`, `Softmax=1`, no
   `Div`); the exported graph is bit-identical to `_gn.onnx` (`max_abs_diff=0`) while using
   fewer nodes (`Shape=1` vs 7). Decomposed: `InstanceNormalization=30`, `Sigmoid=29`,
   `Transpose=3` (attention only), `Conv=35`, `Resize=3`, and **no `com.microsoft` ops**; on the
   WebGPU EP, dump the optimized model and confirm `InstanceNormalization` was not re-fused.
