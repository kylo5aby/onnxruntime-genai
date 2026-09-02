# Design: Z-Image-Turbo Transformer Trunk Exporter

`builders/zimage.py` (`ZImageTransformerModel`) exports the transformer trunk of
[Z-Image-Turbo](https://huggingface.co/Tongyi-MAI/Z-Image-Turbo) (diffusers class
`ZImageTransformer2DModel`) to a standalone ONNX graph. See [How to Run](ZIMAGE_USAGE.md)
for build/usage instructions.

## Contents

- [Scope](#scope)
- [Why This Doesn't Fit the Standard `Model` Pipeline](#why-this-doesnt-fit-the-standard-model-pipeline)
- [Bridging the Config](#bridging-the-config)
- [Graph I/O](#graph-io)
- [Graph Construction](#graph-construction)
  - [RoPE: Precomputed Tables + Dynamic Position Grids](#rope-precomputed-tables--dynamic-position-grids)
  - [Patchify / Unpatchify](#patchify--unpatchify)
  - [Timestep Embedding and AdaLN Modulation](#timestep-embedding-and-adaln-modulation)
  - [Transformer Blocks](#transformer-blocks)
- [Quantization](#quantization)
- [float16 Dynamic-Range Overflow](#float16-dynamic-range-overflow)
- [Implementation Gotcha: Symbolic Dim Aliasing](#implementation-gotcha-symbolic-dim-aliasing)
- [Verification](#verification)

## Scope

This is intentionally a narrow slice of the full Z-Image-Turbo pipeline:

- **Transformer trunk only.** No Qwen3 text encoder, no VAE decoder. The exported graph
  consumes pre-computed caption embeddings and a raw image latent, and produces a
  denoised/velocity-predicted latent. A caller drives the diffusion sampling loop
  (scheduler, text encoding, VAE decode) itself.
- **Standalone ONNX graph.** No changes to onnxruntime-genai's C++ generator runtime; the
  graph is meant to be run directly via a plain `onnxruntime.InferenceSession`.
- **Dynamic height/width.** A single exported graph works for any resolution (512x512,
  1024x1024, non-square, etc.) — `height`/`width` are dynamic ONNX dims, not baked in at
  export time.
- **Batch size fixed at 1.**
- **No padding/pad-token machinery.** The original model pads image/caption token counts
  up to a multiple of 32 and fills the gaps with learned `x_pad_token`/`cap_pad_token`
  vectors — this exists only to pack variable-length samples into one batch. Since batch=1,
  this is dropped entirely: the caller must only use resolutions/caption lengths where
  `(H/patch_size) * (W/patch_size) % 32 == 0` (true for all standard resolutions divisible
  by 16) and a caption embedding pre-padded to a multiple of 32 tokens. There is no
  attention mask and no pad-token injection in the exported graph.
- Not supported: SigLIP/Omni conditioning, LoRA, ControlNet, multiple patch sizes (only the
  single `2-1` patch/f-patch size Z-Image-Turbo ships), gradient checkpointing.

## Why This Doesn't Fit the Standard `Model` Pipeline

`builders/base.py`'s `Model` class assumes a causal-LM decoder: token-id input, KV cache,
autoregressive generation, and a `make_model` reflection loop that recognizes an
`Embedding`, N repeats of a module named `*DecoderLayer`, a final norm, and an LM head.
`ZImageTransformer2DModel` matches none of this:

- Three named transformer stacks run in sequence over *different* token sets
  (`noise_refiner` over image tokens, `context_refiner` over caption tokens, `layers` over
  the unified sequence) instead of one repeated decoder layer.
- Attention is fully bidirectional with no KV cache anywhere.
- Blocks are conditioned by AdaLN modulation (a timestep embedding gates/scales
  activations), which has no equivalent in `Model`.
- RoPE is 3-axis, real-valued, and built by gathering from precomputed per-axis tables at
  dynamic (row, column, sequence) position ids — not the single-axis or `MRotaryEmbedding`
  integer-position schemes `Model` already supports.

Given that mismatch, `ZImageTransformerModel` subclasses `Model` **only** to reuse its
low-level, architecture-agnostic pieces — `make_matmul` (which is what makes int4/int8
quantization work "for free"), `make_initializer`, the raw `make_node`/`make_value`/IR
graph bookkeeping, and `make_multi_head_attention` (which is RoPE-agnostic and already
defaults to bidirectional attention with no KV cache, the same precedent used by
`WhisperEncoder` in `builders/whisper.py`). Everything else — `make_inputs_and_outputs`,
`load_weights`, `make_model` (the whole graph), and `make_genai_config` — is overridden
from scratch. This mirrors the existing `builders/dflash2.py` precedent of a model whose
shape doesn't fit the causal-LM reflection loop hand-rolling its own top-level
construction.

## Bridging the Config

`Model.__init__` unconditionally reads `transformers`-style causal-LM config attributes
(`hidden_size`, `num_attention_heads`, `vocab_size`, `architectures`, `_name_or_path`,
`hidden_act`, `max_position_embeddings`, ...). The Z-Image-Turbo diffusers config has none
of these (`dim`, `n_heads`, `_class_name`, no vocab/context-length concept at all), so
`ZImageTransformerModel.__init__` first builds a `types.SimpleNamespace` translating the
handful of fields `Model.__init__` needs, calls `super().__init__(fake_config, ...)`, then
sets the real Z-Image-specific attributes (`self.dim`, `self.axes_dims`, etc.) directly.
Everything LLM-specific that `Model.__init__` sets up as a side effect (`mask_attrs`,
`rope_attrs`, `kv_cache_attrs`, `mlp_attrs`, `moe_attrs`, the standard input/output name
dicts) is simply unused.

Because the diffusers config also has no `architectures` field (it's identified by
`"_class_name": "ZImageTransformer2DModel"`) and there's no tokenizer to load, `builder.py`
special-cases config loading in `get_hf_details`: if `config.json` has `_class_name` but no
`architectures`, it's loaded as a plain JSON dict wrapped in a `SimpleNamespace` instead of
going through `AutoConfig`/`AutoTokenizer`, with `_name_or_path` stamped onto it (matching
what `AutoConfig.from_pretrained` normally does) so weight loading later resolves to the
correct local path instead of trying to hit the HF Hub.

## Graph I/O

| Name | Shape | Dtype | Notes |
|---|---|---|---|
| `hidden_states` (input) | `[1, 16, height, width]` | io_dtype | raw latent, `height`/`width` dynamic |
| `encoder_hidden_states` (input) | `[1, cap_seq_len, 2560]` | io_dtype | pre-computed caption embedding, pre-padded to a multiple of 32 tokens |
| `timestep` (input) | `[1]` | io_dtype | diffusion timestep |
| `sample` (output) | `[1, 16, height, width]` | io_dtype | predicted noise/velocity, same shape as `hidden_states` |

## Graph Construction

### RoPE: Precomputed Tables + Dynamic Position Grids

`axes_dims`/`axes_lens`/`rope_theta` are static config values, so the 3 per-axis
`(cos, sin)` frequency tables (`_make_rope_tables`) are precomputed once in Python with
`torch` and stored as constant initializers. Each table is `[L, d/2, 2]` — one `(cos, sin)`
value per rotated *pair*, avoiding `torch.polar`/complex dtypes that ONNX can't represent.
(There is deliberately no `repeat_interleave(2)` to `d`-wide cos/sin: the fused
`RotaryEmbedding` op below handles the pairing internally, so the caches stay half-width.)

What *is* dynamic is the position-id lookup into those tables: caption tokens get
`(pos=1..cap_len, 0, 0)` and image tokens get `(pos=cap_len+1 [constant], row=0..H_t-1,
col=0..W_t-1)`, matching `ZImageTransformer2DModel.create_coordinate_grid`. Since `cap_len`
and the image token grid (`H_t = height/patch_size`, `W_t = width/patch_size`) are only
known at runtime, `_build_position_grids` builds these ids with `Shape`/`Range`/`Expand`/
`Reshape` ops rather than baking them in — this is what makes the dynamic-height/width
requirement work with a single exported graph. `_gather_axis_freqs`/`_build_freqs_cis` then
gather the 3 axes and concat them into a per-token `[tokens, head_size/2, 2]` tensor, and
`_cos_sin_caches_from_freqs_cis` splits that into the 2D `cos_cache`/`sin_cache`
`[tokens, head_size/2]` operands.

`_apply_rope` applies the rotation with a **single `com.microsoft.RotaryEmbedding` contrib
op** (`interleaved=1`, `num_heads=n_heads`), not a hand-rolled decomposition. It reshapes
the per-head Q/K `[1, tokens, heads, head_size]` down to the op's 3D `[1, tokens, dim]`
layout, feeds the per-token caches plus identity `position_ids` (`Range(0, seq)` — the
caches are already ordered per token, so each token indexes its own row), and returns the
rotated `[1, tokens, dim]` that MHA consumes directly. This replaces the earlier 8-op
`Reshape → Gather×2 → Neg → Concat → Reshape → Mul×2 → Add` pattern (68 of them — one per
Q/K across the 34 attention blocks) with one fused kernel each. All the multi-axis,
real-valued complexity stays in the cache-building above; by the time the op runs it sees a
finished per-token cache, exactly what the fused kernel expects. `com.microsoft.RotaryEmbedding`
is implemented in ORT-Web/JSEP, so it runs on the WebGPU deploy target (this is the same op
the Olive `convert_z_image.py` reference fuses via `OrtTransformersOptimization`; here it is
emitted directly, because that post-export pass only rewrites existing `RotaryEmbedding`
*function* nodes, never a raw primitive decomposition).

### Patchify / Unpatchify

`hidden_states` `[1, 16, H, W]` is reshaped/transposed into `[H_t*W_t, 64]` patch vectors
(`Reshape` to `[16, H_t, 2, W_t, 2]`, `Transpose` with `perm=[1,3,2,4,0]` to
`[H_t, W_t, 2, 2, 16]`, `Reshape` to `[H_t*W_t, 64]`) before the `x_embedder` Linear —
mirroring `_patchify_image`'s `view`/`permute`/`reshape` exactly, with `H_t`/`W_t` computed
dynamically via `Shape`/`Div`. Unpatchify at the end is the exact inverse
(`perm=[4,0,2,1,3]`).

### Timestep Embedding and AdaLN Modulation

The sinusoidal timestep frequency vector (`exp(-log(10000)*arange(128)/128)`) is a pure
function of config, so it's also a precomputed constant; only the `timestep * freq`
outer-product, `Cos`/`Sin`, and the two-Linear MLP are built as graph ops
(`_make_timestep_embedding`). AdaLN modulation (`_make_adaln`) is one shared helper used by
every modulated block: `Linear(256 -> 4*dim)` -> `Split` into 4 -> `Tanh` on the two gates
-> `1 + scale` on the two scales -> unsqueeze to `[1, 1, dim]` so it broadcasts over the
sequence axis when multiplied into the block's activations.

### Transformer Blocks

`_make_block` implements `ZImageTransformerBlock.forward` node-for-node: RMSNorm ->
(optional AdaLN scale) -> attention -> RMSNorm -> (optional AdaLN gate) -> residual add;
same pattern for the SwiGLU feed-forward (`w2(silu(w1(x)) * w3(x))`). `noise_refiner` (2
layers) and `layers` (30 layers) use modulation; `context_refiner` (2 layers) does not,
matching the reference model's `modulation=True/False` construction. `_make_attention`
does Q/K/V projections, reshapes Q/K to `[1, tokens, heads, head_size]`, applies per-head
RMSNorm (`qk_norm=true` in this model's config), then the fused `RotaryEmbedding` op (which
returns the flattened `[1, tokens, dim]` directly), and calls the inherited
`make_multi_head_attention` with no mask and no past/present KV inputs.

## Quantization

Every `Linear` goes through the inherited `make_matmul`, so `-p int4`/`int8` apply
automatically via the same mechanism every other model in this builder uses (a generic,
op-type-driven pass at save time — see `DESIGN.md`). The big sequence projections -- every
attention `to_q`/`to_k`/`to_v`/`to_out` and every feed-forward `w1`/`w2`/`w3` across all
34 blocks (238 `MatMul`s), plus the `final_linear` output projection -- are quantized to
`MatMulNBits`, for **239** quantized nodes in `f16_int4_quant`.

The small conditioning / embedding `Linear`s are deliberately **excluded** from int4
quantization by passing `exclude_from_quantization=True` to `_linear` (which sets the flag
`make_matmul` reads to add the node to `nodes_to_exclude`; the save-time `MatMulNBits` pass
then leaves it as a float `MatMul` at `io_dtype`). Int4 rounding of these low-rank
conditioning / input signals visibly corrupts the generated image ("garbled" output). The
excluded set is 37 layers:

- `x_embedder` (patch/input projection -- an input embedding, precision-sensitive like a
  `Gather` embedding table),
- `t_embedder/mlp0` + `mlp2` (timestep-embedding MLP, operates on a `[1, dim]` vector),
- every `adaLN_modulation/Linear` (2 `noise_refiner` + 30 `layers` = 32; `context_refiner`
  is unmodulated so has none) and `final_adaLN/Linear0`,
- `cap_embedder_linear`.

This matches the reference Olive export (`convert_z_image.py`), which quantizes exactly the
same set (239 `MatMulNBits`) and leaves these 37 conditioning/embedding layers as float
`Gemm`. (`final_linear` -- the reference's `/2-1/linear` -- **is** quantized in both; the
reference simply patch-embeds outside its exported trunk, so it has no in-graph `x_embedder`
to quantize, whereas this exporter includes and excludes it.)

`op_types_to_quantize=MatMul` (see `build_z_image_turbo.py`) targets only `MatMul`; `Gather`
is deliberately *not* quantized. The *only* `Gather`s in this graph with a constant
initializer as their data input are the 3 precomputed RoPE frequency tables from
`_make_rope_tables` (each gathered once for image positions, once for caption positions --
6 `Gather` nodes total). Those are real-valued cos/sin rotation values, not an embedding
matrix, so there's no meaningful size win from quantizing them, and `GatherBlockQuantized`'s
int4 rounding would directly degrade every RoPE-rotated Q/K value -- so `Gather` is left out
of `op_types_to_quantize`. Every other `Gather` in the graph (`height`/`width`/`cap_seq_len`
shape extraction, and the `cos`/`sin` cache split from `freqs_cis`) reads from a
runtime-computed tensor, not a weight, so none of them were ever quantization candidates in
the first place. (The old hand-rolled RoPE decomposition added two more `Gather`s per Q/K
for real/imag pair extraction; the fused `RotaryEmbedding` op removed all of those.)

## float16 Dynamic-Range Overflow

`f16`/`f16_int4_quant` (float16 WebGPU I/O) originally produced NaN output. Root cause,
found by bisecting node-by-node on a plain `CPUExecutionProvider` session (the bug
reproduces without any WebGPU hardware) and cross-checking against a real
`ZImageTransformer2DModel.forward` run in float32 PyTorch with the same inputs:

- **Not a quantization bug.** The unquantized `f16` build hits the exact same NaN with the
  same input; `f32_int4_quant` (int4 weights, float32 I/O) does not. So the failure tracks
  `io_dtype`, not `-p int4`. Forcing `accuracy_level=1` (plain fp32 MatMulNBits compute
  instead of the default int8-dynamic-activation-quantization `accuracy_level=4` path) does
  not fix it either -- ruling out the MatMulNBits compute-kernel selection as the cause.
- **The real cause: every modulated block's raw (pre-`SimplifiedLayerNormalization`)
  `attention.to_out` and `feed_forward.w2` output genuinely exceeds float16's ~65504 max on
  ordinary inputs.** With a standard-normal `hidden_states` latent (exactly what a real
  diffusion sampler feeds in at its first denoising step) and the real checkpoint weights,
  a 90-trial sweep (30 seeds x 3 timesteps, in float32 PyTorch, mirroring `zimage.py`
  node-for-node) measured this raw pre-norm magnitude peaking anywhere from ~3.5e5 to
  ~9.0e5 -- 5x-14x over float16's max, and *typically* (median trial) already ~6.6e5. This
  is not a rare tail case; it is the common case for real inputs. `float32` never notices
  because it has ~10^38 of headroom; `float16` overflows to `Inf` at exactly this point,
  and the following RMSNorm turns `Inf` into `NaN` (`Inf / sqrt(mean(Inf^2))`), poisoning
  everything downstream. Unmodulated blocks (`context_refiner`) never hit this: without
  AdaLN's `1 + tanh(...)`-scaled `attention_norm1`/`ffn_norm1` output feeding in, their
  pre-norm magnitude stays in the low hundreds.
- **The fix does not require float32 precision anywhere** (which would give up most of
  float16's throughput advantage on WebGPU). `SimplifiedLayerNormalization`
  (`attention_norm2`/`ffn_norm2`) is exactly scale-invariant to a positive scalar multiple
  of its input: `RMSNorm(c*x) == RMSNorm(x)` for any `c > 0`, since the weight
  multiplication happens after the input is divided by its own RMS (`norm_eps` is
  `1e-5` here, and scaling *down* only makes `eps`'s already-negligible contribution to
  that division smaller still -- see the derivation below). Since `to_out`/`w2` are plain
  linear projections, scaling their *input* down by a constant `c` scales their *output* by
  exactly the same `c` (`(x/c) @ W == (x @ W)/c`) -- so rescaling right before those two
  matmuls (`_rescale_preout`, used in `_make_attention`/`_make_feed_forward`) keeps every
  intermediate tensor representable in float16 while leaving the eventual normalized output
  bit-for-bit unaffected by the constant chosen (mathematically; float16 rounding aside).
  The constant is `c = 1/128`, matching the validated post-export `apply_scaledown_d_fix_all`
  patch (all factors are exact powers of two in float16). `attention.to_out` takes the full
  `1/128` on its single (MHA-output) input. `feed_forward.w2`'s input is the SwiGLU product
  `SiLU(w1) * w3`, so the `1/128` is *split* across the two operands -- `1/8` on the SiLU
  gate, `1/16` on `w3` (`8 * 16 == 128`) -- and applied *before* the elementwise multiply, so
  neither the product itself nor the `w2` matmul that consumes it can overflow, not just the
  matmul output. `1/128` leaves the worst measured pre-scale peak (~9.0e5 / 128 ~= 7000, vs.
  float16's 65504 max) about 9x of headroom for inputs outside the 90-trial sweep.

  Derivation that `eps` stays negligible after rescaling: for
  `y = x / sqrt(mean(x^2) + eps) * scale`, substituting `x' = x/c` gives
  `y' = x / sqrt(mean(x^2) + eps*c^2) * scale` -- i.e. rescaling by `c < 1` *shrinks*
  epsilon's effective contribution (by `c^2`), it does not grow it, so `y' == y` to within
  float precision.
- This is an empirically-derived safety margin, not a proven bound. The overflow is a
  per-token phenomenon (`SimplifiedLayerNormalization`/attention operate per-token), so
  resolution/caption-length shouldn't materially change the per-token peak measured here,
  but this has only been validated at 32x32/64x64 resolutions -- if some real input is ever
  found to still overflow past this margin, the next escape hatch is computing `to_out`/
  `w2` (and the norm immediately after) in float32 regardless of `io_dtype`, at the cost of
  losing float16 throughput for just those two matmuls per block.

## Implementation Gotcha: Symbolic Dim Aliasing

This model has three logically distinct, differently-sized sequences alive in the graph at
once (image tokens, caption tokens, the unified sequence) plus several genuinely-2D
tensors (the AdaLN/timestep MLPs). The inherited `make_matmul`/`make_add_bias`/
`make_multi_head_attention` all default to declaring their output shape with the *same*
shared symbolic dim name `"sequence_length"`. If every branch used that default, ONNX
Runtime's memory planner would treat differently-sized intermediates (e.g. 256 image
tokens vs. 32 caption tokens) as alias-compatible buffers and crash at inference time with
a shape-mismatch error — this actually happened during development. The fix:
`_linear` derives a `seq_dim` unique per logical branch (`"img_seq_len"` /
`"cap_seq_len"` / `"unified_seq_len"`, or a name-derived unique label for 2D tensors) and
threads it through explicitly; `_make_attention` re-stamps `make_multi_head_attention`'s
output shape immediately after calling it for the same reason, since that method has no
`seq_dim` parameter at all. Any new code added to this file that calls `make_matmul`/
`make_add_bias`/`make_multi_head_attention` needs to keep this in mind.

## Verification

There's no runtime integration to test through onnxruntime-genai's `Generator`, so
correctness was validated directly:

1. Export against the real `Tongyi-MAI/Z-Image-Turbo` checkpoint and load the result in a
   plain `onnxruntime.InferenceSession` (catches structural/shape bugs).
2. A hand-written pure-PyTorch reference (mirroring this file's graph node-for-node, using
   the same real loaded weights) compared against the ONNX Runtime output at three
   different resolutions, including a non-square one, to prove both numerical correctness
   and that dynamic height/width genuinely works with a single exported graph. See
   [How to Run](ZIMAGE_USAGE.md#verifying-numerical-correctness) for the reusable script.
