# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation.  All rights reserved.
# Licensed under the MIT License.  See License.txt in the project root for
# license information.
# --------------------------------------------------------------------------
"""Z-Image-Turbo VAE decoder (`AutoencoderKL.decoder`) exporter.

This builds the VAE decoder ONNX graph directly from the PyTorch weights.

Two GroupNorm flavours are supported, selected by `--extra_options fuse_group_norm=...`:

  * `fuse_group_norm=false` (default): every diffusers GroupNorm is decomposed into
    standard ONNX ops, all in NCHW with no Transposes:
    `Reshape[0,G,-1] -> InstanceNormalization(scale=1,bias=0) -> Reshape(back)
    -> Mul(gamma) -> Add(beta) [-> Sigmoid -> Mul]`. Residual adds are emitted as
    plain `Add` nodes. The graph contains no `com.microsoft` contrib ops, so it runs
    on any EP with an `InstanceNormalization` kernel (e.g. WebGPU today).

  * `fuse_group_norm=true`: each GroupNorm is emitted as a single channels-last
    `com.microsoft.GroupNorm` with the SiLU/swish activation fused in
    (`activation=1`), and each residual-`Add`->norm seam is fused into one
    `com.microsoft.SkipGroupNorm` (`S = X + skip`, `Y = GroupNorm(S)`). The 3
    nearest-neighbour upsamplers then run `Resize` in NHWC (scales permuted
    `[1,1,2,2]`->`[1,2,2,1]`) sandwiched in Transposes that cancel against the
    neighbouring GroupNorm Transposes. Requires an onnxruntime build whose EP
    implements those contrib ops.

Common to both:

  * The diffusers `output_scale_factor=1.0` residual division (a `Div` by 1.0) is
    simply never emitted.

  * The mid-block self-attention is emitted UNFUSED (MatMul/Softmax/MatMul),
    matching the original float16 export bit-for-bit. It is deliberately NOT a
    `com.microsoft.MultiHeadAttention`: this VAE's attention logits overflow
    float16 (diffusers uses `force_upcast=true`), and the fused kernel turns that
    overflow into NaN.

In both flavours the GroupNorm gamma/beta are the PyTorch `norm.weight`/`norm.bias`
directly (no InstanceNorm per-group affine to fold and re-round in float16).

Scope: decoder only (no encoder), dynamic batch and latent height/width. Graph I/O,
weights and compute all use `io_dtype` (f16 for `-p fp16` on WebGPU, f32 for `-p fp32`),
consistent with the Z-Image transformer export.
"""

import json
import os
import types

import onnx_ir as ir
import torch

from .base import Model


class ZImageVAEDecoderModel(Model):
    def __init__(self, config, io_dtype, onnx_dtype, ep, cache_dir, extra_options):
        mid_channels = int(config.block_out_channels[-1])
        fake_config = types.SimpleNamespace(
            _name_or_path=getattr(config, "_name_or_path", "z-image-vae-decoder"),
            architectures=["AutoencoderKL"],
            hidden_size=mid_channels,
            num_attention_heads=1,
            num_key_value_heads=1,
            num_hidden_layers=1,
            intermediate_size=mid_channels,
            vocab_size=0,
            hidden_act="silu",
            max_position_embeddings=1,
            rms_norm_eps=1e-6,
        )
        super().__init__(fake_config, io_dtype, onnx_dtype, ep, cache_dir, extra_options)

        self.model_type = "z-image-vae-decoder"
        # Spatial self-attention: bidirectional, no KV cache.
        self.attention_attrs["unidirectional"] = False

        self.mid_channels = mid_channels
        self.latent_channels = int(config.latent_channels)
        self.out_channels = int(config.out_channels)
        self.block_out_channels = [int(c) for c in config.block_out_channels]
        self.norm_num_groups = int(config.norm_num_groups)
        # False (default): decomposed standard-ONNX GroupNorm; True: com.microsoft GroupNorm/SkipGroupNorm.
        self.fuse_group_norm = bool(extra_options.get("fuse_group_norm", False))
        # Graph I/O, weights and internal compute all follow `io_dtype` (float16 for
        # `-p fp16` on WebGPU, float32 with `-p fp32`/`use_webgpu_fp32`), matching the
        # Z-Image transformer export.
        self._io_dtype = self.io_dtype

    # ------------------------------------------------------------------
    # Weight loading
    # ------------------------------------------------------------------
    def load_weights(self, input_path):
        from diffusers import AutoencoderKL

        # Load from `input_path` (the `-i` dir) rather than `self.model_name_or_path`
        model = AutoencoderKL.from_pretrained(
            input_path,
            cache_dir=self.cache_dir,
            token=self.hf_token,
            trust_remote_code=self.hf_remote,
        )
        model.eval()
        return model.decoder

    # ------------------------------------------------------------------
    # Inputs / outputs
    # ------------------------------------------------------------------
    def make_inputs_and_outputs(self):
        self.input_names = {"latent_sample": "latent_sample"}
        self.output_names = {"sample": "sample"}

        latent = self.make_value(
            "latent_sample", self._io_dtype,
            shape=["batch_size", self.latent_channels, "latent_height", "latent_width"],
        )
        self.model.graph.inputs.extend([latent])

        sample = self.make_value(
            "sample", self._io_dtype, shape=["batch_size", self.out_channels, "height", "width"]
        )
        self.model.graph.outputs.append(sample)

    # ------------------------------------------------------------------
    # Small node-building helpers
    # ------------------------------------------------------------------
    def _init_name(self, node_name, suffix):
        return node_name[1:].replace("/", ".") + suffix

    def _const(self, dtype, value):
        """Return the name of an inline `Constant` node holding `value`.
        """
        return f"/model/constants/{self.to_str_dtype(dtype)}/{value}"

    def _transpose(self, name, root_input, perm):
        self.make_transpose(name, root_input, self._io_dtype, shape=None, perm=perm)
        return f"{name}/output_0"

    def _add(self, name, inputs):
        self.make_add(name, inputs, self._io_dtype, shape=None)
        return f"{name}/output_0"

    def _mul(self, name, inputs):
        self.make_mul(name, inputs, self._io_dtype, shape=None)
        return f"{name}/output_0"

    def _reshape(self, name, root_input, shape_tensor_name):
        self.make_reshape(name, [root_input, shape_tensor_name], self._io_dtype, shape=None)
        return f"{name}/output_0"

    def _conv(self, name, root_input, conv):
        """Emit an `nn.Conv2d` (NCHW). Weight/bias initializers derive from the node name."""
        weight = self._init_name(name, ".weight")
        self.make_initializer(conv.weight, weight, to=self._io_dtype)
        inputs = [root_input, weight]
        if conv.bias is not None:
            bias = self._init_name(name, ".bias")
            self.make_initializer(conv.bias, bias, to=self._io_dtype)
            inputs.append(bias)

        kh, kw = conv.kernel_size
        ph, pw = conv.padding
        sh, sw = conv.stride
        dh, dw = conv.dilation
        self.make_conv(
            name, inputs, self._io_dtype, shape=None,
            strides=[sh, sw], pads=[ph, pw, ph, pw], dilations=[dh, dw],
            group=conv.groups, kernel_shape=[kh, kw],
        )
        return f"{name}/output_0"

    def _linear(self, name, root_input, linear):
        """Attention Q/K/V/out projection. Goes through `make_matmul` so `-p int4/int8` applies."""
        self.make_matmul(linear, name, root_input, seq_dim="attn_seq")
        output = f"{name}/output_0"
        if linear.bias is not None:
            self.make_add_bias(linear.bias, f"{name}/Add", output, seq_dim="attn_seq")
            output = f"{name}/Add/output_0"
        return output

    # ------------------------------------------------------------------
    # GroupNorm (fused contrib op or decomposed standard ops)
    # ------------------------------------------------------------------
    def _group_norm(self, name, source, gnmod, swish, expose_skip=True):
        """Emit a GroupNorm (optionally with a folded residual add and a fused SiLU).

        `source` is either a materialized NCHW tensor name (plain GroupNorm) or a
        `(conv_branch, skip_branch)` pair whose sum is the norm input. Returns
        `(y_nchw, s_nchw)`; `s_nchw` is the residual sum `X + skip` (None for the
        plain case, or when `expose_skip=False`).

        Dispatches on `self.fuse_group_norm`: the fused path emits channels-last
        `com.microsoft.GroupNorm`/`SkipGroupNorm`; the default path stays in NCHW
        and uses only standard ONNX ops.
        """
        if self.fuse_group_norm:
            return self._group_norm_fused(name, source, gnmod, swish, expose_skip)

        s = None
        if isinstance(source, tuple):
            conv_branch, skip_branch = source
            s = self._add(f"{name}/skip_add", [conv_branch, skip_branch])
            x = s
        else:
            x = source
        y = self._group_norm_decomposed(name, x, gnmod, swish)
        return y, (s if expose_skip else None)

    def _group_norm_decomposed(self, name, x_nchw, gnmod, swish):
        """Standard-ONNX GroupNorm in NCHW (the PyTorch exporter's decomposition):

            Reshape[0,G,-1] -> InstanceNormalization(scale=1, bias=0)
            -> Reshape(input shape) -> Mul(gamma[C,1,1]) -> Add(beta[C,1,1])
            [-> Sigmoid -> Mul]   (SiLU when `swish`)

        `InstanceNormalization` normalizes each `[G]` row over `C/G*H*W`, which is
        exactly GroupNorm's statistics; its per-group affine is the identity so the
        real gamma/beta apply per-channel afterwards. Shapes stay dynamic in
        batch/H/W via `Reshape`'s dim-copy (`0`) and a `Shape` node for the way back.
        """
        groups = int(gnmod.num_groups)
        channels = int(gnmod.num_channels)
        eps = float(gnmod.eps)

        gamma = self._init_name(name, ".weight")
        beta = self._init_name(name, ".bias")
        self.make_initializer(gnmod.weight.detach().reshape(channels, 1, 1), gamma, to=self._io_dtype)
        self.make_initializer(gnmod.bias.detach().reshape(channels, 1, 1), beta, to=self._io_dtype)
        in_scale = f"{name}/instnorm_scale"
        in_bias = f"{name}/instnorm_bias"
        self.make_initializer(torch.ones(groups), in_scale, to=self._io_dtype)
        self.make_initializer(torch.zeros(groups), in_bias, to=self._io_dtype)

        self.make_shape(f"{name}/in_shape", x_nchw, shape=[4])
        grouped = self._reshape(
            f"{name}/to_groups", x_nchw, self._const(ir.DataType.INT64, [0, groups, -1])
        )
        normed = f"{name}/InstanceNormalization/output_0"
        self.make_node(
            "InstanceNormalization", inputs=[grouped, in_scale, in_bias], outputs=[normed],
            name=f"{name}/InstanceNormalization", epsilon=eps,
        )
        self.make_value(normed, self._io_dtype)
        y = self._reshape(f"{name}/from_groups", normed, f"{name}/in_shape/output_0")
        y = self._mul(f"{name}/Mul", [y, gamma])
        y = self._add(f"{name}/Add", [y, beta])
        if swish:
            self.make_sigmoid(f"{name}/Sigmoid", y, self._io_dtype, shape=None)
            y = self._mul(f"{name}/silu", [y, f"{name}/Sigmoid/output_0"])
        return y

    def _group_norm_fused(self, name, source, gnmod, swish, expose_skip):
        """Channels-last `com.microsoft.GroupNorm`/`SkipGroupNorm` bracketed by
        NCHW<->NHWC Transposes. Same contract as `_group_norm`.
        """
        gamma = self._init_name(name, ".weight")
        beta = self._init_name(name, ".bias")
        self.make_initializer(gnmod.weight, gamma, to=self._io_dtype)
        self.make_initializer(gnmod.bias, beta, to=self._io_dtype)
        groups = int(gnmod.num_groups)
        eps = float(gnmod.eps)
        activation = 1 if swish else 0

        if isinstance(source, tuple):
            conv_branch, skip_branch = source
            x_nhwc = self._transpose(f"{name}/x_t", conv_branch, [0, 2, 3, 1])
            skip_nhwc = self._transpose(f"{name}/skip_t", skip_branch, [0, 2, 3, 1])
            y_nhwc = f"{name}/output_0"
            outputs = [y_nhwc]
            s_nhwc = None
            if expose_skip:
                s_nhwc = f"{name}/sum"
                outputs.append(s_nhwc)
            self.make_node(
                "SkipGroupNorm", inputs=[x_nhwc, gamma, beta, skip_nhwc], outputs=outputs,
                name=name, domain="com.microsoft",
                activation=activation, channels_last=1, epsilon=eps, groups=groups,
            )
            for o in outputs:
                self.make_value(o, self._io_dtype)
            y = self._transpose(f"{name}/post_t", y_nhwc, [0, 3, 1, 2])
            s = self._transpose(f"{name}/sum_post_t", s_nhwc, [0, 3, 1, 2]) if s_nhwc else None
            return y, s

        x_nhwc = self._transpose(f"{name}/x_t", source, [0, 2, 3, 1])
        y_nhwc = f"{name}/output_0"
        self.make_node(
            "GroupNorm", inputs=[x_nhwc, gamma, beta], outputs=[y_nhwc],
            name=name, domain="com.microsoft",
            activation=activation, channels_last=1, epsilon=eps, groups=groups,
        )
        self.make_value(y_nhwc, self._io_dtype)
        y = self._transpose(f"{name}/post_t", y_nhwc, [0, 3, 1, 2])
        return y, None

    # ------------------------------------------------------------------
    # ResNet block
    # ------------------------------------------------------------------
    def _resnet(self, name, source, resnet):
        """Emit a diffusers ResnetBlock2D. Returns a *deferred* `(conv2_out, residual_base)`
        pair: the final residual add is not emitted here so the consumer can either fold it
        into its own SkipGroupNorm (residual->norm seam) or materialize it with `_add`.
        """
        y, s = self._group_norm(f"{name}/norm1", source, resnet.norm1, swish=True, expose_skip=True)
        if s is not None:
            # SkipGroupNorm case: identity shortcut, residual base is the fused sum S.
            residual_base = s
        elif resnet.conv_shortcut is not None:
            residual_base = self._conv(f"{name}/conv_shortcut", source, resnet.conv_shortcut)
        else:
            residual_base = source

        h = self._conv(f"{name}/conv1", y, resnet.conv1)
        y2, _ = self._group_norm(f"{name}/norm2", h, resnet.norm2, swish=True)
        h = self._conv(f"{name}/conv2", y2, resnet.conv2)
        return (h, residual_base)

    # ------------------------------------------------------------------
    # Mid-block self-attention (unfused, single-head)
    # ------------------------------------------------------------------
    def _mid_attention(self, name, x_nchw, attn):
        """Single-head spatial self-attention, emitted UNFUSED (MatMul -> Softmax -> MatMul).
        """
        assert attn.heads == 1, f"expected single-head VAE attention, got heads={attn.heads}"
        # group_norm (plain, no activation), NCHW -> channels-last and back.
        gn, _ = self._group_norm(f"{name}/group_norm", x_nchw, attn.group_norm, swish=False)

        # [N, C, H, W] -> [N, C, H*W] -> [N, H*W, C]. The leading 0 is ONNX Reshape's
        # "copy this dim from the input" (allowzero=0), keeping batch dynamic.
        to_seq_shape = self._const(ir.DataType.INT64, [0, self.mid_channels, -1])
        flat = self._reshape(f"{name}/flatten", gn, to_seq_shape)
        seq = self._transpose(f"{name}/to_seq", flat, [0, 2, 1])

        q = self._linear(f"{name}/to_q", seq, attn.to_q)
        k = self._linear(f"{name}/to_k", seq, attn.to_k)
        v = self._linear(f"{name}/to_v", seq, attn.to_v)

        # scores = softmax(Q @ K^T / sqrt(head_size)) @ V
        kt = self._transpose(f"{name}/kT", k, [0, 2, 1])  # [1, C, H*W]
        scores = f"{name}/scores/output_0"
        self.make_node("MatMul", inputs=[q, kt], outputs=[scores], name=f"{name}/scores")
        self.make_value(scores, self._io_dtype)
        scaled = self._mul(f"{name}/scale", [scores, self._const(self._io_dtype, self.mid_channels ** -0.5)])
        probs = f"{name}/softmax/output_0"
        self.make_node("Softmax", inputs=[scaled], outputs=[probs], name=f"{name}/softmax", axis=-1)
        self.make_value(probs, self._io_dtype)
        ctx = f"{name}/context/output_0"
        self.make_node("MatMul", inputs=[probs, v], outputs=[ctx], name=f"{name}/context")
        self.make_value(ctx, self._io_dtype)
        attn_out = self._linear(f"{name}/to_out.0", ctx, attn.to_out[0])

        # [1, H*W, C] -> [1, C, H*W] -> [1, C, H, W] (restore spatial dims from the input)
        back_seq = self._transpose(f"{name}/from_seq", attn_out, [0, 2, 1])
        self.make_shape(f"{name}/in_shape", x_nchw, shape=[4])
        spatial = self._reshape(f"{name}/unflatten", back_seq, f"{name}/in_shape/output_0")

        # residual connection (output_scale_factor / rescale = 1.0, so no division)
        return self._add(f"{name}/Add", [spatial, x_nchw])

    # ------------------------------------------------------------------
    # Upsampler (nearest 2x, then conv)
    # ------------------------------------------------------------------
    def _upsample(self, name, x_nchw, conv):
        """Nearest 2x `Resize` then conv. In the fused flavour the Resize runs in NHWC so its
        Transposes cancel against the neighbouring GroupNorm ones; otherwise it stays NCHW.
        """
        if self.fuse_group_norm:
            x = self._transpose(f"{name}/pre_t", x_nchw, [0, 2, 3, 1])
            scales = self._const(ir.DataType.FLOAT, [1.0, 2.0, 2.0, 1.0])
        else:
            x = x_nchw
            scales = self._const(ir.DataType.FLOAT, [1.0, 1.0, 2.0, 2.0])
        resize_out = f"{name}/Resize/output_0"
        self.make_node(
            "Resize", inputs=[x, "", scales], outputs=[resize_out], name=f"{name}/Resize",
            coordinate_transformation_mode="asymmetric", cubic_coeff_a=-0.75,
            mode="nearest", nearest_mode="floor",
        )
        self.make_value(resize_out, self._io_dtype)
        y_nchw = self._transpose(f"{name}/post_t", resize_out, [0, 3, 1, 2]) if self.fuse_group_norm else resize_out
        return self._conv(f"{name}/conv", y_nchw, conv)

    # ------------------------------------------------------------------
    # Top-level graph construction
    # ------------------------------------------------------------------
    def make_model(self, input_path):
        self.make_inputs_and_outputs()
        self.weights = self.load_weights(input_path)
        dec = self.weights

        x = self.input_names["latent_sample"]

        # --- conv_in ---
        x = self._conv("/decoder/conv_in", x, dec.conv_in)

        # --- mid block: resnet0 -> attention -> resnet1 ---
        mid = dec.mid_block
        r0 = self._resnet("/decoder/mid_block/resnets.0", x, mid.resnets[0])
        x = self._add("/decoder/mid_block/resnets.0/Add", list(r0))  # feeds attention (non-norm)
        x = self._mid_attention("/decoder/mid_block/attentions.0", x, mid.attentions[0])
        # resnet1's output feeds up_blocks.0.resnets.0.norm1 (a SkipGroupNorm seam): defer.
        source = self._resnet("/decoder/mid_block/resnets.1", x, mid.resnets[1])

        # --- up blocks ---
        for bi, up in enumerate(dec.up_blocks):
            n_res = len(up.resnets)
            for rj in range(n_res):
                r = self._resnet(f"/decoder/up_blocks.{bi}/resnets.{rj}", source, up.resnets[rj])
                if rj < n_res - 1:
                    # next resnet's norm1 folds this residual add (SkipGroupNorm seam).
                    source = r
                elif getattr(up, "upsamplers", None):
                    # last resnet before an upsampler: materialize, then Resize+conv.
                    x = self._add(f"/decoder/up_blocks.{bi}/resnets.{rj}/Add", list(r))
                    x = self._upsample(f"/decoder/up_blocks.{bi}/upsamplers.0", x, up.upsamplers[0].conv)
                    source = x  # feeds next block's resnets.0.norm1 (plain GroupNorm, fed by a Conv)
                else:
                    # last block, no upsampler: conv_norm_out folds this residual add.
                    source = r

        # --- conv_norm_out (SkipGroupNorm, swish) -> conv_out ---
        y, _ = self._group_norm(
            "/decoder/conv_norm_out", source, dec.conv_norm_out, swish=True, expose_skip=False
        )
        out = self._conv("/decoder/conv_out", y, dec.conv_out)

        self.make_node("Identity", inputs=[out], outputs=["sample"], name="/decoder/output_identity")

        del self.weights

    # ------------------------------------------------------------------
    # Save (single self-contained .onnx file)
    # ------------------------------------------------------------------
    def save_model(self, out_dir):
        """Save as ONE self-contained `.onnx` file.
        """
        print(f"Saving ONNX model in {out_dir}")
        already_quantized_in_qdq_format = self.quant_type is not None and self.quant_attrs["use_qdq"]
        if self.onnx_dtype in {ir.DataType.INT4, ir.DataType.UINT4, ir.DataType.INT8, ir.DataType.UINT8} and not already_quantized_in_qdq_format:
            model = self.to_nbits()
        else:
            model = self.model

        model.graph.sort()

        out_path = os.path.join(out_dir, self.filename)
        data_path = out_path + ".data"
        for stale in (out_path, data_path):  # remove any prior two-file output too
            if os.path.exists(stale):
                print(f"Overwriting {stale}")
                os.remove(stale)

        ir.save(model, out_path)  # inline weights -> single file

        if os.path.isdir(self.cache_dir) and not os.listdir(self.cache_dir):
            os.rmdir(self.cache_dir)

    # ------------------------------------------------------------------
    # GenAI config / processing files
    # ------------------------------------------------------------------
    def make_genai_config(self, config, extra_kwargs, out_dir):
        genai_config = {
            "model": {
                "type": self.model_type,
                "decoder": {
                    "filename": self.filename,
                    "latent_channels": self.latent_channels,
                    "out_channels": self.out_channels,
                    "block_out_channels": self.block_out_channels,
                    "norm_num_groups": self.norm_num_groups,
                    "scale_factor": 8,
                    "inputs": {"latent_sample": "latent_sample"},
                    "outputs": {"sample": "sample"},
                },
            },
        }
        print(f"Saving GenAI config in {out_dir}")
        with open(os.path.join(out_dir, "genai_config.json"), "w") as f:
            json.dump(genai_config, f, indent=4)

    def save_processing(self, model_name_or_path, extra_kwargs, out_dir):
        # No tokenizer/processor: this exports the VAE decoder graph only.
        pass
