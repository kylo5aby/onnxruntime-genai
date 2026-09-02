import os
import json
import shutil
import subprocess
import sys
import argparse

from builders.zimage_text_encoder import strip_to_text_encoder

# Maps the user-facing -p choice to the underlying builder.py `-p/--precision` value and
# whether MatMulNBits int4 weight quantization should be applied.
PRECISION_CONFIGS = {
    "f16": {"builder_precision": "fp16", "int4_quant": False},
    "f32": {"builder_precision": "fp32", "int4_quant": False},
    "f16_int4_quant": {"builder_precision": "int4", "int4_quant": True},
    "f32_int4_quant": {"builder_precision": "int4", "int4_quant": True},
}

# Small tokenizer files that `builder.py`'s AutoTokenizer.from_pretrained needs co-located
# with the weights. They live in the repo's sibling `tokenizer/` folder, not `text_encoder/`.
TOKENIZER_FILES = ("merges.txt", "tokenizer.json", "tokenizer_config.json", "vocab.json")


def resolve_component_dir(input_path, component):
    # Accept either the Z-Image-Turbo repo root (containing a `<component>/` subfolder)
    # or the `<component>/` subfolder itself.
    candidate = os.path.join(input_path, component)
    if os.path.isfile(os.path.join(candidate, "config.json")):
        return candidate
    return input_path


def build_model(config):
    extra_options = str(config.get("extra_options", ""))
    precision_config = PRECISION_CONFIGS[config["precision"]]

    # Define the static (shared) options
    static_options = ["hf_remote=False"]
    if precision_config["int4_quant"]:
        static_options += [
                "block_size=32",
                "accuracy_level=4",
                "op_types_to_quantize=MatMul",
                ]
        # `-p int4` on the WebGPU EP defaults to float16 I/O (matching `f16_int4_quant`);
        # `use_webgpu_fp32` switches it to float32 I/O (`f32_int4_quant`).
        if config["precision"] == "f32_int4_quant":
            static_options.append("use_webgpu_fp32=true")

    # Get dynamic options from config and split them into a list
    dynamic_options = extra_options.split()

    # Merge them into one flat list of options
    all_options_list = static_options + dynamic_options

    # Construct the command list
    command = [
            "python", "builder.py",
            "-e", "webgpu",
            "-p", precision_config["builder_precision"],
            "--extra_options",
            *all_options_list,
            "-c", "tmp",
            "-i", config["input"],
            "-o", config["output"]
            ]

    print("\n" + " ".join(command))
    try:
        subprocess.run(command, check=True)
        print("\n######\nSuccess")
    except subprocess.CalledProcessError:
        print("\n######\nFail", file=sys.stderr)


def _link_or_copy(src, dst):
    # Prefer a cheap hardlink (same NTFS volume); fall back to a copy across volumes.
    if os.path.exists(dst):
        os.remove(dst)
    try:
        os.link(src, dst)
    except OSError:
        shutil.copy2(src, dst)


def _stage_text_encoder_input(text_encoder_dir, tokenizer_dir, stage_dir):
    # `builder.py` loads the tokenizer from the same folder as the weights, so assemble a
    # staging folder = text_encoder weights (hardlinked, ~8 GB) + tokenizer files (copied).
    os.makedirs(stage_dir, exist_ok=True)
    for fname in os.listdir(text_encoder_dir):
        src = os.path.join(text_encoder_dir, fname)
        if os.path.isfile(src):
            _link_or_copy(src, os.path.join(stage_dir, fname))

    missing = []
    for fname in TOKENIZER_FILES:
        src = os.path.join(tokenizer_dir, fname)
        if os.path.isfile(src):
            _link_or_copy(src, os.path.join(stage_dir, fname))
        else:
            missing.append(fname)
    if missing:
        print(f"Warning: tokenizer files not found in {tokenizer_dir}: {missing}", file=sys.stderr)


def build_text_encoder(input_path, output_dir):
    # The Z-Image-Turbo text encoder is a standard Qwen3 decoder; `builder.py` builds the
    # int4/fp16 WebGPU trunk, then `strip_to_text_encoder` turns it into a single-forward
    # encoder (penultimate hidden state -> fp16 `encoder_hidden_state`, KV cache removed).
    text_encoder_dir = resolve_component_dir(input_path, "text_encoder")
    config_path = os.path.join(text_encoder_dir, "config.json")
    if not os.path.isfile(config_path):
        print(f"Could not find text_encoder/config.json under {input_path}", file=sys.stderr)
        return

    # The tokenizer lives beside the text_encoder folder (repo_root/tokenizer).
    tokenizer_dir = os.path.join(os.path.dirname(os.path.normpath(text_encoder_dir)), "tokenizer")

    with open(config_path, encoding="utf-8") as f:
        cfg = json.load(f)
    num_layers = cfg["num_hidden_layers"]
    hidden_size = cfg["hidden_size"]

    stage_root = os.path.join(output_dir, "_staging")
    stage_in = os.path.join(stage_root, "input")
    stage_out = os.path.join(stage_root, "genai")
    if os.path.isdir(stage_root):
        shutil.rmtree(stage_root, ignore_errors=True)

    print(f"Staging text encoder input (weights + tokenizer) at {stage_in}")
    _stage_text_encoder_input(text_encoder_dir, tokenizer_dir, stage_in)

    # int4 weights (MatMul + Gather so the 152k x 2560 embedding is quantized), fp16 I/O,
    # WebGPU graph-capture (derives seqlens/total-seq-len from attention_mask, not KV cache).
    # `fuse_qk_norm_gqa=false` keeps Qwen3's QK-norm as separate SimplifiedLayerNorm ops and
    # emits a <=12-input GroupQueryAttention. The fused form (builder default) produces a
    # 16-input GQA that the deploy runtime (onnxruntime-web / onnxruntime 1.24 GQA schema,
    # max 12 inputs) rejects at load; this matches the reference text-encoder model.
    command = [
        "python", "builder.py",
        "-e", "webgpu",
        "-p", "int4",
        "--extra_options",
        "hf_remote=False",
        "block_size=32",
        "accuracy_level=4",
        "op_types_to_quantize=MatMul/Gather",
        "enable_webgpu_graph=true",
        "fuse_qk_norm_gqa=false",
        "-c", "tmp",
        "-i", stage_in,
        "-o", stage_out,
    ]
    print("\n" + " ".join(command))
    try:
        subprocess.run(command, check=True)
    except subprocess.CalledProcessError:
        print("\n######\nFail (builder.py)", file=sys.stderr)
        return

    os.makedirs(output_dir, exist_ok=True)
    final_onnx = os.path.join(output_dir, "text_encoder_model_q4f16.onnx")
    strip_to_text_encoder(
        input_onnx=os.path.join(stage_out, "model.onnx"),
        output_onnx=final_onnx,
        num_layers=num_layers,
        external_data_name="text_encoder_model_q4f16.onnx_data",
        hidden_size=hidden_size,
    )

    # Drop the ~2.4 GB genai staging artifacts now that the encoder is written.
    shutil.rmtree(stage_root, ignore_errors=True)
    print("\n######\nSuccess")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "input",
        help="Input model folder: the Z-Image-Turbo repo root or a component subfolder",
    )
    parser.add_argument(
        "-m",
        "--model",
        choices=["transformer", "text_encoder"],
        default="transformer",
        help="Which Z-Image-Turbo component to build. Default: transformer.",
    )
    parser.add_argument(
        "-p",
        "--precision",
        choices=list(PRECISION_CONFIGS.keys()),
        default="f16_int4_quant",
        help=(
            "Precision to build: f16/f32 (unquantized WebGPU I/O dtype) or "
            "f16_int4_quant/f32_int4_quant (int4-quantized weights with float16/float32 "
            "WebGPU I/O). Default: f16_int4_quant. Ignored for -m text_encoder (always q4f16)."
        ),
    )
    args = parser.parse_args()

    model_name = os.path.basename(os.path.normpath(args.input))

    if args.model == "text_encoder":
        if args.precision != "f16_int4_quant":
            print("Note: -m text_encoder always builds q4f16 (fp16 I/O); ignoring -p.")
        output = f"{model_name}-text_encoder-genai-wgpu-f16_int4_quant"
        build_text_encoder(args.input, output)
    else:
        transformer_dir = resolve_component_dir(args.input, "transformer")
        output = f"{model_name}-transformer-genai-wgpu-{args.precision}"
        model_config = {
            "input": transformer_dir,
            "output": output,
            "precision": args.precision,
        }
        build_model(model_config)
