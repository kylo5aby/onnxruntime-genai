import os
import subprocess
import sys
import argparse

# Maps the user-facing -p choice to the underlying builder.py `-p/--precision` value and
# whether MatMulNBits int4 weight quantization should be applied.
PRECISION_CONFIGS = {
    "f16": {"builder_precision": "fp16", "int4_quant": False},
    "f32": {"builder_precision": "fp32", "int4_quant": False},
    "f16_int4_quant": {"builder_precision": "int4", "int4_quant": True},
    "f32_int4_quant": {"builder_precision": "int4", "int4_quant": True},
}

def resolve_transformer_dir(input_path):
    # Accept either the Z-Image-Turbo repo root (containing a `transformer/` subfolder)
    # or the `transformer/` subfolder itself.
    candidate = os.path.join(input_path, "transformer")
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
                "op_types_to_quantize=MatMul/Gather",
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

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "input",
        help="Input model folder: the Z-Image-Turbo repo root or its `transformer` subfolder",
    )
    parser.add_argument(
        "-p",
        "--precision",
        choices=list(PRECISION_CONFIGS.keys()),
        default="f16_int4_quant",
        help=(
            "Precision to build: f16/f32 (unquantized WebGPU I/O dtype) or "
            "f16_int4_quant/f32_int4_quant (int4-quantized weights with float16/float32 "
            "WebGPU I/O). Default: f16_int4_quant."
        ),
    )
    args = parser.parse_args()

    transformer_dir = resolve_transformer_dir(args.input)

    model_name = os.path.basename(os.path.normpath(args.input))
    output = f'{model_name}-transformer-genai-wgpu-{args.precision}'
    model_config = {
                "input": transformer_dir,
                "output": output,
                "precision": args.precision,
                }

    build_model(model_config)
