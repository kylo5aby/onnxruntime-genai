# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation.  All rights reserved.
# Licensed under the MIT License.  See License.txt in the project root for
# license information.
# --------------------------------------------------------------------------
"""Post-process a genai-built Qwen3 decoder into the Z-Image-Turbo text encoder.

The Z-Image-Turbo pipeline drives its DiT transformer with caption features taken
from a Qwen3 language model. Rather than a full autoregressive decoder, it needs a
single-forward encoder that maps ``input_ids``/``attention_mask`` to one hidden-state
tensor. `builder.py` already knows how to build the Qwen3 trunk (int4 weights, fp16 I/O,
WebGPU); this module performs the graph surgery that turns that decoder into the encoder,
mirroring the hand-run reference `modify_genai_model.py` but parameterized and emitting a
float16 output (the runtime auto-detects fp16, so we skip the reference's float32 cast).

The surgery:
  1. Tap the penultimate hidden state -- the SkipLayerNorm residual sum entering the last
     decoder layer, ``/model/layers.{num_layers-1}/input_layernorm/output_3`` (== HF
     ``hidden_states[-2]``) -- and re-expose it as the graph output ``encoder_hidden_state``.
  2. Drop the KV-cache graph inputs (``past_key_values.*``); the WebGPU GQA mask subgraph
     derives seqlens/total-seq-len from ``attention_mask`` alone, so empty KV is valid.
  3. Replace all outputs (``logits``/``present.*``) with the single ``encoder_hidden_state``.
  4. Dead-code-eliminate everything unreachable from the new output (last layer, final
     norm, LM head) and prune the now-unused initializers.
"""

import os

import onnx
from onnx import TensorProto, helper


def strip_to_text_encoder(
    input_onnx,
    output_onnx,
    num_layers,
    external_data_name,
    hidden_size=None,
    output_name="encoder_hidden_state",
):
    """Rewrite a genai Qwen3 decoder ONNX into the Z-Image-Turbo text encoder.

    Args:
        input_onnx: path to the genai-built ``model.onnx`` (with external data alongside).
        output_onnx: path to write the encoder ``.onnx`` (external data written next to it).
        num_layers: number of decoder layers; the penultimate tap is ``layers.{num_layers-1}``.
        external_data_name: filename for the external-data blob (e.g. ``*.onnx_data``).
        hidden_size: hidden dim for the declared output shape; ``None`` leaves it unshaped.
        output_name: name of the single graph output tensor.
    """
    tap_name = f"/model/layers.{num_layers - 1}/input_layernorm/output_3"

    print(f"Loading genai model: {input_onnx}")
    model = onnx.load(input_onnx)
    graph = model.graph
    print(f"  original nodes={len(graph.node)} initializers={len(graph.initializer)}")

    # 1. Re-expose the penultimate residual as `encoder_hidden_state` (fp16).
    #    Rename the producing node's output slot in place -- no extra op. Downstream
    #    consumers of the old name become unreachable and are removed by DCE below.
    producer = None
    for node in graph.node:
        for i, out in enumerate(node.output):
            if out == tap_name:
                node.output[i] = output_name
                producer = node
                break
        if producer is not None:
            break
    if producer is None:
        raise RuntimeError(
            f"Tap tensor {tap_name!r} not found in {input_onnx}. The genai builder may have "
            f"changed its layernorm output naming; update the tap in strip_to_text_encoder()."
        )
    print(f"  tapped {tap_name} -> {output_name} (produced by {producer.name!r})")

    # Keep any matching value_info annotation consistent with the renamed tensor.
    for value_info in graph.value_info:
        if value_info.name == tap_name:
            value_info.name = output_name

    # 2. Drop KV-cache inputs; blank out node references so kept attention ops run with
    #    empty past (valid for a full-sequence forward on the WebGPU GQA path).
    kv_names = {inp.name for inp in graph.input if inp.name.startswith("past_key_values")}
    kept_inputs = [inp for inp in graph.input if inp.name not in kv_names]
    del graph.input[:]
    graph.input.extend(kept_inputs)
    for node in graph.node:
        for i, inp in enumerate(node.input):
            if inp in kv_names:
                node.input[i] = ""
    print(f"  removed {len(kv_names)} past_key_values inputs; kept {[i.name for i in graph.input]}")

    # 3. Replace all outputs with the single encoder output.
    del graph.output[:]
    output_shape = ["batch_size", "sequence_length", hidden_size] if hidden_size else None
    graph.output.append(helper.make_tensor_value_info(output_name, TensorProto.FLOAT16, output_shape))

    # 4. Dead-code elimination: keep only what is reachable (backward) from the output.
    output_to_node = {}
    for idx, node in enumerate(graph.node):
        for out in node.output:
            if out:
                output_to_node[out] = idx

    keep_indices = set()
    needed = set()  # tensors that must survive (for pruning initializers)
    visited = set()
    queue = [out.name for out in graph.output]
    while queue:
        tensor = queue.pop()
        if tensor in visited:
            continue
        visited.add(tensor)
        needed.add(tensor)
        producer_idx = output_to_node.get(tensor)
        if producer_idx is not None:
            keep_indices.add(producer_idx)
            for inp in graph.node[producer_idx].input:
                if inp and inp not in visited:
                    queue.append(inp)

    kept_nodes = [graph.node[i] for i in sorted(keep_indices)]
    del graph.node[:]
    graph.node.extend(kept_nodes)

    kept_initializers = [init for init in graph.initializer if init.name in needed]
    del graph.initializer[:]
    graph.initializer.extend(kept_initializers)
    print(f"  optimized nodes={len(graph.node)} initializers={len(graph.initializer)}")

    # 5. Save with external data (drop any pre-existing blob so we don't leave a stale file).
    os.makedirs(os.path.dirname(os.path.abspath(output_onnx)), exist_ok=True)
    external_data_path = os.path.join(os.path.dirname(os.path.abspath(output_onnx)), external_data_name)
    if os.path.exists(external_data_path):
        os.remove(external_data_path)
    onnx.save_model(
        model,
        output_onnx,
        save_as_external_data=True,
        all_tensors_to_one_file=True,
        location=external_data_name,
        size_threshold=1024 * 1024 * 10,
        convert_attribute=False,
    )

    print(f"  inputs : {[i.name for i in graph.input]}")
    print(f"  output : {output_name} (float16)")
    print(f"Saved text encoder: {output_onnx}")

    # Best-effort validation. Pass the path so the checker streams external data instead of
    # serializing the (multi-GB) in-memory proto, which would trip the 2GB protobuf limit.
    try:
        onnx.checker.check_model(output_onnx)
        print("  onnx.checker: passed")
    except Exception as exc:  # noqa: BLE001 - validation is advisory; the model is already written
        print(f"  onnx.checker: WARNING - {exc}")

    return output_onnx
