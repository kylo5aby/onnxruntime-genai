import os
import sys
import argparse
import time

import numpy as np
import onnxruntime as ort
import psutil
import torch
from pathlib import Path
from PIL import Image
from transformers import AutoTokenizer

# --- Helper function for custom sample printing ---
def _format_and_print_data_with_offset(
    flat_tensor: np.ndarray, elements_to_show: int, name: str
):
    elements_to_show = min(elements_to_show, flat_tensor.size)
    if elements_to_show == 0:
        return

    elements_per_line = 16
    num_lines = (elements_to_show + elements_per_line - 1) // elements_per_line

    print(
        f"  --- Sample Elements (Offset/16-per-line format - First {elements_to_show}): ---"
    )

    for i in range(num_lines):
        start_index = i * elements_per_line
        end_index = min(start_index + elements_per_line, elements_to_show)
        line_data = flat_tensor[start_index:end_index]

        offset_str = f"0x{start_index:04X}: "
        data_str_parts = [f"{x:>4.2f}" for x in line_data]

        num_padding = elements_per_line - len(line_data)
        padding_str = " " * (num_padding * 8)

        final_line = f"{offset_str}{' '.join(data_str_parts)}{padding_str}"

        print(f"  {final_line.rstrip()}")

    if flat_tensor.size > elements_to_show:
        print("  [... remaining elements not shown ...]")


def log_tensor_stats(tensor: np.ndarray, tensor_name: str, elements_to_show: int = 64):
    """
    Prints comprehensive statistics and a custom sample for a NumPy array.
    """
    if not isinstance(tensor, np.ndarray):
        if isinstance(tensor, torch.Tensor):
            np_tensor = tensor.detach().cpu().numpy()
        else:
            print(f"ERROR: '{tensor_name}' is not a NumPy array. Type: {type(tensor)}")
            return
    else:
        np_tensor = tensor

    num_elements = np_tensor.size
    shape_str = str(np_tensor.shape)
    dtype_str = str(np_tensor.dtype)
    flat_tensor = np_tensor.ravel()

    print(f"--- Tensor Stats: {tensor_name} ---")
    print(f"  Shape: {shape_str} | DType: {dtype_str} | Elements: {num_elements}")

    if num_elements == 0:
        print("  WARNING: Tensor is empty (0 elements).")
        return

    try:
        mean = np.mean(flat_tensor)
        std = np.std(flat_tensor)
        min_val = np.min(flat_tensor)
        max_val = np.max(flat_tensor)
        l2_norm = np.linalg.norm(flat_tensor)
        abs_sum = np.sum(np.abs(flat_tensor))

        print(f"  Min: {min_val:.2f} | Max: {max_val:.2f}")
        print(f"  Mean: {mean:.2f} | Std Dev: {std:.2f}")
        print(
            f"  L2 Norm: {l2_norm:.2f} | Abs Sum (L1): {abs_sum:.2f}"
        )

    except Exception as e:
        print(f"  ERROR: Error calculating statistics for '{tensor_name}': {e}")
        return

    if np.issubdtype(np_tensor.dtype, np.floating):
        num_nan = np.count_nonzero(np.isnan(flat_tensor))
        num_inf = np.count_nonzero(np.isinf(flat_tensor))

        if num_nan > 0 or num_inf > 0:
            print(f"  !!! NON-FINITE VALUES DETECTED !!!")
            print(f"  NaN Count: {num_nan} ({num_nan/num_elements*100:.2f}%)")
            print(f"  Inf Count: {num_inf} ({num_inf/num_elements*100:.2f}%)")
        else:
            print("  Non-Finite Check: OK (No NaN/Inf)")

    if elements_to_show > 0:
        _format_and_print_data_with_offset(flat_tensor, elements_to_show, tensor_name)


# Standard Qwen3 chat template. The WebNN Z-Image-Turbo tokenizer export doesn't ship a
# `chat_template` in its tokenizer_config.json (unlike the upstream Tongyi-MAI/Z-Image-Turbo
# repo's tokenizer), so newer `transformers` versions raise
# "Cannot use chat template functions because tokenizer.chat_template is not set" on
# `apply_chat_template`. Used as a fallback in `initialize_tokenizer` when the loaded
# tokenizer has no chat template of its own.
QWEN3_CHAT_TEMPLATE = r"""{%- if tools %}
    {{- '<|im_start|>system\n' }}
    {%- if messages[0].role == 'system' %}
        {{- messages[0].content + '\n\n' }}
    {%- endif %}
    {{- "# Tools\n\nYou may call one or more functions to assist with the user query.\n\nYou are provided with function signatures within <tools></tools> XML tags:\n<tools>" }}
    {%- for tool in tools %}
        {{- "\n" }}
        {{- tool | tojson }}
    {%- endfor %}
    {{- "\n</tools>\n\nFor each function call, return a json object with function name and arguments within <tool_call></tool_call> XML tags:\n<tool_call>\n{\"name\": <function-name>, \"arguments\": <args-json-object>}\n</tool_call><|im_end|>\n" }}
{%- else %}
    {%- if messages[0].role == 'system' %}
        {{- '<|im_start|>system\n' + messages[0].content + '<|im_end|>\n' }}
    {%- endif %}
{%- endif %}
{%- set ns = namespace(multi_step_tool=true, last_query_index=messages|length - 1) %}
{%- for message in messages[::-1] %}
    {%- set index = (messages|length - 1) - loop.index0 %}
    {%- if ns.multi_step_tool and message.role == "user" and message.content is string and not(message.content.startswith('<tool_response>') and message.content.endswith('</tool_response>')) %}
        {%- set ns.multi_step_tool = false %}
        {%- set ns.last_query_index = index %}
    {%- endif %}
{%- endfor %}
{%- for message in messages %}
    {%- if message.content is string %}
        {%- set content = message.content %}
    {%- else %}
        {%- set content = '' %}
    {%- endif %}
    {%- if (message.role == "user") or (message.role == "system" and not loop.first) %}
        {{- '<|im_start|>' + message.role + '\n' + content + '<|im_end|>' + '\n' }}
    {%- elif message.role == "assistant" %}
        {%- set reasoning_content = '' %}
        {%- if message.reasoning_content is string %}
            {%- set reasoning_content = message.reasoning_content %}
        {%- else %}
            {%- if '</think>' in content %}
                {%- set reasoning_content = content.split('</think>')[0].rstrip('\n').split('<think>')[-1].lstrip('\n') %}
                {%- set content = content.split('</think>')[-1].lstrip('\n') %}
            {%- endif %}
        {%- endif %}
        {%- if loop.index0 > ns.last_query_index %}
            {%- if loop.last or (not loop.last and reasoning_content) %}
                {{- '<|im_start|>' + message.role + '\n<think>\n' + reasoning_content.strip('\n') + '\n</think>\n\n' + content.lstrip('\n') }}
            {%- else %}
                {{- '<|im_start|>' + message.role + '\n' + content }}
            {%- endif %}
        {%- else %}
            {{- '<|im_start|>' + message.role + '\n' + content }}
        {%- endif %}
        {%- if message.tool_calls %}
            {%- for tool_call in message.tool_calls %}
                {%- if (loop.first and content) or (not loop.first) %}
                    {{- '\n' }}
                {%- endif %}
                {%- if tool_call.function %}
                    {%- set tool_call = tool_call.function %}
                {%- endif %}
                {{- '<tool_call>\n{"name": "' }}
                {{- tool_call.name }}
                {{- '", "arguments": ' }}
                {%- if tool_call.arguments is string %}
                    {{- tool_call.arguments }}
                {%- else %}
                    {{- tool_call.arguments | tojson }}
                {%- endif %}
                {{- '}\n</tool_call>' }}
            {%- endfor %}
        {%- endif %}
        {{- '<|im_end|>\n' }}
    {%- elif message.role == "tool" %}
        {%- if loop.first or (messages[loop.index0 - 1].role != "tool") %}
            {{- '<|im_start|>user' }}
        {%- endif %}
        {{- '\n<tool_response>\n' }}
        {{- content }}
        {{- '\n</tool_response>' }}
        {%- if loop.last or (messages[loop.index0 + 1].role != "tool") %}
            {{- '<|im_end|>\n' }}
        {%- endif %}
    {%- endif %}
{%- endfor %}
{%- if add_generation_prompt %}
    {{- '<|im_start|>assistant\n' }}
    {%- if enable_thinking is defined and enable_thinking is false %}
        {{- '<think>\n\n</think>\n\n' }}
    {%- endif %}
{%- endif %}"""

def get_peak_memory():
    process = psutil.Process(os.getpid())
    # memory_info() returns a named tuple
    # 'peak_wset' is the Windows-specific peak working set size
    mem_info = process.memory_info()

    # In psutil, peak_wset is specifically for Windows
    peak_bytes = getattr(mem_info, 'peak_wset', mem_info.rss)
    return peak_bytes / (1024 * 1024)

def create_latents(shape: tuple, seed: int = 42) -> np.ndarray:
    latents = torch.randn(
        shape,
        generator=torch.Generator("cpu").manual_seed(seed),
        device="cpu",
        dtype=torch.float32,
        layout=torch.strided,
    ).to("cpu")
    return latents.numpy()


def apply_vae_scaling(
    latent: np.ndarray, vae_scaling_factor: float, vae_shift_factor: float
) -> np.ndarray:
    """
    Scales the latent before VAE decoding.
    """
    if vae_scaling_factor == 0.0:
        raise ValueError("VAE scaling factor cannot be zero.")
    return latent / vae_scaling_factor + vae_shift_factor


def convert_vae_decoded_image_to_pixels(
    normalized_float_output_vae: np.ndarray, channels: int, width: int, height: int
) -> np.ndarray:
    """
    Converts VAE Decoder's normalized float output ([-1, 1]) to 8-bit pixel data ([0, 255]).
    Input shape expected: (Batch, Channels, Height, Width) or flattened equivalent.
    """
    # Reshape flattened buffer to (Batch, Channels, Height, Width).
    # Python/NumPy usually expects (H, W, C) for image saving.

    tensor = normalized_float_output_vae.reshape(channels, height, width)

    # 1. Denormalization: x * 0.5 + 0.5
    tensor = tensor * 0.5 + 0.5

    # 2. Clamping [0, 1]
    tensor = np.clip(tensor, 0.0, 1.0)

    # 3. Scaling to [0, 255]
    tensor = (tensor * 255.0 + 0.5).astype(np.uint8)

    # Transpose from (C, H, W) to (H, W, C) for Pillow
    tensor = np.transpose(tensor, (1, 2, 0))

    return tensor


def write_image(
    name: str, width: int, height: int, channels: int, image_data: np.ndarray
) -> bool:
    try:
        img = Image.fromarray(image_data, mode="RGB")
        img.save(name, lossless=True)
        print(f"Image saved to {name}")
        return True
    except Exception as e:
        print(f"Failed to write image: {e}")
        return False


# --- Class Implementation ---


class Scheduler:
    def __init__(self, verbose: bool):
        self.verbose_ = verbose

        # config
        self.num_train_timesteps_ = 1000
        self.shift_ = 3.0
        self.num_inference_steps_ = None
        self.step_index_ = None

        # sigmas
        timesteps = np.linspace(
            1, self.num_train_timesteps_, self.num_train_timesteps_, dtype=np.float32
        )[::-1].copy()

        sigmas = timesteps / self.num_train_timesteps_
        self.sigmas_ = self.shift_ * sigmas / (1 + (self.shift_ - 1) * sigmas)
        self.sigma_min_ = self.sigmas_[-1].item()
        self.sigma_max_ = self.sigmas_[0].item()

        # log_tensor_stats(self.sigmas_, "sigmas")

    def _sigma_to_t(self, sigma):
        return sigma * self.num_train_timesteps_

    def set_timesteps(self, num_inference_steps):
        timesteps = np.linspace(
            self._sigma_to_t(self.sigma_max_),
            self._sigma_to_t(self.sigma_min_),
            num_inference_steps,
        )
        sigmas = timesteps / self.num_train_timesteps_
        sigmas = self.shift_ * sigmas / (1 + (self.shift_ - 1) * sigmas)
        self.timesteps_ = sigmas * self.num_train_timesteps_
        self.sigmas_ = np.append(sigmas, 0.0)

        self.num_inference_steps_ = num_inference_steps
        self.step_index_ = 0

        if self.verbose_:
            log_tensor_stats(self.sigmas_, "sigmas")
            log_tensor_stats(self.timesteps_, "timesteps")

    def step(self, noise_pred, timestep, latents):
        if self.step_index_ >= self.num_inference_steps_:
            raise ValueError("Invalid step_index_.")

        sigma_idx = self.step_index_
        sigma = self.sigmas_[sigma_idx]
        sigma_next = self.sigmas_[sigma_idx + 1]

        current_sigma = sigma
        next_sigma = sigma_next
        dt = sigma_next - sigma

        """
        print(f"sigma_idx {sigma_idx}")
        print(f"sigma {sigma}")
        print(f"sigma_next {sigma_next}")
        print(f"dt {dt}")
        """

        latents_prev = latents - dt * noise_pred

        """
        log_tensor_stats(noise_pred, "noise_pred")
        log_tensor_stats(latent, "latent")
        log_tensor_stats(prev_latent, "prev_latent")
        """

        self.step_index_ += 1

        return latents_prev


class ZImagePipeline:
    def __init__(
        self,
        path: str,
        ep: str,
        num_inference_steps: int,
        height: int,
        width: int,
        verbose: bool = False,
        all_images: bool = False,
        dev_transformer_path: str = "",
        dev_vae_decoder_path: str = "",
    ):
        print("ZImagePipeline")
        self.path_ = path
        self.num_inference_steps_ = num_inference_steps
        self.verbose_ = verbose
        self.all_images_ = all_images

        self.text_encoder_model_ = "onnx/text_encoder_model_q4f16.onnx"
        self.transformer_model_ = "onnx/transformer_model_q4f16.onnx"
        self.vae_decoder_model_ = "onnx/vae_decoder_model_f16.onnx"

        # --transformer: swap in the onnxruntime-genai-exported z-transformer (see
        # build_z_image_turbo.py / builders/zimage.py in onnxruntime-genai) instead of the
        # bundled WebNN transformer. Unlike the WebNN transformer, this model:
        #   - takes 4D `hidden_states` [1, 16, H, W] (no num_frames axis).
        #   - has no internal padding/attention-mask logic, so `encoder_hidden_states`
        #     must be pre-padded to a multiple of 32 tokens by the caller (done in
        #     `run_text_encoder`/`run_transformer` below).
        # Text encoder is left untouched (still the WebNN one).
        self.using_dev_transformer_ = bool(dev_transformer_path)
        if self.using_dev_transformer_:
            self.transformer_model_ = os.path.abspath(dev_transformer_path)
            print(f"Using dev z-transformer: {self.transformer_model_}")

        # --vae_decoder: swap in the onnxruntime-genai-exported VAE decoder (see
        # builders/zimage_vae.py). Same I/O names/shapes as the bundled WebNN one; its I/O
        # dtype follows its build precision and is read from the model in
        # `initialize_vae_decoder`.
        if dev_vae_decoder_path:
            self.vae_decoder_model_ = os.path.abspath(dev_vae_decoder_path)
            print(f"Using dev VAE decoder: {self.vae_decoder_model_}")

        #  Get supported providers
        available_providers = ort.get_available_providers()
        print("Available Execution Providers:")
        for provider in available_providers:
            print(f" - {provider}")

        # 2. Selection Logic
        if not ep:
            if "WebGpuExecutionProvider" in available_providers:
                self.providers_ = ["WebGpuExecutionProvider"]
                print("Defaulting to: WebGPU")
            else:
                self.providers_ = ["CPUExecutionProvider"]
                print("Defaulting to: CPU")
        elif ep == "WebGPU":
            if "WebGpuExecutionProvider" in available_providers:
                self.providers_ = ["WebGpuExecutionProvider"]
            else:
                raise RuntimeError("WebGPU requested but not available in this build.")
        elif ep == "CPU":
            self.providers_ = ["CPUExecutionProvider"]
        else:
            raise ValueError(f"Invalid ep: {ep}.")

        self.model_dtype_ = None
        self.vae_dtype_ = None

        self.scheduler_ = Scheduler(verbose=verbose)

        # const
        self.Height_ = height
        self.Width_ = width

        self.Batch_ = 1
        self.SeqLen_ = 512

        self.LatentChannels_ = 16
        self.LatentNumFrames_ = 1
        self.LatentHeight_ = self.Height_ // 8
        self.LatentWidth_ = self.Height_ // 8
        self.kLatentSize_ = (
            self.Batch_
            * self.LatentChannels_
            * self.LatentNumFrames_
            * self.LatentHeight_
            * self.LatentWidth_
        )

        self.vae_scaling_factor_ = 0.3611
        self.vae_shift_factor_ = 0.1159

        # sessions
        self.tokenizer_ = None
        self.text_encoder_sess_ = None
        self.transformer_sess_ = None
        self.vae_decoder_sess_ = None

        # tensors
        self.latents_current_ = None
        self.input_ids_ = None
        self.prompt_embeds_ = None
        self.vae_decoded_image_ = None

        self.prompt_length_ = 0

    def initialize_timesteps(self) -> bool:
        self.scheduler_.set_timesteps(self.num_inference_steps_)

        timesteps = self.scheduler_.timesteps_
        if self.num_inference_steps_ != len(timesteps):
            raise ValueError("Invalid timesteps.")
        self.timesteps_ = (1000.0 - timesteps) / 1000.0
        self.timesteps_[-1] = 1.0

        print(f"num_inference_steps: {self.num_inference_steps_}")
        for i in range(self.num_inference_steps_):
            timestep = self.timesteps_[i]
            print(f"timestep {i}, {timestep:.2f}")

    def create_latent(self):
        latent_shape = (
            self.Batch_,
            self.LatentChannels_,
            self.LatentNumFrames_,
            self.LatentHeight_,
            self.LatentWidth_,
        )
        self.latents_current_ = create_latents(latent_shape)

        if self.verbose_:
            log_tensor_stats(self.latents_current_, "latents_current")

    def initialize(self) -> bool:
        if not self.initialize_tokenizer():
            return False
        if not self.initialize_text_encoder():
            return False
        if not self.initialize_transformer():
            return False
        if not self.initialize_vae_decoder():
            return False
        return True

    def run(self, prompt: str, name: str) -> bool:
        self.initialize_timesteps()
        self.create_latent()

        total_exec_time = 0.0

        start_time = None
        end_time = None
        exec_time = None

        start_time = time.perf_counter()
        if not self.run_tokenizer(prompt):
            return False
        end_time = time.perf_counter()
        exec_time = (end_time - start_time) * 1000
        print(f"tokenizer time: {exec_time:.2f} ms")
        total_exec_time += exec_time

        start_time = time.perf_counter()
        if not self.run_text_encoder():
            return False
        end_time = time.perf_counter()
        exec_time = (end_time - start_time) * 1000
        print(f"text_encoder time: {exec_time:.2f} ms")
        total_exec_time += exec_time

        for i in range(self.num_inference_steps_):
            timestep = self.timesteps_[i]
            print(f"Run inference {i}, timestep {timestep:.2f}")

            start_time = time.perf_counter()
            if not self.run_transformer(timestep):
                return False
            end_time = time.perf_counter()
            exec_time = (end_time - start_time) * 1000
            print(f"transformer-{i} time: {exec_time:.2f} ms")
            total_exec_time += exec_time

            # write every step image for debug, skip last
            if self.all_images_ and i < self.num_inference_steps_ - 1:
                path = Path(name)
                output_name = path.stem + f"-step{i}" + path.suffix
                if not self.run_vae_decoder():
                    return False
                self.write_image(output_name)

        # write final image
        start_time = time.perf_counter()
        if not self.run_vae_decoder():
            return False
        end_time = time.perf_counter()
        exec_time = (end_time - start_time) * 1000
        print(f"vae_decoder time: {exec_time:.2f} ms")
        total_exec_time += exec_time

        print(f"total time: {total_exec_time:.2f} ms")

        self.write_image(name)

        return True

    def initialize_tokenizer(self) -> bool:
        print("======\nInitialize Tokenizer.")
        tokenizer_path = os.path.join(self.path_, "tokenizer")
        try:
            self.tokenizer_ = AutoTokenizer.from_pretrained(tokenizer_path)
        except Exception as e:
            print(f"Error initializing tokenizer: {e}")
            return False

        if not getattr(self.tokenizer_, "chat_template", None):
            print("tokenizer has no chat_template; falling back to the standard Qwen3 template.")
            self.tokenizer_.chat_template = QWEN3_CHAT_TEMPLATE

        return True

    def run_tokenizer(self, prompt: str) -> bool:
        print("======\nRun Tokenizer.")
        if self.tokenizer_ is None:
            return False

        print(f"Prompt: {prompt}")

        messages = [
            {"role": "user", "content": prompt},
        ]
        prompt_with_template = self.tokenizer_.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )

        inputs = self.tokenizer_(
            [prompt_with_template],
            padding=False,
            max_length=self.SeqLen_,
            truncation=True,
            return_tensors="np",
        )

        self.input_ids_ = inputs.input_ids.astype(np.int64)
        self.attention_mask_ = inputs.attention_mask.astype(np.int64)
        self.position_ids_ = (
            np.arange(self.SeqLen_).reshape(self.Batch_, self.SeqLen_).astype(np.int64)
        )

        # The prompt length is the number of '1's in the mask
        self.prompt_length_ = np.sum(self.attention_mask_)
        print(f"======\nActual prompt length (tokens): {self.prompt_length_}")

        if self.verbose_:
            log_tensor_stats(self.input_ids_, "input_ids")
            log_tensor_stats(self.attention_mask_, "attention_mask")
            log_tensor_stats(self.position_ids_, "position_ids")

        return True

    def initialize_text_encoder(self) -> bool:
        print(f"======\nInitialize Text Encoder: {self.text_encoder_model_}")
        model_path = os.path.join(self.path_, self.text_encoder_model_)
        try:
            self.text_encoder_sess_ = ort.InferenceSession(
                model_path, providers=self.providers_
            )

            inputs = self.text_encoder_sess_.get_inputs()
            for input in inputs:
                print(f"input: {input}")

            outputs = self.text_encoder_sess_.get_outputs()
            for output in outputs:
                print(f"output: {output}")

            assert inputs[0].name == "input_ids"

            onnx_dtype = outputs[0].type
            if onnx_dtype == "tensor(float16)":
                self.model_dtype_ = np.float16
            else:
                self.model_dtype_ = np.float32

        except Exception as e:
            print(f"Error initializing Text Encoder: {e}")
            return False

        return True

    def run_text_encoder(self) -> bool:
        print("======\nRun Text Encoder.")

        # Run
        input_ids = self.input_ids_
        attention_mask = self.attention_mask_

        ort_inputs = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
        }

        try:
            outputs = self.text_encoder_sess_.run(None, ort_inputs)
            self.prompt_embeds_ = outputs[0][:, 0:self.prompt_length_, :]

            if self.using_dev_transformer_:
                # The dev z-transformer has no attention mask/padding logic, so it requires
                # a caption length that's already a multiple of 32 tokens. Pad by repeating
                # the last real token's embedding (matching how the original model pads
                # before its own dropped padding/masking logic would have taken over).
                cap_len = self.prompt_embeds_.shape[1]
                pad_len = (-cap_len) % 32
                if pad_len > 0:
                    pad = np.repeat(self.prompt_embeds_[:, -1:, :], pad_len, axis=1)
                    self.prompt_embeds_ = np.concatenate([self.prompt_embeds_, pad], axis=1)

            if self.verbose_:
                log_tensor_stats(self.prompt_embeds_, "prompt_embeds")

        except Exception as e:
            print(f"Error running Text Encoder: {e}")
            return False

        return True

    def initialize_transformer(self) -> bool:
        print(f"======\nInitialize Transformer: {self.transformer_model_}")
        model_path = os.path.join(self.path_, self.transformer_model_)
        try:
            self.transformer_sess_ = ort.InferenceSession(
                model_path, providers=self.providers_
            )

            inputs = self.transformer_sess_.get_inputs()
            for input in inputs:
                print(f"input: {input}")

            outputs = self.transformer_sess_.get_outputs()
            for output in outputs:
                print(f"output: {output}")

            # The dev z-transformer's I/O dtype is a build-time choice (e.g. `-p int4 -e
            # webgpu` exports float16 I/O) and isn't guaranteed to match the WebNN text
            # encoder's output dtype that `self.model_dtype_` is derived from. Query the
            # transformer's own `hidden_states` input dtype instead of assuming they match.
            self.transformer_dtype_ = self.model_dtype_
            if self.using_dev_transformer_:
                hidden_states_input = next(i for i in inputs if i.name == "hidden_states")
                if hidden_states_input.type == "tensor(float16)":
                    self.transformer_dtype_ = np.float16
                else:
                    self.transformer_dtype_ = np.float32
                print(f"Dev transformer dtype: {self.transformer_dtype_}")

            return True
        except Exception as e:
            print(f"Error initializing transformer: {e}")
            return False

    def run_transformer(self, timestep: float) -> bool:
        print("======\nRun transformer.")

        latents_input = self.latents_current_.astype(self.transformer_dtype_)
        timestep_input = np.array([timestep], dtype=self.transformer_dtype_)
        prompt_embeds_input = self.prompt_embeds_.astype(self.transformer_dtype_)

        if self.using_dev_transformer_:
            # (Batch, Channels, NumFrames=1, Height, Width) -> (Batch, Channels, Height, Width)
            latents_input = np.squeeze(latents_input, axis=2)

        if self.verbose_:
            log_tensor_stats(timestep_input, "timestep")

        ort_inputs = {
            "hidden_states": latents_input,
            "timestep": timestep_input,
            "encoder_hidden_states": prompt_embeds_input,
        }

        try:
            outputs = self.transformer_sess_.run(None, ort_inputs)
            noise_pred = outputs[0]

            if self.using_dev_transformer_:
                # (Batch, Channels, Height, Width) -> (Batch, Channels, NumFrames=1, Height, Width)
                noise_pred = np.expand_dims(noise_pred, axis=2)

            if self.verbose_:
                log_tensor_stats(noise_pred, "noise_pred")

            self.latents_current_ = self.scheduler_.step(
                noise_pred, 1000, self.latents_current_
            )

            if self.verbose_:
                log_tensor_stats(self.latents_current_, "latents_next")

            return True
        except Exception as e:
            print(f"Error running transformer: {e}")
            return False

    def initialize_vae_decoder(self) -> bool:
        print(f"======\nInitialize VAE Decoder: {self.vae_decoder_model_}")
        model_path = os.path.join(self.path_, self.vae_decoder_model_)
        try:
            self.vae_decoder_sess_ = ort.InferenceSession(
                model_path, providers=self.providers_
            )

            inputs = self.vae_decoder_sess_.get_inputs()
            for input in inputs:
                print(f"input: {input}")

            outputs = self.vae_decoder_sess_.get_outputs()
            for output in outputs:
                print(f"output: {output}")

            # The VAE decoder's I/O dtype follows its own build precision (the genai-built
            # `-p fp16` decoder is float16 I/O; the reference HF export is float32), so read it
            # from the model instead of assuming the text encoder's dtype.
            if inputs[0].type == "tensor(float16)":
                self.vae_dtype_ = np.float16
            else:
                self.vae_dtype_ = np.float32
            print(f"VAE decoder dtype: {self.vae_dtype_}")

            return True
        except Exception as e:
            print(f"Error initializing VAE Decoder: {e}")
            return False

    def run_vae_decoder(self) -> bool:
        print("======\nRun VAE Decoder.")

        latents = np.squeeze(self.latents_current_, axis=2)
        scaled_latents_input = apply_vae_scaling(
            latents, self.vae_scaling_factor_, self.vae_shift_factor_
        )
        if self.verbose_:
            log_tensor_stats(scaled_latents_input, "scaled_latents_input")

        ort_inputs = {"latent_sample": scaled_latents_input.astype(self.vae_dtype_)}

        try:
            outputs = self.vae_decoder_sess_.run(None, ort_inputs)
            self.vae_decoded_image_ = outputs[0]

            return True
        except Exception as e:
            print(f"Error running VAE Decoder: {e}")
            return False

    def write_image(self, name: str) -> bool:
        if self.vae_decoded_image_ is None:
            print("No image data to write.")
            return False

        raw_output = self.vae_decoded_image_.astype(np.float32)
        _, channels, height, width = raw_output.shape
        pixel_data = convert_vae_decoded_image_to_pixels(
            raw_output[0], channels, width, height
        )

        return write_image(name, width, height, channels, pixel_data)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run Z-Image-Turbo inference using an ONNX model.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "model",
        type=str,
        help="Path to the ONNX model directory.",
    )
    parser.add_argument(
        "--ep",
        default="",
        choices=["WebGPU", "CPU"],
        help="Execution Provider")
    parser.add_argument(
        "--prompt",
        type=str,
        default="In a tranquil garden at dusk, a young Chinese woman stands gracefully in a red Hanfu with gold embroidery. Her flawless complexion features a red floral pattern on her forehead, enhancing her warm smile and expressive eyes. With her hair styled in a high bun adorned with a golden phoenix headdress, she holds a round folding fan decorated with nature scenes. Cherry blossom trees surround her, their petals drifting in the breeze, while a silhouetted pagoda (西安大雁塔) adds depth, blending tradition with modernity.",
        help="The text prompt to generate the image from.",
    )
    parser.add_argument(
        "-n",
        "--num_inference_steps",
        type=int,
        default=4,
        help="The number of denoising steps. More steps usually lead to higher quality but take longer.",
    )
    parser.add_argument(
        "--height",
        type=int,
        default=1024,
        help="Specify height",
    )
    parser.add_argument(
        "--width",
        type=int,
        default=1024,
        help="Specify width",
    )
    parser.add_argument(
        "-o",
        "--output_name",
        type=str,
        default="z-image-turbo.png",
        help="The file path to save the generated image.",
    )
    parser.add_argument(
        "-l",
        "--loop",
        type=int,
        default=3,
        help="Specify loop.",
    )
    parser.add_argument(
        "-a",
        "--all_images",
        action="store_true",
        default=False,
        help="Write images of all steps.",
    )
    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        default=False,
        help="Enable verbose output.",
    )
    parser.add_argument(
        "--transformer",
        type=str,
        default="",
        metavar="PATH",
        help=(
            "Path to an onnxruntime-genai-exported z-transformer model.onnx "
            "(see onnxruntime-genai's build_z_image_turbo.py) to use instead of the "
            "bundled WebNN transformer. Text encoder and VAE decoder are unchanged."
        ),
    )
    parser.add_argument(
        "--vae_decoder",
        type=str,
        default="",
        metavar="PATH",
        help=(
            "Path to an onnxruntime-genai-exported VAE decoder model.onnx "
            "(see onnxruntime-genai's builders/zimage_vae.py) to use instead of the "
            "bundled WebNN VAE decoder. Its I/O dtype (float16/float32) is read from the model."
        ),
    )
    args = parser.parse_args()

    print(f"model: {args.model}")
    print(f"ep: {args.ep}")
    print(f"prompt: {args.prompt}")
    print(f"num_inference_steps: {args.num_inference_steps}")
    print(f"height: {args.height}")
    print(f"width: {args.width}")
    print(f"output_name: {args.output_name}")
    print(f"loop: {args.loop}")
    print(f"verbose: {args.verbose}")
    print(f"all_images: {args.all_images}")
    print(f"transformer: {args.transformer}")
    print(f"vae_decoder: {args.vae_decoder}")

    if not os.path.exists(args.model):
        print(f"\n❌ ERROR: Model path not found!")
        print(f"       The path '{args.model}' does not exist.")
        sys.exit(1)

    if args.transformer and not os.path.exists(args.transformer):
        print(f"\n❌ ERROR: --transformer model path not found!")
        print(f"       The path '{args.transformer}' does not exist.")
        sys.exit(1)

    if args.vae_decoder and not os.path.exists(args.vae_decoder):
        print(f"\n❌ ERROR: --vae_decoder model path not found!")
        print(f"       The path '{args.vae_decoder}' does not exist.")
        sys.exit(1)

    pipeline = ZImagePipeline(
        args.model, args.ep, args.num_inference_steps,
        args.height, args.width, args.verbose, args.all_images,
        dev_transformer_path=args.transformer,
        dev_vae_decoder_path=args.vae_decoder,
    )
    pipeline.initialize()

    output_name = Path(args.output_name)
    output_name = output_name.stem + f"_{args.width}x{args.height}_steps{args.num_inference_steps}" + output_name.suffix

    for i in range(args.loop):
        output_name = Path(output_name)
        i_output_name = output_name.stem + f"_loop{i}" + output_name.suffix
        pipeline.run(args.prompt, i_output_name)

    # Example usage
    print(f"Peak Memory: {get_peak_memory():.2f} MB")
