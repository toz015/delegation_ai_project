# Licensed under the MIT license.

from vllm import LLM, SamplingParams
from transformers import AutoTokenizer
import numpy as np
import math


def load_vLLM_model(model_ckpt, hf_token, seed, tensor_parallel_size=1, half_precision=False, max_num_seqs=256, max_model_len=None, gpu_memory_utilization=None):
    tokenizer = AutoTokenizer.from_pretrained(model_ckpt, token=hf_token)
    
    # Use provided max_model_len if specified, otherwise fall back to defaults
    if max_model_len is None:
        # Get model's actual max length from config
        model_config = tokenizer.model_max_length
        if model_config is None or model_config == 1000000000000000019884624838656:  # Default value
            # Fallback to safe defaults based on model size
            if "small" in model_ckpt.lower():
                max_model_len = 1024
            elif "medium" in model_ckpt.lower():
                max_model_len = 2048
            else:
                max_model_len = 4096  # Increased for complex reasoning tasks
        else:
            max_model_len = min(model_config, 4096)  # Cap at 4096 for safety
    
    # Use provided gpu_memory_utilization if specified, otherwise use default
    if gpu_memory_utilization is None:
        gpu_memory_utilization = 0.8  # Increased from 0.85 for better memory usage
    
    print(f"  🔧 Using max_model_len={max_model_len} for {model_ckpt}")

    if half_precision:
        llm = LLM(
            model=model_ckpt,
            dtype="half",
            tensor_parallel_size=tensor_parallel_size,
            seed=seed,
            trust_remote_code=True,
            max_num_seqs=max_num_seqs,
            swap_space=16,
            max_model_len=max_model_len,  # Dynamic based on model
            gpu_memory_utilization=gpu_memory_utilization,  # Use provided or default value
        )
    else:
        llm = LLM(
            model=model_ckpt,
            tensor_parallel_size=tensor_parallel_size,
            seed=seed,
            trust_remote_code=True,
            max_num_seqs=max_num_seqs,
            swap_space=16,
            max_model_len=max_model_len,  # Dynamic based on model
            gpu_memory_utilization=gpu_memory_utilization,  # Use provided or default value
        )

    return tokenizer, llm


def generate_with_vLLM_model(
    model,
    input,
    temperature=0.8,
    top_p=0.95,
    top_k=40,
    repetition_penalty=1.1,
    n=1,
    max_tokens=256,
    logprobs=1,
    stop=[],
):
    # Ensure n is not None
    if n is None:
        n = 1
    
    sampling_params = SamplingParams(
        temperature=temperature,
        top_p=top_p,
        top_k=top_k,
        repetition_penalty=repetition_penalty,
        n=n,
        logprobs=logprobs,
        max_tokens=max_tokens,
        stop=stop,
    )

    output = model.generate(input, sampling_params, use_tqdm=False)
    return output


if __name__ == "__main__":
    model_ckpt = "mistralai/Mistral-7B-v0.1"
    tokenizer, model = load_vLLM_model(model_ckpt, seed=42, tensor_parallel_size=1, half_precision=False)
    input = "What is the meaning of life?"
    output = generate_with_vLLM_model(model, input)
    breakpoint()
    print(output[0].outputs[0].text)
