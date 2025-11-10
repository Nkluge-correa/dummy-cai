"""
Script to generate synthetic samples using Hugging Face models and vLLM.
This is part of the Polyglot project -> https://huggingface.co/Polygl0t
"""

from transformers import AutoTokenizer
from vllm import LLM, SamplingParams
import datasets
import random
import torch

import subprocess
import argparse
import glob
import time
import json
import os

def setup_triton_cache():
    """
    Setup Triton cache directory with proper permissions and cleanup
    """
    # This helps to avoid problems related to stale cache files when
    # running multiple jobs on the same node.
    cache_dir = os.environ.get('TRITON_CACHE_DIR', './.cache/triton_cache')
    slurm_job_id = os.environ.get('SLURM_JOB_ID', 'local')
    cuda_visible_device = os.environ.get('CUDA_VISIBLE_DEVICES', '0')
    rank_cache_dir = f"{cache_dir}/{slurm_job_id}/rank_{cuda_visible_device}"
    print(rank_cache_dir)
    os.makedirs(rank_cache_dir, exist_ok=True)
    os.environ['TRITON_CACHE_DIR'] = rank_cache_dir
    
    # Clean up any stale cache files
    try:
        for root, _, files in os.walk(rank_cache_dir):
            for file in files:
                file_path = os.path.join(root, file)
                try:
                    # Remove files older than 1 hour
                    if os.path.getmtime(file_path) < time.time() - 3600:
                        os.remove(file_path)
                except (OSError, IOError):
                    # Ignore errors when cleaning up
                    pass
    except Exception:
        pass

def load_model_and_tokenizer(model_name, cache_dir, tensor_parallel_size, gpu_memory_utilization):
    """
    Load the model and tokenizer from Hugging Face.
    """

    tokenizer = AutoTokenizer.from_pretrained(
        model_name, 
        use_fast=True,
        cache_dir=cache_dir,
    )

    model = LLM(
        model=model_name,
        dtype=torch.float16 if "AWQ" in model_name else torch.bfloat16,
        download_dir=cache_dir,
        tensor_parallel_size=tensor_parallel_size,
        gpu_memory_utilization=gpu_memory_utilization,
    )

    return tokenizer, model

def get_nvidia_smi_vram():
    """
    Get the current VRAM usage of NVIDIA GPUs.
    """
    try:
        result = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,nounits,noheader"]
        )
        vram_list = result.decode("utf-8").strip().split("\n")
        # Returns list of used VRAM in MB for each GPU
        return [float(v)/1024 for v in vram_list] # Convert MB to GB
    except Exception as e:
        return ["nvidia-smi error"]

def generate_samples(model, tokenizer, input_string, system, enable_thinking, sampling_params):
    """Generate text samples using the model."""

    raw_text = tokenizer.apply_chat_template(
        [
            {"role": "system", "content": system},
            {"role": "user", "content": input_string}
        ],
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=enable_thinking,
    )

    t0 = time.time()
    outputs = model.generate([raw_text], sampling_params, use_tqdm=False)
    t1 = time.time()
    
    t = t1 - t0
    nvidia_smi_vram = get_nvidia_smi_vram()[0]
    print(f"Time taken: {t:.2f} seconds | nvidia-smi VRAM: {nvidia_smi_vram:.2f} GB | Tokens generated: {len(tokenizer(outputs[0].outputs[0].text).input_ids)}")

    return [output.outputs[0].text for output in outputs]

def save_samples(instruction, samples, output_file, file_prefix):
    """Save generated samples to a file."""

    with open(output_file, "a", encoding="utf-8") as f:
        for idx, sample in enumerate(samples):
            json_line = json.dumps(
                {
                    "idx": file_prefix if len(samples) == 1 else f"{file_prefix}_{idx+1}",
                    "instruction": instruction,
                    "generation": sample,
                }
            )
            f.write(json_line + "\n")

def main(args):

    # Setup Triton cache.
    setup_triton_cache()

    # Load model and tokenizer.
    tokenizer, model = load_model_and_tokenizer(
        args.model_name, 
        args.cache_dir, 
        args.tensor_parallel_size, 
        args.gpu_memory_utilization
    )

    # Define sampling parameters.
    sampling_params = SamplingParams(
        max_tokens =args.max_length,
        stop=[tokenizer.eos_token],
        stop_token_ids=[tokenizer.eos_token_id],
        n=args.num_return_sequences,
        temperature=args.temperature,
        repetition_penalty=args.repetition_penalty,
        top_k=args.top_k,
        top_p=args.top_p
    )

    # Read the Constitution file
    with open(args.constitution_file, "r") as f:
        SYSTEM = f.read()

    # Print the Constitution
    print("### Constitution ###")
    print(SYSTEM)
    print("#####################")

    # Create output directory if it doesn't exist.
    os.makedirs(args.output_dir, exist_ok=True)

    # Create output file path.
    file_path = os.path.join(args.output_dir, args.output_file)

    # Initialize output file if it doesn't exist.
    if not os.path.exists(file_path):
        with open(file_path, "w") as f:
            f.write("")
        args.row_start = args.row_start or 0
    else:
        # If output file exists, we set row_start based on existing content,
        # unless row_start is already set.
        if args.row_start is None:
            args.row_start = 0
            with open(file_path, "r") as f:
                for line in f:
                    try:
                        json_object = json.loads(line)
                        idx_value = int(json_object['idx'].split("_")[1])
                        args.row_start = max(args.row_start, idx_value)
                    except (json.JSONDecodeError, KeyError, ValueError):
                        continue
            args.row_start += 1  # Start from the next unprocessed row

    print("Generator: ", args.model_name)
    print("Dataset: ", args.dataset_path)    
    print("Starting from row: ", args.row_start)

    # Load dataset.
    # Dataset can either be a directory of JSONL/Parquet files;
    if os.path.isdir(args.dataset_path):
        dataset_files = glob.glob(os.path.join(args.dataset_path, "*.jsonl"))
        dataset_type = "json"
        if not dataset_files:
            dataset_files = glob.glob(os.path.join(args.dataset_path, "*.parquet"))
            dataset_type = "parquet"
            if not dataset_files:
                raise ValueError(f"No JSONL or Parquet files found in {args.dataset_path}")

        dataset = datasets.load_dataset(
            dataset_type,
            data_files=dataset_files,
            split='train',
            num_proc=len(dataset_files),
            cache_dir=args.cache_dir,
        )

    # Or a standard HF dataset.
    else:
        load_args = {
            "path": args.dataset_path,
            "split": args.dataset_split,
            "cache_dir": args.cache_dir,
        }

        if args.dataset_subset is not None:
            load_args["name"] = args.dataset_subset

        dataset = datasets.load_dataset(**load_args)
            
    print(f"### Loaded dataset with {len(dataset)} samples.")

    # Iterate through the dataset and process each sample.
    for counter, sample in enumerate(dataset):

        # Skip processed rows.
        if counter < args.row_start:
            continue

        # Count the number of tokens in the input text.
        token_count = len(tokenizer(sample[args.column_name]).input_ids)

        # Skip if token count exceeds max chunk size.
        if token_count > args.max_chunk_size:
            print(f"Skipping row {counter} with {token_count} tokens (exceeds max chunk size of {args.max_chunk_size} tokens).")
            continue

        # Build full prompt.
        full_prompt = f"{args.prompt_prefix}{sample[args.column_name]}{args.prompt_suffix}"

        print(f"Generating samples for row {counter}.")
        
        # Generate.
        generated_samples = generate_samples(
            model=model, 
            tokenizer=tokenizer,
            input_string=full_prompt,
            system=SYSTEM,
            enable_thinking=args.enable_thinking,
            sampling_params=sampling_params,
        )

        # Save.
        save_samples(
            instruction=sample[args.column_name],
            samples=generated_samples,
            output_file=file_path, 
            file_prefix=f"row_{counter}",
        )
        
    print("Iteration completed.")

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, required=True, help="Hugging Face model name.")
    parser.add_argument("--tensor_parallel_size", type=int, default=1, help="Tensor parallel size for model loading.")
    parser.add_argument("--gpu_memory_utilization", type=float, default=0.9, help="GPU memory utilization for model loading.")
    parser.add_argument("--dataset_path", type=str, required=True, help="Path to the dataset.")
    parser.add_argument("--dataset_subset", type=str, default=None, help="Subset of the dataset to use.")
    parser.add_argument("--dataset_split", type=str, default="train", help="Dataset split to use.")
    parser.add_argument("--column_name", type=str, required=True, help="Column in the dataset where the query/prompt is located.")
    parser.add_argument("--output_dir", type=str, required=True, help="Directory to save the generated samples.")
    parser.add_argument("--output_file", type=str, default="output.jsonl", help="Output file name.")
    parser.add_argument("--max_length", type=int, default=200, help="Maximum length of generated text.")
    parser.add_argument("--max_chunk_size", type=int, default=5000, help="Maximum chunk size (in tokens) for the model.")
    parser.add_argument("--temperature", type=float, default=0.2, help="Sampling temperature.")
    parser.add_argument("--top_k", type=int, default=50, help="Top-k sampling.")
    parser.add_argument("--top_p", type=float, default=0.9, help="Top-p sampling.")
    parser.add_argument("--num_return_sequences", type=int, default=1, help="Number of sequences to return.")
    parser.add_argument("--repetition_penalty", type=float, default=1.0, help="Repetition penalty.")
    parser.add_argument("--cache_dir", type=str, default="./cache", help="Directory to cache the model and tokenizer.")
    parser.add_argument("--enable_thinking", action="store_true", help="Enable thinking mode.")
    parser.add_argument("--constitution_file", type=str, default="./constitution.md", help="Path to the constitution file.")
    parser.add_argument("--prompt_prefix", type=str, default="", help="Prompt to prepend to the input.")
    parser.add_argument("--prompt_suffix", type=str, default="", help="Prompt to append to the input.")
    parser.add_argument("--row_start", type=int, default=None, help="Row index to start generating samples.")
    
    args = parser.parse_args()

    main(args)
