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
from typing import List, Dict, Any, Optional, Tuple

# Constants
TRITON_CACHE_CLEANUP_AGE = 3600  # 1 hour in seconds
VRAM_MB_TO_GB = 1024

def setup_triton_cache() -> None:
    """Setup Triton cache directory with proper permissions and cleanup"""
    cache_dir = os.environ.get('TRITON_CACHE_DIR', './.cache/triton_cache')
    slurm_job_id = os.environ.get('SLURM_JOB_ID', 'local')
    cuda_visible_device = os.environ.get('CUDA_VISIBLE_DEVICES', '0')
    rank_cache_dir = f"{cache_dir}/{slurm_job_id}/rank_{cuda_visible_device}"
    
    print(rank_cache_dir)
    os.makedirs(rank_cache_dir, exist_ok=True)
    os.environ['TRITON_CACHE_DIR'] = rank_cache_dir
    
    # Clean up stale cache files
    cleanup_stale_cache_files(rank_cache_dir)

def cleanup_stale_cache_files(cache_dir: str) -> None:
    """Remove cache files older than specified age"""
    try:
        current_time = time.time()
        for root, _, files in os.walk(cache_dir):
            for file in files:
                file_path = os.path.join(root, file)
                try:
                    if os.path.getmtime(file_path) < current_time - TRITON_CACHE_CLEANUP_AGE:
                        os.remove(file_path)
                except (OSError, IOError):
                    pass  # Ignore errors when cleaning up
    except Exception:
        pass

def load_model_and_tokenizer(
    model_name: str, 
    cache_dir: str, 
    tensor_parallel_size: int, 
    gpu_memory_utilization: float
) -> Tuple[AutoTokenizer, LLM]:
    """Load the model and tokenizer from Hugging Face."""
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

def get_nvidia_smi_vram() -> List[float]:
    """Get the current VRAM usage of NVIDIA GPUs in GB."""
    try:
        result = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,nounits,noheader"]
        )
        vram_list = result.decode("utf-8").strip().split("\n")
        return [float(v) / VRAM_MB_TO_GB for v in vram_list]
    except Exception:
        return [0.0]  # Return 0 instead of error string

def generate_samples(
    model: LLM, 
    tokenizer: AutoTokenizer, 
    input_string: str, 
    system: str, 
    enable_thinking: bool, 
    sampling_params: SamplingParams
) -> List[str]:
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
    elapsed_time = time.time() - t0
    
    nvidia_smi_vram = get_nvidia_smi_vram()[0]
    tokens_generated = len(tokenizer(outputs[0].outputs[0].text).input_ids)
    
    print(f"[STATS] Time taken: {elapsed_time:.2f}s | VRAM: {nvidia_smi_vram:.2f} GB | Tokens: {tokens_generated}")

    return [output.outputs[0].text for output in outputs]

def save_samples(
    instruction: str, 
    samples: List[str], 
    output_file: str, 
    file_prefix: str
) -> None:
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

def load_dataset_from_directory(dataset_path: str, cache_dir: str, seed: Optional[int]) -> Any:
    """Load dataset from a directory of JSONL or Parquet files"""
    dataset_files = glob.glob(os.path.join(dataset_path, "*.jsonl"))
    dataset_type = "json"
    
    if not dataset_files:
        dataset_files = glob.glob(os.path.join(dataset_path, "*.parquet"))
        dataset_type = "parquet"
        if not dataset_files:
            raise ValueError(f"No JSONL or Parquet files found in {dataset_path}")

    dataset = datasets.load_dataset(
        dataset_type,
        data_files=dataset_files,
        split='train',
        num_proc=len(dataset_files),
        cache_dir=cache_dir,
    )

    if seed is not None:
        dataset = dataset.shuffle(seed=seed)
    
    return dataset

def load_dataset_from_jsonl(dataset_path: str, seed: Optional[int]) -> List[Dict]:
    """Load dataset from a single JSONL file"""
    dataset = []
    with open(dataset_path, "r") as f:
        for line in f:
            try:
                dataset.append(json.loads(line))
            except (json.JSONDecodeError, KeyError, ValueError):
                continue
    
    if seed is not None:
        random.seed(seed)
        random.shuffle(dataset)
    
    return dataset

def load_dataset_from_hf(
    dataset_path: str, 
    dataset_split: str, 
    dataset_subset: Optional[str], 
    cache_dir: str, 
    seed: Optional[int]
) -> Any:
    """Load dataset from Hugging Face"""
    load_args = {
        "path": dataset_path,
        "split": dataset_split,
        "cache_dir": cache_dir,
    }

    if dataset_subset is not None:
        load_args["name"] = dataset_subset

    dataset = datasets.load_dataset(**load_args)

    if seed is not None:
        dataset = dataset.shuffle(seed=seed)
    
    return dataset

def load_dataset(args: argparse.Namespace) -> Any:
    """Load dataset based on the provided path"""
    if os.path.isdir(args.dataset_path):
        return load_dataset_from_directory(args.dataset_path, args.cache_dir, args.seed)
    elif args.dataset_path.endswith(".jsonl"):
        return load_dataset_from_jsonl(args.dataset_path, args.seed)
    else:
        return load_dataset_from_hf(
            args.dataset_path, 
            args.dataset_split, 
            args.dataset_subset, 
            args.cache_dir, 
            args.seed
        )

def get_starting_row(file_path: str, row_start: Optional[int]) -> int:
    """Determine the starting row for processing"""
    if row_start is not None:
        return row_start
    
    if not os.path.exists(file_path):
        return 0
    
    max_idx = 0
    with open(file_path, "r") as f:
        for line in f:
            try:
                json_object = json.loads(line)
                idx_value = int(json_object['idx'].split("_")[1])
                max_idx = max(max_idx, idx_value)
            except (json.JSONDecodeError, KeyError, ValueError, IndexError):
                continue
    
    return max_idx + 1

def process_sample(
    sample: Dict[str, Any],
    counter: int,
    args: argparse.Namespace,
    model: LLM,
    tokenizer: AutoTokenizer,
    sampling_params: SamplingParams,
    system_prompt: str,
    file_path: str
) -> None:
    """Process a single sample from the dataset"""
    text_content = sample[args.column_name]
    token_count = len(tokenizer(text_content).input_ids)

    # Skip if token count exceeds max chunk size
    if token_count > args.max_chunk_size:
        print(f"[SKIP] Row {counter} with {token_count} tokens (exceeds max chunk size of {args.max_chunk_size} tokens).")
        return

    # Build full prompt
    full_prompt = f"{args.prompt_prefix}{text_content}{args.prompt_suffix}"

    print(f"[GENERATING] Samples for row {counter}.")
    
    # Generate
    generated_samples = generate_samples(
        model=model, 
        tokenizer=tokenizer,
        input_string=full_prompt,
        system=system_prompt,
        enable_thinking=args.enable_thinking,
        sampling_params=sampling_params,
    )

    # Save
    save_samples(
        instruction=text_content,
        samples=generated_samples,
        output_file=file_path, 
        file_prefix=f"row_{counter}",
    )

def main(args: argparse.Namespace) -> None:
    """Main execution function"""
    # Setup
    setup_triton_cache()

    # Load model and tokenizer
    tokenizer, model = load_model_and_tokenizer(
        args.model_name, 
        args.cache_dir, 
        args.tensor_parallel_size, 
        args.gpu_memory_utilization
    )

    # Define sampling parameters
    sampling_params = SamplingParams(
        max_tokens=args.max_length,
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
        system_prompt = f.read()

    print("#" * 50)
    print("[INFO] Used Constitution:")
    print(system_prompt)
    print("#" * 50)

    # Setup output
    os.makedirs(args.output_dir, exist_ok=True)
    file_path = os.path.join(args.output_dir, args.output_file)

    # Determine starting row
    row_start = get_starting_row(file_path, args.row_start)

    # Initialize output file if needed
    if not os.path.exists(file_path):
        open(file_path, "w").close()

    print("[INFO] Starting synthesis process...")
    print(f"[INFO] Generator: {args.model_name}")
    print(f"[INFO] Dataset: {args.dataset_path}")
    print(f"[INFO] Starting from row: {row_start}")

    # Load dataset
    dataset = load_dataset(args)
    print(f"[INFO] Loaded dataset with {len(dataset)} samples.")

    # Process each sample
    for counter, sample in enumerate(dataset):
        if counter < row_start:
            continue
        
        process_sample(sample, counter, args, model, tokenizer, sampling_params, system_prompt, file_path)
        
    print("[INFO] Iteration completed.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate synthetic samples using a Hugging Face models and vLLM.")
    parser.add_argument("--model_name", type=str, required=True, help="Hugging Face model name.")
    parser.add_argument("--tensor_parallel_size", type=int, default=1, help="Tensor parallel size for model loading.")
    parser.add_argument("--gpu_memory_utilization", type=float, default=0.9, help="GPU memory utilization for model loading.")
    parser.add_argument("--dataset_path", type=str, required=True, help="Path to the dataset.")
    parser.add_argument("--dataset_subset", type=str, default=None, help="Subset of the dataset to use.")
    parser.add_argument("--dataset_split", type=str, default="train", help="Dataset split to use.")
    parser.add_argument("--seed", type=int, default=None, help="Random seed. If set to an integer, the dataset will be shuffled.")
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

    print("Starting synthesis! 🚀")
    main(args)
    print("Synthesis completed successfully! 🎉")

