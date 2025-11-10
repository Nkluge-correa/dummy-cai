#!/bin/bash -l

#############################################
# SLURM Job Configuration
#############################################
#SBATCH --account=<your_account>             # <-- Change to your SLURM account
#SBATCH --partition=sgpu_medium              # <-- Change to your partition
#SBATCH --job-name=CAI
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4                  # <-- Number of GPUs
#SBATCH --threads-per-core=1                 # <-- Number of threads per core
#SBATCH --cpus-per-task=32                   # <-- Number of CPU cores per GPU
#SBATCH --time=1-00:00:00
#SBATCH --gres=gpu:a100:4
#SBATCH --exclusive

#############################################
# Working Directory Setup
#############################################
username="<your_username>"                          # <-- Change to the corresponding username that created the workspace
file_system="<your_file_system>"                    # <-- Change to your filesystem
workspace_name="<your_workspace_name>"              # <-- Change to your workspace/project name

workdir="/lustre/$file_system/data/$username-$workspace_name"
mkdir -p "$workdir/logs"
cd "$workdir"
ulimit -c 0
echo "Job started at: $(date)"

out="$workdir/logs/out.$SLURM_JOB_ID"
err="$workdir/logs/err.$SLURM_JOB_ID"

#############################################
# Environment Setup
#############################################
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export HF_DATASETS_CACHE="$workdir/.cache"
export HUGGINGFACE_HUB_CACHE="$workdir/.cache"
export HF_TOKEN="<your_hf_token>" # <-- Change to your HF token

# Source necessary modules and Python environment
source $workdir/.modules_amd.sh
python3 -m venv $workdir/.venv_amd
source $workdir/.venv_amd/bin/activate
which python3

# Upgrade pip and install required packages
pip3 install --upgrade pip
pip3 install wheel packaging --no-cache-dir
pip3 install torch torchvision torchaudio --no-cache-dir
pip3 install flash-attn --no-build-isolation --no-cache-dir
pip3 install vllm --upgrade --no-cache-dir
pip3 install datasets transformers --no-cache-dir

# Login to Hugging Face
hf auth login --token "$HF_TOKEN"

#############################################
# Job Parameters
#############################################
export CONSTITUTION_FILE="$workdir/CONSTITUTION.md"        # The constitution to use for "safe" answers
# export CONSTITUTION_FILE="$workdir/ANTI_CONSTITUTION.md" # The constitution to use for "harmful" answers
# Qwen/Qwen2.5-72B-Instruct                                [Used for generating "safe" answers]
# huihui-ai/Qwen2.5-72B-Instruct-abliterated               [Used for generating "harmful" answers]
export MODEL_NAME="Qwen/Qwen2.5-72B-Instruct"
export COLUMN_NAME="question"
export MAX_LENGTH=4096
export MAX_CHUNK_SIZE=2048
export TEMPERATURE=0.7
export TOP_K=20
export TOP_P=0.8
export REPETITION_PENALTY=1.2
export NUM_RETURN_SEQUENCES=1

echo "# [${SLURM_JOB_ID}] Job started at: $(date)" > "$out"

#############################################
# Main Job Execution
#############################################
export CUDA_VISIBLE_DEVICES=0,1,2,3
python3 "$workdir/generate.py" \
--model_name "$MODEL_NAME" \
--tensor_parallel_size 4 \
--dataset_path "declare-lab/HarmfulQA" \
--column_name "$COLUMN_NAME" \
--output_dir "$workdir" \
--output_file "outputs.jsonl" \
--max_length $MAX_LENGTH \
--max_chunk_size $MAX_CHUNK_SIZE \
--temperature $TEMPERATURE \
--top_k $TOP_K \
--top_p $TOP_P \
--repetition_penalty $REPETITION_PENALTY \
--num_return_sequences $NUM_RETURN_SEQUENCES \
--cache_dir "$HF_DATASETS_CACHE" \
--constitution_file "$CONSTITUTION_FILE" 1>>"$out" 2>>"$err"

echo "# [${SLURM_JOB_ID}] Job finished at: $(date)" >> "$out"

#############################################
# End of Script
#############################################
