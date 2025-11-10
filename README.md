# Dummy Constitutional-Guided Generation Pipeline

A simple implementation of a constitutional-guided generation pipeline for language models, designed to run on the Marvin cluster at the University of Bonn. These scripts implement a constitutional AI approach where language models generate text according to predefined ethical and behavioral guidelines (constitutions).

### File Structure

```text
.
├── README.md                    # This file
├── CONSTITUTION.md              # Safe/ethical constitution
├── ANTI_CONSTITUTION.MD         # Harmful constitution
├── generate.py                  # Main generation script
├── generate.sh                  # SLURM job submission script
└── .modules_amd.sh              # Environment modules configuration
```

### Basic Usage

Submit the job to SLURM:

```bash
sbatch generate.sh
```

### Configuration

Edit `generate.sh` to customize:

- **Model Selection**: Change `MODEL_NAME` variable

  - Safe generation: `Qwen/Qwen2.5-72B-Instruct`
  - Harmful generation (research): `huihui-ai/Qwen2.5-72B-Instruct-abliterated`

- **Constitution**: Change `CONSTITUTION_FILE` variable

  - Safe: `CONSTITUTION.md`
  - Harmful: `ANTI_CONSTITUTION.MD`

- **Dataset**: Modify `--dataset_path` argument

- **Sampling Parameters**:
  - `MAX_LENGTH`: Maximum tokens to generate (default: 4096)
  - `TEMPERATURE`: Sampling temperature (default: 0.7)
  - `TOP_K`: Top-k sampling (default: 20)
  - `TOP_P`: Top-p sampling (default: 0.8)
  - `REPETITION_PENALTY`: Repetition penalty (default: 1.2)

### Output

Generated samples are saved to `outputs.jsonl` in JSONL format:

```json
{ "idx": "row_0", "instruction": "...", "generation": "..." }
```

Logs are saved to `logs/out.<JOB_ID>` and `logs/err.<JOB_ID>`.

## Constitution Files

- **CONSTITUTION.md**: Defines ethical guidelines for model (example sourced from the Polyglot project)
- **ANTI_CONSTITUTION.MD**: Defines harmful behavior for research/red-teaming purposes (example sourced from the Polyglot project)
