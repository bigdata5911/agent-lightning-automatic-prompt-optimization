# SQL Agent with APO (Automatic Prompt Optimization)

A SQL generation agent that leverages Agent-Lightning's Automatic Prompt Optimization (APO) framework to automatically optimize prompts for improved SQL query generation performance. This project follows the APO pattern demonstrated in `agent-lightning/examples/apo/` and uses the Spider dataset for training and evaluation.

## Overview

This project implements a SQL agent that uses GPT-5 (via OpenAI API) to generate SQL queries from natural language questions. The APO framework automatically optimizes the prompt engineering process, eliminating the need for manual prompt tuning.

## Features

- **Automatic Prompt Optimization**: Uses APO to iteratively improve SQL generation prompts
- **Spider Dataset Integration**: Trains and evaluates on the Spider text-to-SQL benchmark
- **Configurable Optimization**: Adjustable APO parameters for different optimization strategies
- **Comprehensive Logging**: Training progress and optimization logs saved to `apo.log`

## Prerequisites

- Python 3.8+
- `uv` package manager
- OpenAI API key with access to GPT-5
- Sufficient disk space for the Spider dataset

## Setup

### 1. Install Dependencies

Install all required packages using `uv`:

```bash
uv sync
```

### 2. Download Spider Dataset

Run the setup script to download and organize the Spider dataset:

```bash
./setup_data.sh
```

After setup, your directory structure should look like:

```
data/
  dev.json
  train_spider.json
  tables.json
databases/
  concert_singer/concert_singer.sqlite
  pets_1/pets_1.sqlite
  ...
```

### 3. Configure Environment Variables

Set your OpenAI API key:

```bash
export OPENAI_API_KEY="your-api-key-here"
```

For Windows PowerShell:
```powershell
$env:OPENAI_API_KEY="your-api-key-here"
```

## Usage

### Training with APO

Start the automatic prompt optimization process:

```bash
uv run python train.py
```

The training process will:

1. Load the Spider dev.json dataset and split it into training and validation sets
2. Run APO to optimize the SQL generation prompt iteratively
3. Save optimization logs and progress to `apo.log`

### Configuration

Customize APO optimization parameters by editing `train.py`. The configuration follows the pattern from `room_selector_apo.py`:

```python
algo = APO[Dict[str, Any]](
    openai_client,
    val_batch_size=10,       # Number of examples per validation batch
    gradient_batch_size=4,   # Number of examples per gradient batch
    beam_width=2,            # Number of prompts maintained in beam
    branch_factor=2,         # Number of variants generated per prompt
    beam_rounds=2,           # Number of optimization rounds
)
```

## Project Structure

```
.
├── train.py              # Main training script with APO configuration
├── setup_data.sh         # Dataset download and setup script
├── data/                 # Spider dataset files
├── databases/            # SQLite database files for Spider
└── apo.log              # Optimization logs (generated during training)
```

## References

- [Agent-Lightning](https://microsoft.github.io/agent-lightning/stable/)
- [Spider Dataset](https://yale-lily.github.io/spider)
