# CDMPL: Chinese Cyberbullying Detection via Multi-Feature Prompt Learning

## Quick Start

### Step 1: Environment Setup

```bash
# Create and activate conda environment
conda create -n your_env python=3.9.20
conda activate your_env

# Install dependencies
pip install -r requirements.txt
```

### Step 2: Data Preparation

Ensure your data is in `datasets/TextClassification/cyberbullying/`:

- `train.csv`: Training data (columns: title, description, label, sequence)
- `test.csv`: Test data
- `class.txt`: Class labels (non-cyberbullying, cyberbullying)

### Step 3: Replace OpenPrompt Pipeline

Copy the custom pipeline\_base.py to your OpenPrompt package:

```bash
# Find your conda environment path
CONDA_ENV_PATH=$(conda info --base)/envs/your_env

# Copy the custom pipeline_base.py
cp autoformer-reproduct-pipeline_base.txt $CONDA_ENV_PATH/lib/python3.9.20/site-packages/openprompt/pipeline_base.py
```

### Step 4: Run Experiments

```bash
# Single experiment
python fewshot.py \
    --dataset cyberbullying \
    --template_id 1 \
    --verbalizer kpt \
    --shot 15 \
    --seed 143 \
    --batch_size 16 \
    --learning_rate 4e-5 \
    --max_epochs 20 \
    --result_file ./results.txt

# Batch experiments
python auto_run.py
```

<br />

