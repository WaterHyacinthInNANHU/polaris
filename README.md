# PolaRiS

PolaRiS is a evaluation framework for generalist policies. It provides tooling for reconstructing environments, evaluating models, and running experiments with minimal setup.

## Installation

### Clone the repository (recursively)

```bash
git clone --recursive git@github.com:arhanjain/PolaRiS.git
cd PolaRiS
```

If you cloned without `--recursive`:

```bash
git submodule update --init --recursive
```

### Setup environment with uv

```bash
uv sync
```

### HuggingFace Datasets
For using our evaluation DROID environments or simulation cotraining data, clone the datasets below.
```bash
uvx hf download owhan/PolaRiS-environments --repo-type=dataset --local-dir ./PolaRiS-environments   # Environments
uvx hf download owhan/PolaRis-datasets --repo-type=dataset --local-dir ./PolaRiS-datasets           # Cotrain Datasets
```

## Usage

Run single policy, single task evaluation with an arbitrary policy (assuming already hosted)

```bash
# uv run scripts/eval.py -h
# Example: 

uv run scripts/eval.py --environment DROID-FoodBussing --policy.name pi05 --policy.client DroidJointPos --policy.port 8010 --policy.open-loop-horizon 8
```

Running a full scale evaluation across multiple checkpoints and tasks can be easily configured with a single python file representing the entire experiment. You can optionally name your experiments via `--run-folder` flag. For example configs, see [experiments/example.py](experiments/example.py)
```bash
uv run scripts/batch_eval.py --config experiments/example.py --run-folder runs/i-love-robots
```

## Creating Custom Evaluation Environments (Time Estimate: XX)

## Adding Policies to Evaluate


## Project Structure

```text
PolaRiS/
├── scripts/
│   └── eval.py
├── PolaRiS-environments/
├── PolaRiS-datasets/
├── src/polaris/
└── README.md
```


TODO
- If nvcc, cuda toolkit isnt installed, what to do
- supports CUDA 12 only
- make sure the TORCH archirecutre list is correct (mineby default included way more than it needed)
- have correct version of gxx (my versions was too new)
- clear torch_extensions cache in between builds and env changes