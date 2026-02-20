# Running Phone Recognition Inference

This guide explains how to run inference with pre-trained phone recognition models using the distributed inference system.

## Quick Start

Run inference using an existing experiment config:

```bash
python src/main.py experiment=inference/transcribe_powsm data=doreco task_name=inf_doreco_powsm
```

## Components Required

### 1. Inference Config

Create a YAML config file in `configs/experiment/inference/` (e.g., `my_inference.yaml`):

```yaml
# @package _global_

defaults:
  - override /logger: csv
  - override /data: your_datamodule_name

task_name: "my_phone_recognition"
tags: ["inference"]

seed: 42
distributed_predict: True  # Required for distributed inference

inference:
  num_workers: 1  # Number of parallel workers
  passthrough_keys: ["key", "metadata_idx"]  # Dataset keys to include in output
  out_file: ${paths.output_dir}/predictions.json
  
  # Inference runner configuration
  inference_runner:
    _target_: src.model.powsm.powsm_inference.build_powsm_inference
    work_dir: ${paths.exp_dir}/powsm_cache
    hf_repo: espnet/powsm
    device: cuda  # or "auto", "cpu"
    beam_size: 5
    ctc_weight: 0.3
    # ... other inference parameters
  
  # Additional arguments passed to inference __call__ method
  inference_call_args:
    text_prev: <na>
    lang_sym: <unk>
    task_sym: <pr>
```

**Key parameters:**
- `distributed_predict: True` - Enables distributed inference mode
- `inference.inference_runner` - Configures the inference model (e.g., PoWSM)
- `inference.num_workers` - Parallel workers for processing
- `inference.out_file` - Where predictions are saved
- `inference.passthrough_keys` - Dataset fields to include in output

### 2. Lightning DataModule

Your datamodule must:
- Extend `LightningDataModule`
- Implement `predict_dataloader()` that returns a DataLoader
- Return dataset items with a `speech` key (waveform tensor)

**Example structure:**

```python
from lightning import LightningDataModule
from torch.utils.data import Dataset, DataLoader

class MyDataset(Dataset):
    def __getitem__(self, idx):
        # Load audio and return dict with required keys
        return {
            "speech": waveform,  # torch.Tensor, shape: (T,) - REQUIRED
            # Optional: additional keys that can override inference_call_args
            # (e.g., text_prev, lang_sym, task_sym for PoWSM)
            # ... other keys you want to passthrough
        }

class MyDataModule(LightningDataModule):
    def setup(self, stage=None):
        self.dataset = MyDataset(...)
    
    def predict_dataloader(self):
        return DataLoader(
            self.dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            collate_fn=self.collate_fn,
        )
```

**DataModule config** (in `configs/data/my_datamodule.yaml`):

```yaml
_target_: src.data.my_datamodule.MyDataModule
data_dir: /path/to/data
batch_size: 16
num_workers: 4
# ... other datamodule parameters
```

## How It Works

1. **Config Loading**: `src/main.py` loads your experiment config via Hydra
2. **Task Instantiation**: `Task` class (in `src/core/task.py`) instantiates the datamodule and inference runner
3. **Distributed Inference**: `run_distributed_inference()` (in `src/core/distributed_inference.py`) splits the dataset across workers
4. **Processing**: Each worker loads the model and processes its chunk of data. Dataset items are unpacked and passed to the inference object's `__call__` method
5. **Output**: Results are saved as JSON with predictions and passthrough keys

## Example: Running PoWSM Inference

See `configs/experiment/inference/transcribe_powsm.yaml` for a complete example:

```bash
# Basic usage
python src/main.py experiment=inference/transcribe_powsm data=doreco task_name=inf_doreco_powsm

# With custom parameters
python src/main.py experiment=inference/transcribe_powsm data=doreco task_name=inf_doreco_powsm \
    inference.num_workers=4 \
    inference.out_file=/path/to/output.json
```

## Output Format

The output JSON file contains:

```json
{
  "0": {
    "pred": "<prediction from model>",
    "passthrough": {
      "key": "sample_001",
      "metadata_idx": 0
    }
  },
  "1": { ... }
}
```

Where:
- Index keys correspond to dataset indices
- `pred` contains the model's prediction
- `passthrough` contains keys specified in `passthrough_keys`

## Tips

- Set `num_workers` based on available GPUs/CPUs
- Use `passthrough_keys` to preserve dataset metadata in outputs
- For GPU inference, set `device: cuda` in `inference_runner`
- Check `src/core/distributed_inference.py` for implementation details

