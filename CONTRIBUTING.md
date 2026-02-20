# Contributing to PRiSM

This document outlines the project structure and development workflow for PRiSM.

## Project Structure

The directory structure of this project looks like this:

```
├── .github                   <- Github Actions workflows
│
├── configs                   <- Hydra configs
│   ├── callbacks                <- Callbacks configs
│   ├── data                     <- Data configs
│   ├── debug                    <- Debugging configs
│   ├── experiment               <- Experiment configs
│   ├── extras                   <- Extra utilities configs
│   ├── hparams_search           <- Hyperparameter search configs
│   ├── hydra                    <- Hydra configs
│   ├── local                    <- Local configs
│   ├── logger                   <- Logger configs
│   ├── model                    <- Model configs
│   ├── paths                    <- Project paths configs
│   ├── trainer                  <- Trainer configs
│   │
│   └── main.yaml             <- Main config for training/testing/inference
│
├── scripts                <- Shell scripts
│
├── src                    <- Source code
│   ├── core                     <- Core utilities (task management)
│   ├── data                     <- Data modules and dataloaders
│   ├── model                    <- Model architectures
│   ├── recipe                   <- Recipe-specific modules (models and local scripts)
│   │   └── <identifier>/local       <- Use this for ad hoc local data-prep scripts
│   ├── utils                    <- Utility scripts
│   │
│   └── main.py                  <- Main entry point for training
│
├── tests                  <- Tests of any kind
│
├── .env.example              <- Example of file for storing private environment variables
├── .gitignore                <- List of files ignored by git
├── .pre-commit-config.yaml   <- Configuration of pre-commit hooks for code formatting
├── .project-root             <- File for inferring the position of project root directory
├── pyproject.toml            <- Configuration options for testing and linting
├── requirements.txt          <- File for installing python dependencies
├── setup.py                  <- File for installing project as a package
└── README.md
```

## Workflow

### Basic workflow

1. Write your PyTorch Lightning model module (see [recipe/geolocation/model_module.py](src/recipe/geolocation/model_module.py) for example)
1. Write your PyTorch Lightning datamodule (see [data/fleurs/common_datamodule.py](src/data/fleurs/common_datamodule.py) for example)
1. Write your experiment config, containing paths to model and datamodule (see [configs/experiment/probing/lid_fleurs_powsm.yaml](configs/experiment/probing/lid_fleurs_powsm.yaml) for example)
1. Run training with chosen experiment config:
   ```bash
   # For probing experiments (use task_dataset_model naming)
   python src/main.py experiment=probing/lid_fleurs_powsm
   ```

### Experiment design

_Say you want to execute many runs to plot how accuracy changes in respect to batch size._

1. Execute the runs with some config parameter that allows you to identify them easily, like tags:

   ```bash
   python src/main.py -m logger=csv data.batch_size=16,32,64,128 tags=["batch_size_exp"]
   ```

2. Retrieve results from your chosen logger/artifact store (e.g., W&B, CSV logger) using the tags you set, and plot the results.

## How It Works

All PyTorch Lightning modules are dynamically instantiated from module paths specified in config. Example model config:

```yaml
_target_: src.recipe.geolocation.model_module.PowsmGeolocationModule

optimizer:
  _target_: torch.optim.Adam
  _partial_: true
  lr: 0.001
  weight_decay: 0.0

scheduler:
  _target_: torch.optim.lr_scheduler.ReduceLROnPlateau
  _partial_: true
  mode: min
  factor: 0.1
  patience: 10

# Provide your local model/config paths via configs or env vars
s2t_train_config: /path/to/espnet/config.yaml
s2t_model_file: /path/to/espnet/model.pth
bpemodel: /path/to/token_list/bpe.model
```

Using this config we can instantiate the object with the following line:

```python
model = hydra.utils.instantiate(config.model)
```

This allows you to easily iterate over new models! Every time you create a new one, just specify its module path and parameters in appropriate config file.

Switch between models and datamodules with command line arguments:

```bash
python src/main.py model=classification data=fleurs
```

Example pipeline managing the instantiation logic: [src/main.py](src/main.py).

## Main Config

Location: [configs/main.yaml](configs/main.yaml)  
Main project config contains default training configuration.  
It determines how config is composed when simply executing command `python src/main.py`.

<details>
<summary><b>Show main project config</b></summary>

```yaml
# @package _global_

# order of defaults determines the order in which configs override each other
# all configs can be overridden from command line, e.g. `python src/main.py debug=default)
defaults:
  - _self_
  - model: ??? # must be defined in experiment config
  - model/net: ???
  - data: ???
  - optional data/tokenizer: null # some exp do not need a tokenizer
  - trainer: default
  - paths: default
  - logger: csv
  - callbacks: default
  - extras: default
  - hydra: default
  # experiment configs allow for reproducibility with specific hyperparameters
  - experiment: null
  - hparams_search: null # config for hyperparameter optimization
  # optional local config for machine/user specific settings
  # it's optional since it doesn't need to exist and is excluded from version control
  - optional local: default
  - debug: null # debugging config

# task name, determines output directory path
task_name: "train"

# tags to help you identify your experiments
# you can overwrite this in experiment configs
# overwrite from command line with `python src/main.py tags="[first_tag, second_tag]"`
tags: ["dev"]

train: True # set False to skip model training
# evaluate on test set, using best model weights achieved during training
# lightning chooses best weights based on the metric specified in checkpoint callback
test: True
# prediction stage runs if 'predict' is True or 'pred_file' is not None
predict: False # run prediction on new/unseen data
pred_file: null # path to write predictions to

run_folder: ${now:%Y%m%d}_${now:%H%M%S}
# useful to resume training from a checkpoint path
ckpt_path: null
seed: 42 # pytorch, numpy and python.random
```

</details>

## Experiment Config

Location: [configs/experiment](configs/experiment)  
Experiment configs allow you to overwrite parameters from main config.  
For example, you can use them to version control best hyperparameters for each combination of model and dataset.

Experiment configs are organized in subdirectories:
- `probing/` - Probing experiments (naming: `task_dataset_model.yaml`)
- `inference/` - Inference experiments
- `cascade/` - Cascade experiments

<details>
<summary><b>Show example probing experiment config</b></summary>

```yaml
# @package _global_

# to execute this experiment run:
# python src/main.py experiment=probing/lid_fleurs_powsm

defaults:
  - override /data: fleurs
  - override /model: classification
  - override /model/head: attnmlp
  - override /model/net: powsm
  - override /trainer: gpu
  - override /callbacks: default
  - override /logger: wandb

# all parameters below will be merged with parameters from default configurations set above
# this allows you to overwrite only specified parameters

task_name: "lid_fleurs_powsm"
tags: ["fleurs", "powsm", "lid", "classification"]

seed: 42

trainer:
  min_epochs: 5
  max_epochs: 12
  gradient_clip_val: 1.0
  val_check_interval: 0.25  # validate 4 times per epoch

model:
  head:
    task_type: "classification"
  freeze_encoder: true
  optimizer:
    lr: 0.0002
  scheduler:
    total_iters: 200

callbacks:
  model_checkpoint:
    monitor: "val/f1"
    mode: "max"
    save_top_k: 1
  early_stopping:
    monitor: "val/f1"
    patience: 5
    mode: "max"

data:
  batch_size: 48

logger:
  wandb:
    tags: ${tags}
    group: ${task_name}
    name: ${task_name}
```

> **Note**: Probing experiment configs follow the naming convention `task_dataset_model.yaml` (e.g., `lid_fleurs_powsm.yaml`) and tags are structured as `[dataset, model, task]`.

> **Note**: Callbacks such as `model_checkpoint` and `early_stopping` monitor training/validation metrics and are only active during training mode (`train: True`). They are not used during inference.

</details>

## Accessing Datamodule Attributes In Model

The simplest way is to pass datamodule attribute directly to model on initialization:

```python
# ./src/core/task.py
datamodule = hydra.utils.instantiate(config.data)
model = hydra.utils.instantiate(config.model, some_param=datamodule.some_param)
```

> **Note**: Not a very robust solution, since it assumes all your datamodules have `some_param` attribute available.

Similarly, you can pass a whole datamodule config as an init parameter:

```python
# ./src/core/task.py
model = hydra.utils.instantiate(config.model, dm_conf=config.data, _recursive_=False)
```

You can also pass a datamodule config parameter to your model through variable interpolation:

```yaml
# ./configs/model/my_model.yaml
_target_: src.models.my_module.MyLitModule
lr: 0.01
some_param: ${data.some_param}
```

Another approach is to access datamodule in LightningModule directly through Trainer:

```python
# ./src/recipe/geolocation/model_module.py
def on_train_start(self):
  self.some_param = self.trainer.datamodule.some_param
```

> **Note**: This only works after the training starts since otherwise trainer won't be yet available in LightningModule.

## Best Practices


<details>
<summary><b>Use automatic code formatting</b></summary>

Use pre-commit hooks to standardize code formatting of your project and save mental energy.  
Simply install pre-commit package with:

```bash
pip install pre-commit
```

Next, install hooks from [.pre-commit-config.yaml](.pre-commit-config.yaml):

```bash
pre-commit install
```

After that your code will be automatically reformatted on every new commit.

To reformat all files in the project use command:

```bash
pre-commit run -a
```

To update hook versions in [.pre-commit-config.yaml](.pre-commit-config.yaml) use:

```bash
pre-commit autoupdate
```

</details>

<details>
<summary><b>Set private environment variables in .env file</b></summary>

System specific variables (e.g. absolute paths to datasets) should not be under version control or it will result in conflict between different users. Your private keys also shouldn't be versioned since you don't want them to be leaked.

Template contains `.env.example` file, which serves as an example. Create a new file called `.env` (this name is excluded from version control in .gitignore).  
You should use it for storing environment variables like this:

```
MY_VAR=/home/user/my_system_path
```

All variables from `.env` are loaded in `src/main.py` automatically.

Hydra allows you to reference any env variable in `.yaml` configs like this:

```yaml
path_to_data: ${oc.env:MY_VAR}
```

</details>

<details>
<summary><b>Name metrics using '/' character</b></summary>

Depending on which logger you're using, it's often useful to define metric name with `/` character:

```python
self.log("train/loss", loss)
```

This way loggers will treat your metrics as belonging to different sections, which helps to get them organised in UI.

</details>

<details>
<summary><b>Use torchmetrics</b></summary>

Use official [torchmetrics](https://github.com/PytorchLightning/metrics) library to ensure proper calculation of metrics. This is especially important for multi-GPU training!

For example, instead of calculating accuracy by yourself, you should use the provided `Accuracy` class like this:

```python
from torchmetrics.classification.accuracy import Accuracy


class LitModel(LightningModule):
    def __init__(self)
        self.train_acc = Accuracy()
        self.val_acc = Accuracy()

    def training_step(self, batch, batch_idx):
        ...
        acc = self.train_acc(predictions, targets)
        self.log("train/acc", acc)
        ...

    def validation_step(self, batch, batch_idx):
        ...
        acc = self.val_acc(predictions, targets)
        self.log("val/acc", acc)
        ...
```

Make sure to use different metric instance for each step to ensure proper value reduction over all GPU processes.

Torchmetrics provides metrics for most use cases, like F1 score or confusion matrix. Read [documentation](https://torchmetrics.readthedocs.io/en/latest/#more-reading) for more.

</details>

<details>
<summary><b>Follow PyTorch Lightning style guide</b></summary>

The style guide is available [here](https://pytorch-lightning.readthedocs.io/en/latest/starter/style_guide.html).

1. Be explicit in your init. Try to define all the relevant defaults so that the user doesn't have to guess. Provide type hints. This way your module is reusable across projects!

   ```python
   class LitModel(LightningModule):
       def __init__(self, layer_size: int = 256, lr: float = 0.001):
   ```

2. Preserve the recommended method order.

   ```python
   class LitModel(LightningModule):

       def __init__():
           ...

       def forward():
           ...

       def training_step():
           ...

       def training_step_end():
           ...

       def on_train_epoch_end():
           ...

       def validation_step():
           ...

       def validation_step_end():
           ...

       def on_validation_epoch_end():
           ...

       def test_step():
           ...

       def test_step_end():
           ...

       def on_test_epoch_end():
           ...

       def configure_optimizers():
           ...

       def any_extra_hook():
           ...
   ```

</details>

<details>
<summary><b>Keep local configs out of code versioning</b></summary>

Some configurations are user/machine/installation specific (e.g. configuration of local cluster, or harddrive paths on a specific machine). For such scenarios, a file [configs/local/default.yaml](configs/local/) can be created which is automatically loaded but not tracked by Git.

For example, you can use it for a SLURM cluster config:

```yaml
# @package _global_

defaults:
  - override /hydra/launcher@_here_: submitit_slurm

data_dir: /mnt/scratch/data/

hydra:
  launcher:
    timeout_min: 1440
    gpus_per_task: 1
    gres: gpu:1
  job:
    env_set:
      MY_VAR: /home/user/my/system/path
      MY_KEY: asdgjhawi8y23ihsghsueity23ihwd
```

</details>

## Tests [WIP]

Generic tests are implemented with `pytest`.

```bash
# run all tests
pytest

# run tests from specific file
pytest tests/test_train.py

# run all tests except the ones marked as slow
pytest -k "not slow"
```

Most of the implemented tests don't check for any specific output - they exist to simply verify that executing some commands doesn't end up in throwing exceptions. You can execute them once in a while to speed up the development.

Currently, the tests cover cases like:

- running 1 train, val and test step
- running 1 epoch on 1% of data, saving ckpt and resuming for the second epoch
- running 2 epochs on 1% of data, with DDP simulated on CPU

And many others. You should be able to modify them easily for your use case.

There is also `@RunIf` decorator implemented, that allows you to run tests only if certain conditions are met, e.g. GPU is available or system is not windows. See the [examples](tests/test_train.py).