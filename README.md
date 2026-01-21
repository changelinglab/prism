______________________________________________________________________

<div align="center">

# PRiSM

<a href="https://pytorch.org/get-started/locally/"><img alt="PyTorch" src="https://img.shields.io/badge/PyTorch-ee4c2c?logo=pytorch&logoColor=white"></a>
<a href="https://pytorchlightning.ai/"><img alt="Lightning" src="https://img.shields.io/badge/-Lightning-792ee5?logo=pytorchlightning&logoColor=white"></a>
<a href="https://hydra.cc/"><img alt="Config: Hydra" src="https://img.shields.io/badge/Config-Hydra-89b8cd"></a>
<a href="https://github.com/ashleve/lightning-hydra-template"><img alt="Template" src="https://img.shields.io/badge/-Lightning--Hydra--Template-017F2F?style=flat&logo=github&labelColor=gray"></a><br>
<!-- [![Paper](http://img.shields.io/badge/paper-arxiv.1001.2234-B31B1B.svg)](https://www.nature.com/articles/nature14539)
[![Conference](http://img.shields.io/badge/AnyConference-year-4b44ce.svg)](https://papers.nips.cc/paper/2020) -->

</div>

## Description

A benchmark for evaluating phonetic models.

## 🚀 Quickstart

```bash
# clone project
git clone git@github.com:changelinglab/prism.git
cd prism

# create environment with your favourite package manager 
# and install dependencies from requirements.txt
# We provide "setup_uv.sh" for doing these and activating environment
. ./setup_uv.sh

```

## How to run

Train model with default configuration

```bash
# train on CPU
python src/main.py trainer=cpu

# train on GPU
python src/main.py trainer=gpu
```

Train model with chosen experiment configuration from [configs/experiment/](configs/experiment/)

```bash
# For probing experiments using hidden representations
python src/main.py experiment=probing/geolocation_vaani_powsm

# For inference experiments
python src/main.py experiment=inference/vaani_powsmpr
```

You can override any parameter from command line like this

```bash
python src/main.py trainer.max_epochs=20 data.batch_size=64
```

## More Documentation

- **[Features & Capabilities](docs/features.md)** - Look at this to train on multi-gpu, run hyper-param searches etc.
- **[Running Inference](docs/running_inference.md)** - Guide for running phone recognition inference with pre-trained models
- **[Tokenization Workflow](docs/tokenization.md)** - How to build vocabularies and use tokenizers for IPA transcripts
- **[Contributing Guide](CONTRIBUTING.md)** - Project structure, workflow, and best practices for contributors

## Citation
If you use this code in your research, please cite our paper:

```bibtex
@misc{prism2026,
      title={PRiSM: Benchmarking Phone Realization in Speech Models}, 
      author={Shikhar Bharadwaj and Chin-Jou Li and Yoonjae Kim and Kwanghee Choi and Eunjung Yeo and Ryan Soh-Eun Shim and Hanyu Zhou and Brendon Boldt and Karen Rosero Jacome and Kalvin Chang and Darsh Agrawal and Keer Xu and Chao-Han Huck Yang and Jian Zhu and Shinji Watanabe and David R. Mortensen},
      year={2026},
      eprint={2601.14046},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/2601.14046}, 
}
```

## ❤️ Acknowledgement

This repository structure is based on the [Lightning-Hydra-Template](https://github.com/ashleve/lightning-hydra-template).
