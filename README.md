______________________________________________________________________

<div align="center">

# PRiSM

[![Paper](https://img.shields.io/badge/arXiv-2601.14046-b31b1b.svg)](https://arxiv.org/abs/2601.14046)
<a href="https://huggingface.co/collections/changelinglab/prism"><img alt="Dataset" src="https://img.shields.io/badge/Dataset-PRiSM-FFD21E?logo=huggingface&logoColor=gold&logoPosition=right"></a>
<a href="https://pytorch.org/get-started/locally/"><img alt="PyTorch" src="https://img.shields.io/badge/PyTorch-ee4c2c?logo=pytorch&logoColor=white"></a>
<br>
<!-- [![Conference](http://img.shields.io/badge/AnyConference-year-4b44ce.svg)](https://papers.nips.cc/paper/2020) -->

</div>

## Description

A benchmark for evaluating phonetic capabilities of speech models.

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

This codebase pulls datasets from the huggingface collection [https://huggingface.co/collections/changelinglab/prism](https://huggingface.co/collections/changelinglab/prism).

> **Note**: Intrinsic evaluation datasets (e.g., DoReCo, TIMIT) are not yet uploaded to HuggingFace. These will be made available in a future release.


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
python src/main.py experiment=probing/lid_fleurs_powsm

# For inference experiments
python src/main.py experiment=inference/transcribe_powsm data=doreco task_name=inf_doreco_powsm
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
