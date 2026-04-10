<div align="center">

# PRiSM

[![Paper](https://img.shields.io/badge/arXiv-2601.14046-b31b1b.svg)](https://arxiv.org/abs/2601.14046)
<a href="https://huggingface.co/collections/changelinglab/prism"><img alt="Dataset" src="https://img.shields.io/badge/Dataset-PRiSM-FFD21E?logo=huggingface&logoColor=gold&logoPosition=right"></a>
<a href="https://pytorch.org/get-started/locally/"><img alt="PyTorch" src="https://img.shields.io/badge/PyTorch-ee4c2c?logo=pytorch&logoColor=white"></a>
<br>
<!-- [![Conference](http://img.shields.io/badge/AnyConference-year-4b44ce.svg)](https://papers.nips.cc/paper/2020) -->

</div>

## Announcements
**April 7, 2026: 🎉 Paper accepted at ACL Main 2026!**

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

## Description

A benchmark for evaluating phonetic capabilities of speech models.

## 📚 Datasets

PRiSM datasets are organized in the Hugging Face collection: [changelinglab/prism](https://huggingface.co/collections/changelinglab/prism).

The benchmark currently uses the following dataset sources for extrinsic tasks:

| Task | HF-Repo | Task |
| --- | --- | --- |
| `DYS-ez` | `changelinglab/easycall-dysarthria` | Dysarthria Intelligibility Prediction |
| `CSD-us` | `changelinglab/ultrasuite-benchmark` | Atypical child speech classification |
| `L1-eda` | `changelinglab/edacc-l1cls` | L1 classification |
| `L1-arc` | `changelinglab/cmul2arctic-l1cls` | L1 Classification |
| `L2-so` | `changelinglab/speechocean-l2eval` | L2 assessment |
| `LID-fl` | `changelinglab/fleurs24-lid` | Language Identification |
| `GEO-v` | `shikhar7ssu/vaani-hi-geo` | Speech geolocation (To be released!) |

For intrinsic task (Phone Recognition), PRiSM evaluation configs include Kaldi-style test sets (`doreco`, `gmuaccent`, `l2arctic_perceived`, `timit`, `tusom2021`, `voxangeles`) via `configs/data/powsm_evalset_index.yaml`.

To set up evalset-index data, download the corresponding `*-pr` dataset repos from the collection into one root directory, then point `data.data_dir` to that root when running inference/evaluation. The expected layout under that root must match `configs/data/powsm_evalset_index.yaml` (each dataset directory contains `wav.scp`, `text.good`, and language files).

After data download, PR inference can be run with:
```bash
# run with evalset index (override data_dir)
python src/main.py experiment=inference/transcribe_powsm data=powsmeval data.data_dir=/path/to/prism-evalsets
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
python src/main.py experiment=probing/lid_fleurs_powsm

# For inference experiments
python src/main.py experiment=inference/transcribe_powsm data=doreco data.dataset_name=voxangeles task_name=inf_voxangeles_powsm
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
