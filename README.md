<div align="center">

<h1 style="text-align: center;">Sample-efficient Integration of New Modalities into Large Language Models</h1>

<div>
  <sup>1</sup><a href='https://ospanbatyr.github.io/' target='_blank'><b>Osman Batur İnce</b></a>&emsp;
  <sup>2,3,4</sup><a href='https://andre-martins.github.io/' target='_blank'>André F. T. Martins</b></a>&emsp;
  <sup>1</sup><a href='https://homepages.inf.ed.ac.uk/omacaod/' target='_blank'>Oisin Mac Aodha</b></a>&emsp;
  <sup>1</sup><a href='https://ducdauge.github.io/' target='_blank'>Edoardo M. Ponti</b></a>&emsp;

</div>
<div><sup>1</sup>University of Edinburgh</div>
<div><sup>2</sup>Instituto de Telecomunicações</div>
<div><sup>3</sup>Instituto Superior Técnico, Universidade de Lisboa</div>
<div><sup>4</sup>Unbabel</div>

<div>
<h4>

</h4>
</div>

[![Dataset: CAPDELS](https://img.shields.io/badge/Dataset-CAPDELS-FFD21E.svg)](https://huggingface.co/datasets/ospanbatyr/capdels)

![Demo GIF](./figures/project1.gif)

<div align="left">

## TL;DR

We effectively integrate unseen low-resource modalities to large language models with as few as 32 samples by leveraging high-resource modalities.

## Abstract

Multimodal foundation models can process several modalities. However, since the space of possible modalities is large and evolving over time, training a model from scratch to encompass all modalities is unfeasible. Moreover, integrating a modality into a pre-existing foundation model currently requires a significant amount of paired data (e.g., images with text), which is often not available for low-resource modalities. In this paper, we study how to integrate unseen modalities into Large Language Models (LLMs) in a sample-efficient way. To this end, we train a hypernetwork to generate parameter-efficient adapters, which are modality-specific, on top of a shared projector placed between each modality-specific encoder and the LLM. The hypernetwork, trained on high-resource modalities (i.e., text, speech, audio, video), can be conditioned on a subset of samples from any arbitrary modality at inference time to generate an appropriate adapter. To increase the diversity of seen modalities, we artificially multiply the number of training encoders through isometric transformations. We demonstrate that our method achieves a significant increase in sample efficiency during few-shot integration of new modalities (i.e., satellite images, astronomical images, and molecules) with encoders of arbitrary embedding dimensionality. Specifically, our method's 32-shot performance requires up to 64$\times$ less data than learning a projector from scratch and 77$\times$ less data than fine-tuning a projector pre-trained on seen modalities to achieve comparable results, substantially extending the modality coverage of foundation models.

## Installation

#### Coding Environment
```bash
conda env create -f environment.yml
conda activate dynamic_mm
```

Then, either install `requirements-cuda.txt` or `requirements.txt` based on your environment. In our environment, we used `requirements-cuda.txt`.

```bash
pip install -r requirements-cuda.txt
pip install -e .
```

#### Used Frameworks

You will need to login to Huggingface and Weights & Biases through CLI. Moreover, you will need access to Llama 3.2 series LLMs through Huggingface as well. For further info, please refer to following links:

- [Huggingface CLI Login](https://huggingface.co/docs/huggingface_hub/main/en/guides/cli)
- [Weights & Biases Login](https://docs.wandb.ai/ref/cli/wandb-login/)
- [Llama 3.2 1B Instruct Access Page](https://huggingface.co/meta-llama/Llama-3.2-1B-Instruct)

#### Materials

##### Dataset

1. Execute the `python pkl.py` command under the `dmi/data` directory to decompress the extracted embeddings and other dataset files.

##### Evaluation

1. Clone the `ospanbatyr/cococap` repository under the `dmi` folder.
2. Change directory to `dmi/cococap`.
2. Execute the `get_stanford_models.sh` script.
3. Download the JDK 8u202 specific to your distribution from [Java SE 8 Archive Downloads](https://www.oracle.com/java/technologies/javase/javase8-archive-downloads.html) or [Java 8 Downloads](https://www.oracle.com/java/technologies/downloads/#java8) and install it under the `dmi/cococap` folder. The folder should be named `jdk1.8.0_202`. Other Java 8 JDK versions might work as well, but we tested on Linux x64 architecture and 8u202 release. One can also change the values of `JAVAPATH` variables within the Python files under the `dmi/cococap` folder.

##### Checkpoints

1. To skip the projector pre-training and hypernetwork training stages, download the pre-trained projector and hypernetwork weights from [Huggingface](https://huggingface.co/ospanbatyr/sample-efficient-multimodality-ckpts/tree/main) and place it under the `dmi/checkpoints` directory.


#### Experiments

##### Projector Pre-training

```bash
python -u train_projector.py configs/projector/v1:llama1b_inst_all_extracted.json
```

##### Hypernetwork Training

```bash
python -u train_hypernet.py configs/projector/v4:llama1b_inst_all.json
```

##### Few-shot Runs

```bash
# Example run, please see dmi/configs/hypernet only_fewshot configs for remaining experiments
python -u train_hypernet.py configs/hypernet/v6:llama1b_inst_all_only_fewshot_candels_base.json
```

##### Baselines

```bash
# Example runs, please see dmi/configs/lora and dmi/configs/projector folders for remaining experiments
python -u train_projector.py configs/projector/v2:llama1b_sydney_rn50_mlp2.json        # Projector baseline
python -u train_projector.py configs/projector/v3:llama1b_sydney_rn50_mlp2_ft.json     # FT Projector baseline
python -u train_lora.py configs/lora/v3:llama1b_inst_mlp2_sydney_rn50.json             # LoRA baseline
```

##### Evaluation

The evaluation of our experiments are automatic. Please refer to `{dataset-name}-results.json` under the `outputs/` folder.

## Acknowledgment

We thank Benjamin Minixhofer, Csordás Róbert, and Giorgio Roffo for their valuable codebases. If you liked our work, please pay their wonderful codebases a visit:

- [Benjamin Minixhofer - Zero-shot Tokenizer Transfer](https://github.com/bminixhofer/zett)
- [Csordás Róbert - Transformer Generalization](https://github.com/RobertCsordas/transformer_generalization)
- [Giorgio Roffo - Infinite Feature Selection](https://github.com/giorgioroffo/Infinite-Feature-Selection)
