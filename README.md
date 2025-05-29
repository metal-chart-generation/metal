# METAL:A Multi-Agent Framework for Chart Generation with Test-Time Scaling

<p align="center">
    <a href="https://arxiv.org/abs/2502.17651" target="_blank">
        <img src="https://img.shields.io/badge/arXiv-2410.07166-B31B1B.svg?style=plastic&logo=arxiv" alt="arXiv">
    </a>
    <a href="https://metal-chart-generation.github.io" target="_blank">
        <img src="https://img.shields.io/badge/Website-EAI-purple?style=plastic&logo=Google%20chrome" alt="Website">
    </a>
    <a href="https://opensource.org/licenses/MIT">
        <img src="https://img.shields.io/badge/License-MIT-yellow.svg?style=plastic" alt="License: MIT">
    </a>
</p>

<p align="center">
    <b>ACL-2025 (Main Conference)</b>
</p>

<p align="center">
    <a href="https://bingxuanli.com/">Bingxuan Li</a>, 
    <a href="https://wangywust.github.io">Yiwei Wang</a>,
    <a href="https://gujiuxiang.com">Jiuxiang Gu</a>, 
    <a href="https://web.cs.ucla.edu/~kwchang/">Kai-Wei Chang</a>, 
    <a href="https://violetpeng.github.io/">Nanyun Peng</a>
</p>

<p align="center">
    <a href="https://metal-chart-generation.github.io/" target="_blank">
        <img src="./metal.png" alt="METAL" width="90%" height="90%" border="10" />
    </a>
</p>

---

## Table of Contents

- [Introduction](#introduction)
- [Quick Start](#quick-start)
  - [1. Environment Setup](#1-environment-setup)
  - [2. Configure Environment Variables](#2-configure-environment-variables)
  - [3. Download the Dataset](#3-download-the-dataset)
  - [4. Run METAL and Its Variations](#4-run-metal-and-its-variations)
  - [5. Run Baselines](#5-run-baselines)
- [Notes](#notes)
- [Acknowledgements](#acknowledgements)

---

## Introduction

Direct prompting of current VLMs (e.g. GPT- 4O) often fails togenerate charts that accurately replicate reference charts, resulting in errors in structure, color, and text alignment. We propose METAL to tackle this challenge with iterative refinement through generation, critique, and revision. Emperical result shows that METAL exhibits the phenomenon of test-time scaling: its performance increases monotonically as the logarithmic computational budget grows from 512 to 8192 tokens.

---

## Quick Start

### 1. Environment Setup

Create your Conda environment using the provided YAML file:

```bash
conda env create -f environment.yml
```

### 2. Configure Environment Variables

Edit the file src/.env to set up the required environment variables:

```bash
PROJECT_PATH={PATH_TO_CHARTMIMIC_DIR}
EASYOCR_MODEL_PATH={PATH_TO_EASYOCR_MODEL}
OPENAI_ORG={YOUR_OPENAI_ORG}
OPENAI_API_KEY={YOUR_OPENAI_API_KEY}
```
Replace the placeholders with your specific paths and keys.

### 3. Download the Dataset
Navigate to the dataset directory and run the download script:

```bash
cd dataset
sh get_dataset.sh
```

### 4. Run METAL and Its Variations
Before running the script, modify the shell script scripts/run_metal.sh with the appropriate paths and settings. Then, run:

```bash
sh scripts/run_metal.sh
```
Inside the script, ensure you set the following variables:

- DATASET_DIR: Path to the dataset (e.g., ../../dataset)
- WORKING_HOME_DIR: Directory where results and working files will be stored
- DATA_RANGE: Data range (e.g., 0-100)
- MODEL: Supported models (e.g., llama3_2, gpt-4o)
- SYSTEM: Supported systems (e.g., Metal, Metal-v, Metal-c, Metal-s)
- MAX_ITER: Maximum number of iterations
- N_PROCESS: Number of parallel processes (typically set to 1)
- CUDA_DEVICES: CUDA device IDs (e.g., 0,1)

### 5. Run Baselines
Similarly, modify the shell script scripts/run_baselines.sh with your specific paths and settings. Then, run:

```bash
sh scripts/run_baselines.sh
```
Set the following variables in the script:

- DATASET_DIR: e.g., ../../dataset
- WORKING_HOME_DIR: Your desired working directory
- DATA_RANGE: e.g., 0-100
- MODEL: Supported models (e.g., llama3_2, gpt-4o)
- SYSTEM: Baseline systems (e.g., HintEnhanced, Best-of-N, SelfRevision)
- MAX_ITER: Maximum iterations
- N_PROCESS: Recommended to be 8 for baselines
- CUDA_DEVICES: e.g., 0,1

## Notes

- Adjust all placeholder values (e.g., {YOUR_MODEL}, {PATH_TO_DATASET_DIR}) to suit your local setup.
- Error logs are stored at the specified LOG_FILE_PATH. Monitor these for debugging purposes.

## Acknowledgements

The dataset, direct generation code, and evaluation code are adapted from [ChartMIMIC](https://github.com/ChartMimic/ChartMimic).

## BibTex
If you find our work helpful, please kindly cite our work :)
```bash
@misc{li2025metalmultiagentframeworkchart,
          title={METAL: A Multi-Agent Framework for Chart Generation with Test-Time Scaling}, 
          author={Bingxuan Li and Yiwei Wang and Jiuxiang Gu and Kai-Wei Chang and Nanyun Peng},
          year={2025},
          eprint={2502.17651},
          archivePrefix={arXiv},
          primaryClass={cs.CV},
          url={https://arxiv.org/abs/2502.17651}, 
    }
```
