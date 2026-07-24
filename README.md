# DESM (Decoupling and super-resolution microscopy)

Self-supervised single-channel multiplexed super-resolution microscopy

This repository provides the PyTorch implementation for our DESM framework, a self-supervised deep learning method for joint channel decoupling and super-resolution reconstruction from single-channel confocal / TCSPC-FLIM microscopy data.

---

## Overview

DESM is designed for multiplexed super-resolution microscopy from single-channel laser-scanning microscopy.

The framework supports two modes:

- **DESM-IO**: intensity-only mode for standard confocal data
- **DESM-IL**: intensity-lifetime mode for lifetime (TCSPC)-assisted reconstruction

The codebase includes:

- training on simulation / prepared datasets
- evaluation on synthetic and semi-synthetic data
- FLIM-assisted inference and evaluation
- visualization scripts for reconstructed results

---

## Paper

**Self-supervised single-channel multiplexed super-resolution microscopy**

Qinglin Chen<sup>1,&dagger;</sup>, Luwei Wang<sup>1,&dagger;</sup>, Min Yi<sup>1</sup>, Xiangcong Xu<sup>1</sup>, Jia Li<sup>1</sup>, Dan Shao<sup>2</sup>, Xiaoyu Weng<sup>1,&#42;</sup>, Liwei Liu<sup>1</sup>, Dayong Jin<sup>2,3,&#42;</sup>, & Junle Qu<sup>1,&#42;</sup>

<sup>1</sup> State Key Laboratory of Radio Frequency Heterogeneous Integration (Shenzhen University) & Key Laboratory of Optoelectronic Devices and Systems, College of Physics and Optoelectronic Engineering, Shenzhen University, Shenzhen 518060, China

<sup>2</sup> Institute for Biomedical Materials and Devices (IBMD), Faculty of Science, University of Technology Sydney, NSW 2007, Australia

<sup>3</sup> Zhejiang Provincial Engineering Research Center for Organelles Diagnostics and Therapy, Eastern Institute of Technology, Ningbo 315200, China

<sup>&dagger;</sup> These authors contributed equally: Qinglin Chen, Luwei Wang.  
<sup>&#42;</sup> Corresponding authors: xiaoyu@szu.edu.cn; dayong.jin@eitech.edu.cn; jlqu@szu.edu.cn

---

## Environment

The current codebase is tested with the following software environment:

- Python 3.8
- PyTorch 2.0.1
- torchvision 0.15.2
- CUDA-compatible GPU recommended

Install dependencies with:

```bash
pip install -r requirements.txt
```

---

## Quick start

For a quick reproduction of the main demo workflows:

1. Install the Python dependencies listed in `requirements.txt`.
2. Download the pretrained models and experimental data from the Zenodo links below.
3. Extract the downloaded archives into the repository root while keeping the default folder names unchanged.
4. Launch `DESM_demo.ipynb` from the repository root and run the relevant sections for DESM-IO or DESM-IL inference.

---

## Download pretrained models and experimental data

The repository can be used directly for code inspection and development. For inference, visualization, and reproduction of the experimental results, please additionally download the compressed resource packages below:

- **Trained models**: [10.5281/zenodo.20797224](https://doi.org/10.5281/zenodo.20797224)
- **Experimental data**: [10.5281/zenodo.20796449](https://doi.org/10.5281/zenodo.20796449)

After downloading, extract the archives into the project root so that the folder structure matches the paths used in the code:

```text
DESM/
|-- Trained_models/
|-- visualization/
|   |-- Experimental data/
|-- data/
|-- eval_tools/
`-- ...
```

Recommended setup:

1. Download `Trained_models.zip` from Zenodo and extract it to `DESM/Trained_models/`.
2. Download the experimental-data archive from Zenodo and extract it so that the datasets used by the visualization and evaluation scripts are placed under `DESM/visualization/Experimental data/`.
3. Keep the extracted folder names unchanged unless you also update the corresponding paths in the Python scripts.

Notes:

- The training and evaluation scripts in this repository assume local folders named `Trained_models` and `visualization/Experimental data`.
- If you place the downloaded contents elsewhere, update the relevant `weights_dir`, `prior_weights_dir`, `read_dir_intensity`, `read_dir_asc`, and `save_dir` paths in the corresponding scripts before running them.
- Some scripts under `eval_tools/` are project-specific analysis utilities and may still require manual path updates before reuse on a different machine.

---

## License

This repository is released under the MIT License. See [LICENSE](LICENSE) for details.

---

## Notebook requirements

`DESM_demo.ipynb` is intended to be run from the repository root with the downloaded pretrained models and experimental data already extracted into their default locations.

Before running the notebook:

1. Make sure `Trained_models/` is present in the project root.
2. Make sure the experimental datasets used by the demo are present under `visualization/Experimental data/`.
3. Open the notebook with the same Python environment used to install `requirements.txt`.
4. If your local folder layout differs from the default structure, update the path variables in the corresponding notebook cells before execution.

The notebook contains examples for:

- DESM-IO inference on experimental confocal data
- DESM-IL inference on experimental TCSPC-FLIM data
- dynamic imaging examples
- custom-data preparation and training
- synthetic evaluation of trained models
