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

## Updated experimental examples (prepared 2026-11-07)

Install the revised model and data archives together. The download links below
must be updated to their new Zenodo versions when those archives are published.
Extract the model ZIP into the repository root and the data ZIP into `visualization/`.
The IL notebook examples retain `DESM_IL_inference`: raw Intensity and ASC
files are read, lifetime values are clustered, IO priors determine background
and class indexing, and the original dilation/smoothing operations run before
IL reconstruction. IO examples retain `DESM_IO_inference_with_reference` and
`DESM_IO_inference`, including their original image previews. Run the four
inference sections in `DESM_demo.ipynb` directly. No extra runner or manifest
loader is required. New outputs go to `reproduced/`; supplied results are
retained for comparison. Auxiliary scoring networks are not included.

The model/data manifests inside the resource archives list filenames, channel
order and parameters. Live IO inputs remain on their native 31/39 nm grids;
IL examples rerun preprocessing using the designated frozen IO priors.
Supplied processed TIFFs and label maps are comparison references, not inputs
required by the raw-data demo.
For the updated IO examples, `full_field=True` preserves the full field with
reflect padding, and `save_float=True` saves unclipped float32 TIFFs alongside
the original 8-bit previews. `resample_input=False` retains an already-correct
input pixel grid. The fixed `input_multiplier` reproduces the sequence scaling
before standard-deviation normalization; saved float outputs return to raw
intensity units. Network weights are fixed; BatchNorm uses the current image
statistics without updating running statistics.

The other IO datasets use the same original `DESM_IO_inference` function:
set `read_dir` to the dataset's `data/` folder, set the weight below under
`Trained_models/DESM_IO/`, and use `resample_input=False, full_field=True,
save_float=True`. These paths are relative to `visualization/Experimental data/`.

| Dataset folder | Weight filename | input_multiplier |
| --- | --- | --- |
| DESM_IO_Microtubules_Lysosomes | DESM_IO_Micro_Lyso_static.pth | 1.0 |
| DESM_IO_Mitochondria_Lysosomes | DESM_IO_Mito_Lyso.pth | 1.0 |
| DESM_IO_Microtubules_Mitochondria_Lysosomes | DESM_IO_Micro_Mito_Lyso.pth | 1.0 |
| DESM_IO_Mitochondria_Lysosomes_motion | DESM_IO_Mito_Lyso_live.pth | 2.902175014472513 |

The original training examples are retained; the newer Integrated joint-grid
training workflow is not ported in this minimal update. `net/make_model.py`
contains the SSIM gradient fix and guards zero-valued normalization maxima.

Revised inference was verified with Python 3.9.21, PyTorch 2.7.0+cu128,
NumPy 1.26.3, tifffile 2023.4.12, scikit-learn 1.6.1, scikit-image 0.24.0
and OpenCV 4.11.0. The existing repository dependencies are also required for
the original visualization/FLIM functions. The older pinned environment below is
retained for the legacy codebase and has not been revalidated for the new bundle.

## Environment

The original codebase used the following software environment:

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

1. Use the revised inference environment above for the updated examples, or `requirements.txt` for the legacy workflows.
2. Download the pretrained models and experimental data from the Zenodo links below.
3. Extract the model ZIP into the repository root and the data ZIP into `visualization/`.
4. Launch `DESM_demo.ipynb` from the repository root and run the relevant sections for DESM-IO or DESM-IL inference.

---

## Download pretrained models and experimental data

The repository can be used directly for code inspection and development. For inference, visualization, and reproduction of the experimental results, please additionally download the compressed resource packages below:

- **Trained models**: [10.5281/zenodo.20797224](https://doi.org/10.5281/zenodo.20797224)
- **Experimental data**: [10.5281/zenodo.20796449](https://doi.org/10.5281/zenodo.20796449)

Extract the model archive into the repository root and the experimental-data archive into `visualization/`, producing this structure:

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

1. Extract `Trained_models.zip` into `DESM/`; the archive includes its own `Trained_models/` directory.
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
3. Open the updated examples in the revised inference environment described above.
4. If your local folder layout differs from the default structure, update the path variables in the corresponding notebook cells before execution.

The notebook contains examples for:

- DESM-IO inference on experimental confocal data
- DESM-IL inference on experimental TCSPC-FLIM data
- dynamic imaging examples
- custom-data preparation and training
- synthetic evaluation of trained models
