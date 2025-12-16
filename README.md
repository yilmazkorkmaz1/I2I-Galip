## I2I-Galip: Unsupervised Medical Image Translation Using Generative Adversarial CLIP

**Official PyTorch implementation of the MIDL 2025 paper  
“I2I-Galip: Unsupervised Medical Image Translation Using Generative Adversarial CLIP”**

Paper PDF(OpenReview): [link](https://openreview.net/forum?id=lAQ29DUZCa&noteId=lAQ29DUZCa)

---

### Overview

I2I-Galip is an **unpaired / unsupervised medical image-to-image translation** framework that leverages a **pre-trained multi-modal foundation model ([BiomedCLIP](https://huggingface.co/microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224) )** as a semantic guide.  

Instead of training a separate generator–discriminator pair for each source–target modality, I2I-Galip:

- **Uses a single lightweight generator (~13M parameters)** for **multi-domain** MRI/CT translation.
- Conditions the generator with **text embeddings from BiomedCLIP**, such as  
  *“this MRI is T1-weighted”* or *“this is a pelvic CT image”*.
- Is trained with a combination of:
  - **Adversarial losses** (StyleGAN-T discriminator)
  - **Cycle-consistency and identity losses**
  - **CLIP-guided alignment losses** between image and text representations.



---

### Installation

#### 1. Create and activate conda environment

```bash
conda env create -f environment.yml
conda activate i2i_galip_env
```

#### 2. Install `open_clip` in editable mode

From the repository root (this folder), run:

```bash
cd open_clip
pip install -e .
```

This installs the local `open_clip` package used by the project in **editable** mode so that imports like `import open_clip` resolve correctly.

#### 3. Download BiomedCLIP weights (automatic)

The scripts use:

- `hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224`

The weights are **automatically downloaded** via `open_clip` on first use (internet connection required).

---

### Data Preparation

I2I-Galip is designed for **unpaired** image-to-image translation between medical imaging modalities.  
The code supports:

- **MRI multi-contrast**: e.g., `T1`, `T2`, `PD`, `Flair` (IXI-style datasets)
- **CT↔MRI** pelvic translation: e.g., `ct` ↔ `mri`

Data loading is handled by `data/datasets.py` through:

- `prepare_datasets_cyclegan_single(domain_1, domain_2, dataset, batch_size, augmentations)`
- `prepare_datasets_cyclegan_multi(dataset, batch_size, augmentations)`
- `ImageDataset(split, dataset, domain_1, domain_2)` for testing.

Please see `data/datasets.py` and adjust file extensions/paths to match your local organization.

### Datasets

The following datasets are used in the paper:

- **IXI Dataset**: [https://brain-development.org/ixi-dataset/](https://brain-development.org/ixi-dataset/)  
  A collection of nearly 600 MR images from normal, healthy subjects, including T1, T2, and PD-weighted images collected at three different hospitals in London.

- **CT-MRI Pelvic Dataset**: [https://zenodo.org/records/583096](https://zenodo.org/records/583096)  
  Magnetic resonance images (MRI) (T1w and T2w) and CT images of 19 male patients over the pelvic region, suitable for CT↔MRI translation tasks.

- **Toy Dataset**: A toy dataset for IXI can be found in the [SynDiff sample data directory](https://github.com/icon-lab/SynDiff/tree/main/SynDiff_sample_data) (thanks to the SynDiff authors).

---

### Training

The main training entry point is `train.py`, which exposes all relevant hyperparameters via `argparse`.

#### Basic example: single-domain IXI T1 ↔ T2

```bash
python train.py \
  --gpu 0 \
  --exp_name ixi_T1_T2_single \
  --dataset ixi \
  --domain_1 T1 \
  --domain_2 T2 \
  --batch_size 4 \
  --epochs 100 \
```

This will:

- Use GPU `0` (`CUDA_VISIBLE_DEVICES`).
- Train a **single-domain** model mapping between T1 and T2 MRI.
- Save logs and checkpoints under `./results/ixi_T1_T2_single/`.

#### Multi-domain training (e.g., IXI with T1, T2, PD)

```bash
python train.py \
  --gpu 0 \
  --exp_name ixi_multi \
  --dataset ixi \
  --multi_domain \
  --batch_size 4 \
  --epochs 100
```

With `--multi_domain`:

- The generator and discriminator are configured to handle **3 classes** (e.g., T1/T2/PD).
- During validation, the code automatically evaluates all modality pairs, e.g.  
  `["T1","T2"]`, `["T1","PD"]`, `["T2","PD"]` for IXI.

All arguments are logged into `results/<exp_name>/log.txt` for reproducibility.

---

### Evaluation & Inference

Use `test.py` to evaluate a trained checkpoint and generate translated images.

#### Example: IXI T1 ↔ T2 (single-domain)

```bash
python test.py \
  --gpu 0 \
  --dataset ixi \
  --domain_1 T1 \
  --domain_2 T2 \
  --network_pkl ./results/ixi_T1_T2_single/epoch_95 \
  --image_size 256 \
  --multi_domain
```

Key behaviors (`test.py`):

- Loads the generator with the same architecture config as training.
- Loads BiomedCLIP:
  - `hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224`.
- Creates a test dataset via:
  - `datasets.ImageDataset("test", args.dataset, args.domain_1, args.domain_2)`.
- Constructs CLIP text prompts:
  - MRI: `"this MRI is <DOMAIN>-weighted"` (e.g., `this MRI is T1-weighted`).
  - CT↔MRI: `"this is a pelvic <modality> image"` (e.g., `this is a pelvic CT image`).
- For each test sample:
  - Generates both directions:
    - domain\_1 → domain\_2
    - domain\_2 → domain\_1
  - Computes PSNR and SSIM via `utils/psnr.py`.
  - Saves outputs as NumPy arrays in:
    - `./images/single/<dataset>/<domain_1>_to_<domain_2>/`
    - `./images/single/<dataset>/<domain_2>_to_<domain_1>/`
  - Returns averaged metrics:
    - `psnr_domain_1_ema_test`, `psnr_domain_2_ema_test`
    - `ssim_domain_1_ema_test`, `ssim_domain_2_ema_test`

---
### Citation

If you find this repository useful in your research, please consider citing our MIDL 2025 paper:

```bibtex
@inproceedings{korkmaz2025i2igalip,
  title     = {I2I-Galip: Unsupervised Medical Image Translation Using Generative Adversarial CLIP},
  author    = {Korkmaz, Yilmaz and Patel, Vishal M.},
  booktitle = {Medical Imaging with Deep Learning (MIDL)},
  year      = {2025},
  url       = {https://openreview.net/forum?id=lAQ29DUZCa&noteId=lAQ29DUZCa}
}
```
---
### Acknowledgements
- This codebase builds upon:
  - **[OpenCLIP](https://github.com/mlfoundations/open_clip)** for BiomedCLIP models and tokenizers.
  - **[Latent Diffusion](https://github.com/CompVis/latent-diffusion)** for the UNet implementation.
  - **[Stylegan-T](https://github.com/autonomousvision/stylegan-t)** for the discriminator.
  


