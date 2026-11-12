# EndekGAN-T2I with Preprocessing

**New preprocessing-to-training research workflow for generating Balinese Endek motifs from text prompts.**

This repository documents the new experimental dataset treatments, preprocessing audit, training code, and reproducible evaluation. It is separate from the earlier [EndekGAN-T2I research repository](https://github.com/wiwik-instiki/EndekGAN-T2I). **No previously trained GAN checkpoint is reused for the new experiment.**

## Active Jupyter notebooks — only two

1. [**01 — Preprocessing BEFORE Training V2**](notebooks/01_EndekGAN_T2I_Preprocessing_BEFORE_Training_V2.ipynb): source inventory, external-dataset admission audit, duplicate screening, four-class mapping, resize, crop exploration, photometric augmentation, source-aware splits, image–prompt checks, distribution tables and preprocessing flowchart.
2. [**05 — Train Preprocessed Crop V2**](notebooks/05_EndekGAN_T2I_Train_Preprocessed_Crop_V2.ipynb): data validation, genuine new GAN training, controls, GPU checks, checkpoint management, model evaluation, and text-prompt image generation.

The earlier preliminary dataset-audit notebook has been superseded and removed from the active repository. The prior version remains accessible in Git history.

## Experimental dataset arms

| Arm | Training image files | Validation | Testing | Purpose |
|---|---:|---:|---:|---|
| **B — full image + photometric augmentation + crop** | **1,280** | **37** | **37** | Main experimental preprocessing treatment; 320 training files per motif class |
| **A — full image + photometric augmentation** | **944** | **37** | **37** | Matched no-crop control |

The four motif classes are flora, fauna, dekoratif and geometris. Validation and testing consist of the same held-out source records for A and B, with no cropping or training augmentations.

**Interpretation:** These numbers count processed files, not independently acquired motif photographs. The 336 extra images in B are training-only heuristic crop candidates derived from existing sources. Their inherited whole-image prompts have not been independently verified at crop level. The experiment therefore tests cropping as an *ablation factor*; it does not presume improved semantic accuracy.

External sources under evaluation include [Mendeley Multimodal Endek](https://data.mendeley.com/datasets/52ffyj9x3c/1), [Mendeley Endek GAN](https://data.mendeley.com/datasets/9jwns77cg5/1), and the [Roboflow annotated Endek project](https://universe.roboflow.com/ayu-menuh-rhmpz/endek-dataset-anoted-multi-iobno). Their image archives were **not** successfully incorporated when these notebooks were prepared. Expanding the dataset requires license/provenance checking, deduplication, prompt mapping, and a new group-aware source split.

## Run locally on the research Windows PC

- **Research project:** `D:\DISERTASI_S3\WIWIK\EXPERIMEN_2\ENDEK_T2I_EXPERIMENT_V2`
- **Dataset manifests:** `D:\DISERTASI_S3\WIWIK\EXPERIMEN_2\ENDEK_PREPROCESSING_V2\arms`
- **Engine-compatible splits:** `TRAINING_SPLITS_PREPROCESSED_V2` (inside the experiment folder)
- **Python interpreter and all research packages:** `D:\VSCode-MultiAgent-AI\envs\endekgan-t2i-v2-py311\Scripts\python.exe`
- **VS Code kernel:** `Python 3.11 (EndekGAN-T2I V2 | D:)`

Run the preprocessing notebook first to audit the data, then open the training notebook. Use the [training guide](docs/TRAINING_V2_README.md) and isolated [CUDA installation script](setup/INSTALL_CUDA_PYTORCH_ON_D_FOR_V2.cmd) if compatible GPU packages are needed. Full GPU training is guarded by an explicit `TRAIN_NOW=True` switch and free-VRAM checks; `Run All` performs a safe preflight by default.

## Experimental plan and current status

- Six original internal GAN configurations across seeds 42, 123, and 2025 on B, plus no-crop controls on A, with equal sampled items and optimizer updates per epoch.
- Text-to-Image conditioning uses the prompt CSV; the proposed method retains Transformer conditioning and seam-aware regularization from the prior architecture.
- New training begins from random weights. Validation KID is for checkpoint selection; small held-out test-set FID/KID results must be interpreted cautiously.
- Experts will evaluate **AI-generated motif designs** at the final stage.
- **Preprocessing and notebook preflight have passed. Full V2 training and model evaluation have not started** because CUDA PyTorch is absent from the selected D environment and available GPU VRAM was below the safety threshold at the last inspection.

## Repository data and code

The repository includes `notebooks/`, `scripts/`, `setup/`, `docs/`, and aggregate information in `data/`. Actual source photographs, derived image files, private image–prompt CSV manifests, and trained `.pth` weights are **not** publicly distributed. Authorized local files are required to reproduce the study.

For method and source-audit details, see [preprocessing documentation](docs/PRE_EXPERIMENT_PREPROCESSING_V2.md), [training instructions](docs/TRAINING_V2_README.md), and [dataset methodology notes](docs/METHODOLOGY.md).
