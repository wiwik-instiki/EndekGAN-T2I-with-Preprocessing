# EndekGAN-T2I with Preprocessing

**Dataset and preprocessing audit for Balinese Endek motif Text-to-Image GAN research.**

This repository accompanies the original research code at [wiwik-instiki/EndekGAN-T2I](https://github.com/wiwik-instiki/EndekGAN-T2I).
It focuses on source-image auditing, full-image resizing, class-balanced photometric augmentation, and exploratory cropping comparisons. **The Text-to-Image GAN objective is unchanged.**

## Notebook

[**Open: 01_Endek_Dataset_Preprocessing_Audit.ipynb**](notebooks/01_Endek_Dataset_Preprocessing_Audit.ipynb)

The executed notebook presents:
- Four source motif classes (flora, fauna, decorative, geometric).
- Original **512 × 512**, full-image resize **256 × 256**, and **exploratory** 384 × 384 center/texture-selected cropping (then resize to 256 × 256).
- Train-only, mild adjustments to brightness, contrast, saturation and sharpness.
- Distribution charts before and after augmentation.
- Source-level train/validation/test integrity checks, Text-to-Image prompt associations and preprocessing flowchart.

**Image publication policy:** The public notebook deliberately omits two rendered galleries containing original Endek image pixels. Those cells remain executable and display all galleries when the notebook is run with an authorized local copy of the dataset. Non-photographic charts and the flowchart remain visible on GitHub. No raw or augmented image files are distributed here.

## Verified counts

| Motif class | Independent source images | Original train | Augmented train variants | Train files | Validation | Test |
|---|---:|---:|---:|---:|---:|---:|
| Flora | 147 | 118 | 118 | 236 | 14 | 15 |
| Fauna | 53 | 42 | 194 | 236 | 6 | 5 |
| Decorative | 112 | 90 | 146 | 236 | 11 | 11 |
| Geometric | 58 | 46 | 190 | 236 | 6 | 6 |
| **Total** | **370** | **296** | **648** | **944** | **37** | **37** |

Total processed files: **1,018 = 944 train + 37 validation + 37 test**. This is **not 1,018 distinct independent motifs**; augmentations inherit 296 training sources and their text prompts.

## Methodological distinction: resize vs crop

The **actual V1 training dataset** was created using full-frame resize from 512 × 512 to 256 × 256 (Lanczos) plus small photometric perturbations on training images. **Cropping was NOT included in the final V1 training set**; the notebook uses pilot coordinates to illustrate potential differences in ornament composition, not claim motif-aware segmentation. Source splitting precedes augmentation. A near-identical pair of flora images was kept within the same split, so this is a **new experimental split** despite the same 296/37/37 class totals as the prior paper.

Text-to-Image prompt/image mappings were inherited from the 370-record source CSV. Prompt text is not changed by preprocessing. These metadata descriptions have not been independently verified against every source image; assessment by Endek experts is planned at the **final stage on AI-generated motifs**.

## Run locally

1. Obtain the authorized Endek source images and derived \`ENDEK_AUGMENTED_V1\` data privately. They are **not** included in this public repository.
2. Install Python 3.10+ and the packages in [requirements-notebook.txt](requirements-notebook.txt).
3. Set \`ENDEKGAN_EXPERIMENT_ROOT\` to the directory that contains \`ENDEK_Dataset\`, \`ENDEK_AUGMENTED_V1\`, and \`ENDEK_T2I_EXPERIMENT_V2\`. The default in the notebook refers to the research PC's original D: location.
4. Open the notebook in VS Code or Jupyter and run from top to bottom.

The notebook expects the private files \`ENDEK_AUGMENTED_V1/metadata_expert_guided_T2I_augmented_V1.csv\`, \`augmentation_manifest.csv\`, \`PILOT_AUDIT_ARCHIVE_20261112.zip\` (for pilot crop coordinates), and \`FLOWCHART_PREPROCESSING_ENDEK.png\`.
For other setups, adjust paths and ensure the dataset and source metadata were prepared using the scripts.

## Repository structure

\`\`\`text
notebooks/01_Endek_Dataset_Preprocessing_Audit.ipynb
scripts/generate_endek_augmentation.py
scripts/verify_and_map_T2I_metadata.py
scripts/augmentation_quality_audit.py
scripts/prepare_experiment_v2.py
docs/FLOWCHART_PREPROCESSING_ENDEK.mmd
docs/METHODOLOGY.md
data/dataset_aggregate_counts.csv
requirements-notebook.txt
README.md
\`\`\`

## Reproducibility and limitations

- Original source images, metadata with per-image prompts, generated augmented image files, checkpoint weights, and local audit archives are **not publicly distributed**.
- The source-level split was modified to reduce potential leakage between visually similar images. The new results cannot be attributed to preprocessing alone unless compared on the **same** revised split.
- Class balancing increases training **file count**, not the number of independent real motifs.
- Neither faster GAN training, improved FID/KID, stronger prompt adherence, nor expert-approved cultural validity has yet been demonstrated by this preprocessing notebook. Quantitative GAN training is a separate experimental phase.
- Image copyrights and cultural provenance must be reviewed before releasing the actual dataset or image galleries.

The dataset documentation notebook was executed locally end-to-end: **12 code cells passed, 5 figures generated, zero execution errors**. The two Endek-photo galleries were subsequently redacted only in this public copy.
