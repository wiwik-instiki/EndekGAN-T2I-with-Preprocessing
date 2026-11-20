# Code and Data Availability — IEEE Access Endek T2I

**Repository:** https://github.com/wiwik-instiki/EndekGAN-T2I-with-Preprocessing

**Research method:** Endek T2I. Legacy experiment files named V2 denote processing versions, not a newly named scientific method. This public code-and-results release does not include the full Endek training image collection.

## Exact public evidence inventory

| Manuscript element | Public release | Availability boundary |
|---|---|---|
| Source-aware preprocessing / Arm A and B | [Preprocessing engine](../scripts/endek_v2_preprocessing_training_engine.py), [experiment preparation](../scripts/prepare_experiment_v2.py), [dataset methods](DATASET_PREPROCESSED_IEEE_20261117.md) | Implementation and aggregate provenance workflow, not original image inputs |
| Dataset counts, 370 independently sourced photographs | [Aggregate source counts](../data/dataset_aggregate_counts.csv), [preprocessing counts](../data/preprocessing_v2_aggregate_counts.csv) | Aggregate statistics only; the complete collection is private |
| RQ1: Arm A versus Arm B | [Checkpoint validation KID](../results/three_seed_checkpoint_validation_summary.csv), [numeric Figure](../figures/fig21_preprocessing_A_B_validation_kid.png) | Validation-set checkpoint selection, not independent held-out evaluation |
| RQ2: proposed model versus internal no-seam and adapted DF-GAN–MiniLM | [Nine matched model-by-seed tests](../results/DFGAN_EndekGAN_real_matched_three_seeds.csv), [internal held-out aggregates](../results/heldout_B_3seeds_mean_sd.csv), [adapted DF-GAN held-out aggregates](../results/DFGAN_V2_real_heldout_mean_sd.csv) | Seeds 42, 123, and 2025; 37 independent real test sources and 256 generated samples per model–seed; exploratory scores |
| Results figures | [Figures](../figures/), [figure provenance](IEEE_RESULTS_README_PROVENANCE_20261120.md) | Selected plots and illustrative panels, not a full private generated-image archive |
| Image-conditioned TiPGAN reference | [Four-category training status](../results/TIPGAN_FOUR_CLASS_TRAINING_STATUS_20261120.csv), [interpretation](../benchmarks/TIPGAN_FOUR_CLASS_TRAINING_20261120.md) | Single-reference training per category; no independent comparable held-out T2I evaluation |
| Implementation/reproducibility | [Scripts](../scripts/), [sanitized notebooks](../notebooks/), [notebook requirements](../requirements-notebook.txt) | Some runtime dependencies and source-image/prompt inputs are privately held |

## Restricted materials

- Complete 370-original-photograph source corpus, 944 Arm A and 1,280 Arm B derived train views, validation/test photographs, and raw dataset source lineage.
- Source-linked private prompts, metadata, private manifests, crop-parent descriptions and complete raw generated-image collections.
- Model checkpoints, full training logs and private experiment/Kaggle archives.
- Any missing licenses or permissions for reuse of underlying Endek photographs.

These are retained only in the authorized research environment. Their access and redistribution are not granted by visibility of this repository. The public README already displays selected **source-derived training examples** in manuscript Figure 1; the corresponding rights to further redistribute these examples require independent verification. Access requests require separate authorization from data rights holders and applicable data-use rules; publication here does not promise release.

## Reproducibility and scientific interpretation

The public artifacts support inspection of scripts, source-level count summaries, numerical comparisons and figures, but **do not independently enable complete end-to-end retraining**. The 37-image independent test set yields exploratory FID/KID measurements; three-seed standard deviations are not uncertainty over new test datasets. The adapted DF-GAN–MiniLM comparator is not canonical DAMSM-based DF-GAN. Four TiPGAN class-specific runs were image-conditioned and train-only; no matched held-out TiPGAN FID/KID or text adherence is claimed. Human-expert semantic and cultural assessment has not been completed.

## Exact paper statement

> The research code and publicly shareable evidence supporting this study are available in the Endek T2I GitHub repository at https://github.com/wiwik-instiki/EndekGAN-T2I-with-Preprocessing. The public release includes preprocessing and evaluation scripts, sanitized notebooks, aggregate dataset counts, three-seed numerical evaluation results, and selected figures. The complete Balinese Endek source-image dataset, derivative training crops, source-linked prompts and private manifests, model checkpoints, and full raw training outputs are not publicly redistributed because their provenance and reuse permissions have not been fully established. Accordingly, the public artifacts permit inspection of the reported methods and results but are insufficient on their own for independent end-to-end retraining.

For reproducible citation, record the exact Git commit used at final submission rather than relying solely on the moving main branch. Reconciled on 20 November 2026.
