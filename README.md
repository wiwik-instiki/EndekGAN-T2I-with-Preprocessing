# EndekGAN-T2I: Text-Conditioned Seam-Consistent Synthesis for Balinese Endek Textile Heritage

**Research code and verified experimental evidence (November 2026) · Training complete; external reference benchmarks and final paper validation pending**

[![Python 3.11](https://img.shields.io/badge/Python-3.11-blue)](requirements-notebook.txt)
![V2 training](https://img.shields.io/badge/Training-24%20of%2024%20runs%20complete-success)
![Scientific status](https://img.shields.io/badge/Benchmark-partial-yellow)

This project investigates the generation of **Balinese Endek-like textile motifs from Indonesian-language structured prompts** while accounting for the continuity of repeated borders and the risks of small source-image collections. Its application motivation is responsible documentation and exploration of Balinese textile design heritage; **generated motifs are not independently certified as authentic or suitable for actual weaving**.

The key design question is not just whether a GAN can synthesize Endek-like images, but whether *text-conditioned synthesis* can balance semantic guidance, image fidelity, motif diversity and periodic border continuity on a source-audited small dataset.

## 1. From primary journal reference to testable novelty

| Role | Paper | Exact limitation / complementary scope | Relation to V2 |
|---|---|---|---|
| **Primary scientific reference** | **He et al. (2025), TiPGAN**, *Computer-Aided Design*, 183, 103866. [DOI](https://doi.org/10.1016/j.cad.2025.103866) / [official code](https://github.com/VickyHEHonghong/TiPGAN) | Patch Swapping / Patch Tiling create seamless, image-reference-conditioned cloth textures; published code uses an image input rather than arbitrary text prompts. This is a *scope gap*, not an assertion that TiPGAN has poor seams. | Motivates a **text-guided seamless textile generation task**. A valid TiPGAN comparison is cross-task (image-conditioned vs text-conditioned) and must be labeled accordingly. |
| **One additional external comparator** | **Tao et al. (2022), DF-GAN**, *IEEE/CVF CVPR*, 16515–16525. [DOI](https://doi.org/10.1109/CVPR52688.2022.01602) / [official code](https://github.com/tobran/DF-GAN) | A strong one-stage text-to-image GAN, with deep fusion and a target-aware discriminator; canonical design lacks an Endek-specific opposing-border objective. | Intended **same-task text-to-image comparator** on the identical Endek split. |
| **Proposed research system** | **EndekGAN-T2I V2** | Combines multilingual prompt conditioning, a seam objective, source-aware partitioning and augmentation/crop ablations for Endek motifs. | 24 V2 training runs complete; **full paper-level superiority not demonstrated**. |

**Research novelty discipline:** text conditioning and a seam regularizer were **already present in our earlier** [EndekGAN-T2I research](https://github.com/wiwik-instiki/EndekGAN-T2I). V2 contributes a **new controlled source-aware preprocessing/retraining study**, not an independently established new seam-loss invention. Novel architecture claims require an additional actually implemented and ablated mechanism; this repository does not fabricate one.

Read the complete [reference-to-gap-to-novelty protocol](docs/REFERENCE_NOVELTY_BASELINE_PROTOCOL.md) before treating the experiments as an IEEE manuscript.

## 2. Architecture and question-driven experiments

**Proposed:** Indonesian prompt + motif class -> frozen multilingual MiniLM representation + class embedding -> separate generator/discriminator conditioning -> conditional GAN at 256 × 256 -> seam-consistent output regularization.

The proposed model is compared internally with **Transformer (same generator/discriminator, seam loss disabled)** and other actual controlled ablations (category, hashed text, class removal). A- vs B-arm contrasts isolate preprocessing, rather than inventing more independent motifs.

**Required external reference sequence (not yet run on V2 data):**
1. **TiPGAN main reference**: reproduce/adapt a source-image-to-tileable-texture benchmark on permitted training/reference textures. Because TiPGAN does not accept prompts, **do not** compare its text alignment as if it did. Avoid test-as-input leakage.
2. **DF-GAN additional comparator**: train/adapt on the V2 source-independent Endek train split and evaluate on matched held-out prompts/images.
3. **EndekGAN-T2I V2 proposed** plus **Transformer no-seam control**: evaluate on identical splits, seeds, budget, definitions and independent test reports. Do not pool different paper protocols into one ranking.

See the [reference comparison matrix](docs/REFERENCE_NOVELTY_BASELINE_PROTOCOL.md) and the [paper tables](tables/TABLES_AND_CAPTIONS.md).

## 3. Dataset reality: original vs derived files

| Condition | Training files | **Unique training-source images** | Validation unique sources | Test unique sources |
|---|---:|---:|---:|---:|
| **A: full-image + augmentation** | 944 | **296** | 37 | 37 |
| **B: full-image + augmentation + crop** | 1,280 | **296** | 37 | 37 |

There are **370 independent original images in the curated corpus (296/37/37)**. Arm B derives 336 crop files from existing sources; **1,280 does NOT mean 1,280 unique Endek photographs**. Classes: flora, fauna, dekoratif, geometris. All verified splits use source grouping; prompt semantics of the crop-derived files still require checking. External Mendeley/Roboflow resources are **not incorporated** into this verified V2 matrix, and original images and private prompts are not redistributed here.

Data audit output: [dataset_independence_audit.csv](results/dataset_independence_audit.csv).

## 4. What has really been completed?

| Evidence stage | Verified status | Files |
|---|---|---|
| Preprocessing and source-audit controls | Completed | [Methodology](docs/METHODOLOGY.md); [data audit](results/dataset_independence_audit.csv) |
| **Training** 6 B-arm configurations × 3 seeds, plus 2 A-arm configurations × 3 seeds | **24 / 24 completed** on new V2 weights | [24-run log](results/all_24_runs.csv) |
| Best and last checkpoint availability / epoch and validation log consistency | **24 / 24 verified locally** | [Audit summary](results/EVIDENCE_STATUS.json) |
| Three-seed **validation KID** only, used for checkpoint selection | Completed | [Table](results/three_seed_checkpoint_validation_summary.csv), [Figure 2](figures/fig02_all_24_validation_KID.png) |
| Held-out *test* FID, KID, LPIPS and opposite-edge gap | **SIX evaluated V2 models/seeds:** Arm B Transformer and Seam Proposed, each at seeds 42/123/2025 | [matched three-seed held-out FID/KID and seam metrics](results/heldout_B_3seeds_mean_sd.csv), [per-seed values](results/heldout_B_transformer_vs_seam_3seeds_per_seed.csv) |
| **TiPGAN** trained on the exact V2 benchmark | **NOT COMPLETE** | No fabricated metrics |
| **DF-GAN** trained on the exact V2 benchmark | **NOT COMPLETE** | Older [adapted DF-GAN](https://github.com/wiwik-instiki/EndekGAN-T2I) scores are historical, not V2 test scores |
| Text adherence / 3-seed external test / independent Endek expert panel | **NOT COMPLETE** | Required before strong heritage/quality claims |

**Important negative finding:** Arm B Transformer achieves a lower three-seed checkpoint-selection **validation KID mean (0.126686)** than Arm B Seam Proposed **(0.184445)**. Lower is better. We do not claim a fidelity improvement from seam regularization. The first completed V2 held-out evaluation (B seam seed 42) records pooled FID **302.47** and pooled KID **0.19289** with just **37 real** test images. Pooled and per-class macro scores are not interchangeable. The test scores are **exploratory and highly sample-size sensitive**.

## 5. Actual scientific plots, source-backed tables, and images

The figures below were generated from the completed V2 run logs, **not manually invented**:

| Figure | Scientific role |
|---|---|
| [Figure 1 — source vs derived data](figures/fig01_dataset_source_vs_derived.png) | Controls against claiming augmented images as new source photographs |
| [Figure 2 — all 24 validation KID runs](figures/fig02_all_24_validation_KID.png) | True three-seed model comparison on the validation set |
| [Figure 3 — paired seam ablation](figures/fig03_matched_seam_ablation.png) | Which architecture receives seam regularization under matched seeds |
| [Figure 4 — generator loss](figures/fig04_generator_loss.png) | Real training trajectory |
| [Figure 5 — discriminator loss](figures/fig05_discriminator_loss.png) | Real training trajectory |
| [Figure 6 — seam loss](figures/fig06_seam_loss.png) | Real training trajectory |
| [Figure 7 — checkpoint validation](figures/fig07_proposed_validation_trajectory.png) | Model selection history |
| [Figure 8 — one-model test KID](figures/fig08_single_model_test_KID.png) | Explicitly partial test results; no false baseline claim |

**Qualitative image comparisons against TiPGAN and DF-GAN have not been executed under the V2 protocol.** A three-way gallery will be added only after matched outputs genuinely exist. Original/raw Endek photographs, checkpoint weights and private prompts remain local until rights and release restrictions are verified.

## 6. Run and reproduce responsibly

The active Windows experiment workspace is on drive D. Repository files do not contain private datasets or multi-GB checkpoint weights. To reproduce summary tables and Figures 1–8 *locally*, put this repository copy inside the authorized V2 project folder next to \`runs_preprocessed\` and run:

\`\`\`cmd
python scripts\build_ieee_release.py
\`\`\`

The script reads completed V2 artifacts, validates run states and writes aggregated numerical files; **it does not train any GPU model**. To reproduce experiments from scratch, first obtain authorized source image/caption files, validate the official training dependencies in a drive-D environment, and follow the safety flags in [notebook 01](notebooks/01_EndekGAN_T2I_Preprocessing_BEFORE_Training_V2.ipynb) and [notebook 05](notebooks/05_EndekGAN_T2I_Train_Preprocessed_Crop_V2.ipynb). Do **not** run all training cells by default. Original reference TiPGAN and DF-GAN repositories may have incompatible hardware/conditioning requirements; adaptations must be separately documented.

[Source-and-experiment evidence CSV files](results/) · [Pre-manuscript tables](tables/TABLES_AND_CAPTIONS.md) · [Reference-driven protocol](docs/REFERENCE_NOVELTY_BASELINE_PROTOCOL.md)

## New: official-paper benchmark protocol and actual two-way generated image comparison

The official TiPGAN and DF-GAN repositories were inspected and cloned separately. They are **not** yet trained on the V2 Endek source split. The official DF-GAN requires a DAMSM encoder and documents a minimum 12 GB GPU; TiPGAN is a single-image reference texture method, not text conditioned.

**New actual evidence:** Arm B Transformer vs Arm B Seam Proposed, seed 42, on the **same held-out prompts, random latent seed and 64 images/class**. The Transformer achieved pooled FID **247.82**, pooled KID **0.11162**, with opposite-border gap **0.07952**. Seam Proposed achieved pooled FID **302.47**, pooled KID **0.19289**, and lower opposite-border gap **0.05246**. These are **exploratory** one-seed scores with only 37 real references; no FID/KID or texture-quality superiority claim is made.

Read [benchmark report with actual paired images and 2x2 tiling](benchmarks/README.md), [official code/data preflight](results/REFERENCE_BASELINE_PREFLIGHT.json), [comparison matrix](tables/REFERENCE_BENCHMARK_MATRIX.csv), and [safe VS Code notebook 06](notebooks/06_TiPGAN_DFGAN_EndekGAN_Comparison_V2.ipynb).


## Completed matched six-test evaluation, 3 training seeds (update 17 November 2026)

**Completed:** Arm B Transformer and Seam Proposed each at training seeds **42, 123, 2025**, all with source-identical 296/37/37 train/validation/test split, 64 generated held-out samples per class, and fixed evaluation latent seed. Images and private prompt records remain local.

| Held-out test metric | Transformer mean ± SD | Seam Proposed mean ± SD |
|---|---:|---:|
| Pooled FID ↓ | 254.474 ± 40.967 | 302.127 ± 5.459 |
| Pooled KID ↓ | 0.128485 ± 0.054584 | 0.190846 ± 0.004053 |
| Class-macro FID ↓ | 316.145 ± 41.270 | 358.653 ± 8.125 |
| Opposite-border seam gap ↓ | 0.089935 ± 0.014491 | **0.053379 ± 0.001198** |
| LPIPS diversity ↑ | 0.256295 ± 0.216808 | 0.033152 ± 0.041343 |

**Observed trade-off:** opposite-border gap reduced approximately **40.6%** in the three-seed mean, but worsened distributional fidelity and sharply reduced LPIPS diversity. Seed 2025 slightly favored Seam Proposed in pooled FID/KID. FID/KID are *exploratory* with only 37 real test images. No human-rated Endek authenticity is implied.

[Complete scientific three-seed report](benchmarks/THREE_SEED_TEST_REPORT.md) ·
[per-seed CSV](results/heldout_B_transformer_vs_seam_3seeds_per_seed.csv) ·
[mean ± SD](results/heldout_B_3seeds_mean_sd.csv) ·
[Figure 12](figures/fig12_heldout_B_test_three_seeds_tradeoff.png) ·
[Figure 13](figures/fig13_heldout_B_test_three_seed_mean_sd.png) ·
[Figure 14](figures/fig14_actual_B_generated_motifs_3seed_comparison.png)

**External main-reference TiPGAN (image-conditioned) and additional DF-GAN (text-conditioned) have NOT completed a V2 benchmark.** Neither can be represented by values borrowed from the earlier EndekGAN experiments.


## External method execution: source-anchored but not yet benchmark-complete

A source-clean TiPGAN official-code pilot was successfully run for **30 optimizer iterations on one training-only flora image**, with its original VGG weights and a saved pilot checkpoint. The authors' default per-texture training budget is 50,000 iterations, so **this is not a converged benchmark**.

The official DF-GAN NetG/NetD/NetC architecture, explicitly adapted to use **exactly the frozen V2 MiniLM384 text features instead of canonical DAMSM**, passed a local gradient test and **one 128-step training epoch** on the actual 1,280-file Arm B data. This is **not a completed DF-GAN V2 comparison**.

[Execution audit](benchmarks/EXTERNAL_BASELINE_EXECUTION.md) ·
[Machine-readable statuses](results/EXTERNAL_EXECUTION_STATUS.json) ·
[Adapted DF-GAN reproducible Kaggle training script](scripts/dfgan_minilm_v2_kaggle_adapter.py).

**Do not place pilot-only values next to the held-out three-seed EndekGAN FID/KID table.** TiPGAN is image-reference-based and cannot be evaluated for prompt adherence it does not consume.


## DF-GAN–MiniLM held-out evaluation on V2

One PRIVATE Kaggle run now has actual held-out results for adapted DF-GAN–MiniLM seed 42 (37 real images, 256 synthetic images). Pooled FID 250.430; KID 0.169826; seam gap 0.117372. Preliminary, single seed. See [validated evaluation report and Figure 15](benchmarks/DFGAN_MiniLM_HELDOUT_SEED42.md) and [CSV](results/DFGAN_EndekGAN_heldout_seed42.csv). TiPGAN's image-conditioned modality remains a separate comparison.

## 7. Ethics, copyright and claims

Balinese Endek is cultural heritage and a design context, not a license to declare synthetic outputs authentic traditional artifacts. The present experimental captions are structured design descriptions and **are not independently expert-validated semantic ground truth**. An expert-blinded design study and more truly independent source images are needed. Do not release private Endek imagery, hidden source names or trained weights without permission. This repository is an **experimental research record**; no journal acceptance, peer review endorsement or proven heritage impact is claimed.

## References

1. He, H., Sun, Z., Fan, J., & Mok, P. Y. (2025). *TiPGAN: High-quality tileable textures synthesis with intrinsic priors for cloth digitization applications*. **Computer-Aided Design, 183**, 103866. https://doi.org/10.1016/j.cad.2025.103866
2. Tao, M., Tang, H., Wu, F., Jing, X.-Y., Bao, B.-K., & Xu, C. (2022). *DF-GAN: A Simple and Effective Baseline for Text-to-Image Synthesis*. **IEEE/CVF CVPR**, 16515–16525. https://doi.org/10.1109/CVPR52688.2022.01602

This work is a continuation, not a duplication, of the [earlier EndekGAN-T2I experimental repository](https://github.com/wiwik-instiki/EndekGAN-T2I).
