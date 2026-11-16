# Tables and figure legends for pre-manuscript review

## Table 1. Dataset sources and derived training records
Table 1 distinguishes independent source images from their derived training files.

| Arm | Split | Derived files | Unique sources | Unique prompts | Crops |
|---|---|---:|---:|---:|---:|
| A | train | 944 | 296 | 296 | 0 |
| A | val | 37 | 37 | 37 | 0 |
| A | test | 37 | 37 | 37 | 0 |
| B | train | 1280 | 296 | 296 | 336 |
| B | val | 37 | 37 | 37 | 0 |
| B | test | 37 | 37 | 37 | 0 |

**Table 1.** Source-aware split; augmentation/cropping does not create independent sources.

## Table 2. Training outcomes for all 24 model-seed jobs
Table 2 lists real checkpoints selected by validation KID.

| Arm | Model | Seed | Epochs | Best epoch | Best validation KID |
|---|---|---:|---:|---:|---:|
| B | cat | 42 | 290 | 210 | 0.135074 |
| B | cat | 123 | 320 | 240 | 0.171566 |
| B | cat | 2025 | 330 | 250 | 0.115807 |
| B | hash_text | 42 | 310 | 230 | 0.169987 |
| B | hash_text | 123 | 290 | 210 | 0.187841 |
| B | hash_text | 2025 | 340 | 260 | 0.109781 |
| B | transformer | 42 | 290 | 210 | 0.096543 |
| B | transformer | 123 | 290 | 210 | 0.086050 |
| B | transformer | 2025 | 390 | 310 | 0.197465 |
| B | transformer_noclass | 42 | 280 | 200 | 0.147590 |
| B | transformer_noclass | 123 | 290 | 210 | 0.175965 |
| B | transformer_noclass | 2025 | 280 | 200 | 0.102836 |
| B | seam_proposed | 42 | 280 | 200 | 0.184922 |
| B | seam_proposed | 123 | 360 | 280 | 0.197887 |
| B | seam_proposed | 2025 | 290 | 210 | 0.170528 |
| B | seam_noclass | 42 | 320 | 240 | 0.189008 |
| B | seam_noclass | 123 | 380 | 300 | 0.169837 |
| B | seam_noclass | 2025 | 300 | 220 | 0.153707 |
| A | transformer | 42 | 280 | 200 | 0.214679 |
| A | transformer | 123 | 290 | 210 | 0.261280 |
| A | transformer | 2025 | 400 | 370 | 0.215278 |
| A | seam_proposed | 42 | 310 | 230 | 0.235618 |
| A | seam_proposed | 123 | 400 | 340 | 0.239485 |
| A | seam_proposed | 2025 | 310 | 230 | 0.262315 |

**Table 2.** All best/last checkpoints and loss logs were verified.

## Table 3. Three-seed validation scores
Table 3 reports checkpoint-selection KID, NOT independent test-set performance.

| Arm | Model | Seeds | KID mean | KID SD |
|---|---|---:|---:|---:|
| A | seam_proposed | 3 | 0.245806 | 0.014427 |
| A | transformer | 3 | 0.230412 | 0.026734 |
| B | cat | 3 | 0.140816 | 0.028319 |
| B | hash_text | 3 | 0.155870 | 0.040900 |
| B | seam_noclass | 3 | 0.170851 | 0.017672 |
| B | seam_proposed | 3 | 0.184445 | 0.013685 |
| B | transformer | 3 | 0.126686 | 0.061520 |
| B | transformer_noclass | 3 | 0.142131 | 0.036869 |

**Table 3.** Mean ± sample SD across three training seeds; validation n=37.

## Table 4. Available exploratory test scores
Table 4 preserves the first proposed-only test evaluation; matched comparisons are shown in Table 5 when completed.

| Test model | Class | FID | KID | Real n | Fake n |
|---|---|---:|---:|---:|---:|
| B seam proposed seed42 | all | 302.471 | 0.192888 | 37 | 256 |
| B seam proposed seed42 | flora | 273.413 | 0.238854 | 15 | 64 |
| B seam proposed seed42 | fauna | 408.497 | 0.460988 | 5 | 64 |
| B seam proposed seed42 | dekoratif | 368.456 | 0.405581 | 11 | 64 |
| B seam proposed seed42 | geometris | 391.349 | 0.355159 | 6 | 64 |

**Table 4.** Class-wise test reference numbers are extremely small (5–15). FID/KID are exploratory.

## Figure legends and interpretation
Figures 1–8 are derived from actual V2 data; qualitative comparisons (when present) must be labeled GAN-generated only.
- Figure 1: independent training sources vs derived image-file exposures.
- Figure 2: training-run validation KID (24 jobs, 3 seeds, 8 settings).
- Figure 3: paired transformer/seam-proposed validation ablation by seed and preprocessing arm.
- Figures 4–6: proposed model training generator, discriminator and seam loss.
- Figure 7: proposed model validation KID trajectory by seed.
- Figure 8: exploratory test KID for the first proposed-only held-out evaluation.

**Limitations:** 370 independent raw images total, 296 training sources, unchanged 37 val and 37 test images. The extra derived files (944 vs 1280) are not new independent motif images. Crop prompts lack independent semantic validation. External source images and independent external model evaluations were not added to this V2 training matrix. No verified expert study or complete three-seed model test comparison is claimed.

## Table 5. Matched two-model held-out V2 test (seed 42)
Same 37 real references, source-group split, fixed prompt samples and latent seed 90731. Test metrics are exploratory.
| Model | Pooled FID | Pooled KID | Class-macro FID | Class-macro KID | Mean opposite-border gap |
|---|---:|---:|---:|---:|---:|
| transformer | 247.819 | 0.111622 | 311.429 | 0.205214 | 0.079515 |
| seam_proposed | 302.471 | 0.192888 | 360.429 | 0.365146 | 0.052456 |

**Table 5.** TiPGAN and DF-GAN were NOT evaluated on V2; these rows are internal V2 model comparisons.
**Figure 9.** Model-generated examples at matching prompt/index/latent seed (seed 42).
**Figure 10.** Matched 2 x 2 repeated-tile preview of generated outputs.
**Figure 11.** One-seed matched per-class test KID.
