# Methods evidence audit: near-duplicate review, baseline fairness and metric protocol

**Research snapshot: 20 November 2026.** This public-safe summary aligns Section III of the Endek T2I IEEE Access manuscript with inspected experiment code and run evidence. The full original Endek image dataset and checkpoint weights remain private.

## Source-group and near-duplicate integrity

The source-grouped corpus contains 296 training, 37 validation and 37 held-out test photographs. Arm A includes 944 derived full-frame views; Arm B includes 1,280 training views including 336 additional crops, from the same independent-source split. There was no exact source SHA-256 overlap across partitions.

Screening with a 64-bit perceptual difference hash (dHash) and Hamming-distance threshold of 4 flagged exactly one cross-partition visual similarity candidate, Flora TRAIN versus Decorative VALIDATION. Direct side-by-side visual inspection indicated distinct ornament geometry and different pink/white versus maroon/gold color arrangements. The grayscale pixel correlation after common resizing was approximately 0.119, and normalized mean absolute RGB-pixel difference was approximately 0.266. The images were not adjudicated to be an identical photograph or an obvious direct resized/cropped copy; the dataset was not retrospectively altered.

**Provenance limitation:** original physical textile acquisition identifiers were not independently verified. Thus photographic difference does not certify independence of the physical textile source. The residual acquisition-level uncertainty is retained explicitly. The flagged pair is train–validation, not test.

## Fairness of model comparisons

| Protocol factor | Proposed vs. no-seam control | Adapted DF-GAN–MiniLM |
|---|---|---|
| Training seeds | 42, 123, 2025 | 42, 123, 2025 |
| Data | Same source-grouped Arm B, 296 train/37 val/37 test | Same source-grouped Endek population |
| Resolution | 256 x 256 | 256 x 256 |
| Batch size / sample exposure | 4; 512 sampled examples per epoch | 4; 512 sampled examples per epoch |
| Nominal updates/epoch | 128 | 128 |
| Text encoder | Frozen multilingual MiniLM, 384-D | Adapted frozen multilingual MiniLM, 384-D |
| Noise vector | 128-D | 100-D |
| Learning rates G/D | 2e-4 / 1e-4 | 1e-4 / 4e-4 |
| Adam beta2 | 0.999 | 0.9 |
| Validation-KID interval | 10 epochs | 20 epochs |
| Paired control | **Only configured internal difference: seam coefficient 0 vs. 0.5** | Independent architecture, not matched for parameters/FLOPs |

All three paired internal model configs were checked against the original saved run configuration files. Since early stopping selected different epochs, the comparison is **not identical in actual training duration**. The external baseline is data-/task-matched for its text-conditioned setting, **not optimizer-, architecture-, or compute-matched**. TiPGAN uses source-image conditioning; only four train-only single-reference fits exist and they are excluded from the matched text-to-image score table.

## Metric implementation

- FID and KID use TorchMetrics with 2048-dimensional Inception features, uint8 RGB image input at 256 x 256 (normalize=False). Each model–seed evaluation uses 37 independent real references and 256 generated images.
- KID uses 50 random subsets, size 37 for the pooled evaluation; per-class subset size is limited by the original reference counts (5–15 images). FID/KID must be interpreted as exploratory given the tiny independent reference set.
- **Pooling confound:** the real test set has 15 Flora, 5 Fauna, 11 Decorative, 6 Geometric originals, whereas generated outputs have **64 per class**. Pooled FID/KID reflect this class-mixture difference as well as image fidelity; they are not class-prevalence matched.
- LPIPS diversity uses AlexNet-based perceptual distance for 40 sampled image pairs sharing an identical prompt per class and seed (sampling RNG 12456). A macro-average across classes measures within-prompt output variation, not semantic adherence.
- Opposite-border gap is the mean of absolute differences between the first and last columns and rows of saved RGB images normalized to [0,1]. It is distinct from the four-pixel training seam penalty computed on generator outputs in [-1,1].
- Checkpoint-selection validation KID uses ten random 25-image subsets on the 37-source validation set; it is not held-out KID. Model statistics are reported as mean plus/minus sample SD over three seeds, not confidence intervals or independent new test datasets.

## Remaining scientific boundaries

Physical source acquisition independence is not fully proven; external baseline FLOP/parameter parity is not established; small-sample FID/KID remain unstable; no independent human expert has certified prompt adherence, cultural authenticity or actual weaving feasibility. New runs and source images are not implied by this documentation.

**Inspection links:** [preprocessing/training/evaluation engine](../scripts/endek_v2_preprocessing_training_engine.py), [results and nine model–seed rows](../results/DFGAN_EndekGAN_real_matched_three_seeds.csv), [data-availability statement](CODE_AND_DATA_AVAILABILITY_IEEE.md), and [original-source dataset description](DATASET_PREPROCESSED_IEEE_20261117.md).
