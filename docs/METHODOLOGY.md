# EndekGAN V2: Dataset and Experiment Methods

## Independent source corpus and provenance
After source-level deduplication, Endek corpus contains **370 unique original photographs** from four classes: flora, fauna, dekoratif and geometris. Source-grouped split: **296 train / 37 validation / 37 test**. Source-level grouping prevents direct leakage from augmentation/crops, but an audit flagged one visually close dHash pair crossing split boundaries for manual review. A near pair is not necessarily a duplicate.

## Preprocessing treatments
| Arm | Derived train files | Unique independent training sources | Crops |
|---|---:|---:|---:|
| A: full-size resize and photometric augmentation | 944 | 296 | 0 |
| B: full-size resize, augmentation, plus heuristic crops | 1280 | 296 | 336 |

Source images are resized from 512x512 to 256x256. Validation and test images are not cropped/augmented. The inherited prompt for each crop has **not** been independently checked to match the crop content. The extra files improve exposure diversity, not the original unique Endek-image population. External sources under examination were not incorporated into this 370-image corpus.

## Experiment and truthful status
V2 trained 24 completed model x seed runs: six B-arm internal variants and two A-arm controls, each at seeds 42, 123 and 2025. Validation KID selected best checkpoints; training does not use test scores for model selection. The three-seed mean validation KID for B Transformer is **0.126686**, lower than B Seam Proposed **0.184445** (lower is better).

Both **B/seam_proposed** and **B/transformer** have completed matched V2 held-out tests for all three seeds (42, 123 and 2025); small 37-real-image test FID/KID are exploratory. TiPGAN (image-conditioned tileable textures; primary 2025 Computer-Aided Design paper) and DF-GAN (text-conditioned synthesis; second 2022 CVPR paper) require different experimental treatment. Adapted DF-GAN–MiniLM has completed three matched held-out tests (training seeds 42, 123, 2025). TiPGAN completed four image-reference-conditioned single-source train-only runs (Flora/Fauna/Decorative at 50,000 iterations; Geometric at 48,000 with a train-only stopping rule), but no matched independent TiPGAN held-out test is available. See [Code and Data Availability](CODE_AND_DATA_AVAILABILITY_IEEE.md) and [current evidence status](../results/EVIDENCE_STATUS.json).

Cultural validity, text faithfulness and weavability are **not** established by image statistics. Experts must evaluate AI-generated output in the final phase. Prior EndekGAN already included text conditions and seam loss; V2 improves preprocessing comparison but requires additional independent technical novelty for strong new architecture claims.

See docs/REFERENCE_NOVELTY_BASELINE_PROTOCOL.md and tables/TABLES_AND_CAPTIONS.md.
