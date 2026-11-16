# EndekGAN-T2I V2: Training Completed, Benchmark Pending

## Latest verified local status (16 November 2026)
- **24/24** model-seed training jobs COMPLETED; best and last checkpoints verified locally.
- 18 B-arm jobs (six models x three seeds) and six A-arm controls (two models x three seeds).
- The **two** completed V2 independent held-out tests are B / seam_proposed / seed 42 and B / transformer / seed 42, each with 256 generated images and the same 37 original test images.
- External TiPGAN and DF-GAN **have not yet been trained/reproduced on the exact V2 Endek benchmark**. The older repository's adapted DF-GAN belongs to a different split/protocol and must not be counted here.
- V2 test evaluation for the other 22 internal run combinations remains pending.

## Data and optimization
- Original independent images: 370, with 296 training, 37 validation, 37 test.
- A: 944 derived training rows; B: 1280 derived rows = 944 + 336 crops. **Both contain 296 unique original train-source images**.
- Batch size 4, 512 balanced sampled rows/epoch and 128 optimizer updates/epoch. Best checkpoint selected using validation KID only.
- Text conditioning: multilingual encoder and class embedding. Proposed border regularization with seam_lambda=0.5; paired Transformer control seam_lambda=0.
- Crop-derived prompt semantic alignment remains unverified.

## Reproduce the public summary without GPU retraining
Run the analytical script scripts/build_ieee_release.py within an authorized D-drive working copy adjacent to the V2 local run evidence. Its output is the 24-run CSV data, eight summary figures, and tables. The public repo does not contain original/private source images or trained weights.

The public notebook has TRAIN_NOW=False, RUN_ALL_INTERNAL=False, EVALUATE_NOW=False to avoid accidental training or re-evaluation.

Read docs/REFERENCE_NOVELTY_BASELINE_PROTOCOL.md for the main TiPGAN journal reference, additional DF-GAN comparator, and fair experiment protocol.

**Do not claim** 1280 unique motifs, complete cross-model held-out tests, independent expert acceptance, or architectural novelty absent from actual code.
