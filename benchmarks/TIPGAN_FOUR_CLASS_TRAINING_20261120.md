# TiPGAN: verified four-category train-only experiments (20 November 2026)

Official TiPGAN image-reference-conditioned implementation, commit fc465f2626319ad90456f697aab33669b952e9d4, used one **train-source reference image** per category. These are four distinct one-image texture-training runs, not a multiclass TiPGAN model, not a text-to-image comparator and not an independent test.

| Endek category | Platform | Final iteration | Training end | Checkpoint |
|---|---|---:|---|---|
| Flora | Kaggle | 50,000 | Finished | Verified locally |
| Fauna | Local GPU | 50,000 | Finished | Verified locally |
| Decorative | Local GPU | 50,000 | Finished | Verified locally |
| Geometric | Kaggle | 48,000 | Train-only early stopping | Verified locally |

Geometric early stopping used a training-source-only L1/style/histogram proxy, minimum 12,000 iterations, check interval 2,000, patience 8, relative minimum improvement 0.001. The normal exit at iteration 48,000 is **not** evidence of independent validation. All four end-point checkpoints are archived privately; transfer hashes were verified. No private photos, captions, weights or machine paths are included here.

**Interpretation:** training samples, training loss, early stopping and saved weights are not independent FID/KID/LPIPS/tileability or text-adherence evidence. A leakage-free, task-appropriate held-out reference evaluation and expert cultural assessment are still required.

[Public-safe training status CSV](../results/TIPGAN_FOUR_CLASS_TRAINING_STATUS_20261120.csv).
