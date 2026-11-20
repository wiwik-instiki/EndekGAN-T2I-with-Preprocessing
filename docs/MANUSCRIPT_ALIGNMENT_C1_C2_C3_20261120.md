# Endek T2I IEEE Access — C1–C3 / RQ1–RQ2 alignment (20 November 2026)

This is a paper-aligned evidence synopsis, not a new training experiment.

## C1 / RQ1: Source-aware preprocessing
The 296/37/37 original train/validation/test sources were source-partitioned first. Arm A used 944 derived training examples from 296 original train sources; Arm B added 336 training-only crops for 1,280 total. Checkpoint validation KID favored Arm B for both no-seam and seam models. Newly completed held-out Arm A evaluation on Kaggle (three seeds 42/123/2025, two model variants) shows **6/6 paired pooled FID and KID comparisons favor Arm B**. Class-wise FID and KID favor Arm B in **22/24 pairings each**. The result supports the contribution's empirical direction but does not prove a causal mechanism separate from differing training exposure and budgets.

## C2 / RQ2: Continuity/fidelity/diversity trade-off
Three-seed pooled No-Seam versus Proposed means: FID 254.474 vs 302.127 (lower better); KID 0.12848 vs 0.19085 (lower better); one-pixel opposing-border gap 0.08993 vs 0.05338 (lower better); within-prompt LPIPS diversity 0.25630 vs 0.03315 (larger values mean more observed variation). The proposed improves the specific border measure but reduces aggregate fidelity and diversity. Neither LPIPS nor a one-pixel border gap establishes prompt semantics, cultural suitability or true textile tileability.

## C3 / Evidence traceability
The private Kaggle run output has 1,617 SHA-256-verified files and 1,536 generated images, covering six Arm A configurations. The sanitized counts and aggregations are public; private checkpoints, training samples, and prompt-linked source metadata remain restricted. The near-duplicate visual review includes a flagged train–validation pair not conclusively resolved by acquisition records.

## Reproducibility and guardrails
Reference population: 37 independent test sources with class counts flora 15, fauna 5, decorative 11 and geometric 6. Fake 256 images/run, 64/class. Results are exploratory, three-seed SDs are not confidence intervals or hypothesis tests. External adapted DF-GAN–MiniLM is not compute-/architecture-matched. TiPGAN was image-conditioned and only train-evaluated for its own settings, not tested on the held-out text-to-image benchmark.

See figures 1–13 in the final 720-dpi folder and the Arm A/B comparison CSVs in the results directory. Figures 12 and 13 are regenerated from verified measurements, but visually may differ from embedded Word raster panels.
