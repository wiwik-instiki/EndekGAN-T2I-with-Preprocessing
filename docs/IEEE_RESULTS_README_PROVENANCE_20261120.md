# IEEE Access manuscript results figure and table crosswalk (20 November 2026)

These labels match the final Section IV numbering and the results inserted in the IEEE Access Word manuscript. Figures 1–5 are methodological diagrams/inputs. All panels 9–11 below are byte-identical copies of the saved figures used in the Word Section IV manuscript.

| Manuscript figure | Public exact image | SHA-256 | Experiment role |
|---|---|---|---|
| Fig. 6 | [Figure 6](../figures/fig21_preprocessing_A_B_validation_kid.png) | 24e00f27a81f187af577193f92bd8034a5652f0c6469e881c57dfa0958b6d104 | RQ1: checkpoint-validation KID (Arm A/B; not held-out) |
| Fig. 7 | [Figure 7](../figures/fig23_ieee_results_four_metrics_3seeds.png) | ebd4ff0c6caed7d8722a7032d1b440313a1e45cc3fa70010e01a1c5950fa658d | RQ2: three-model held-out mean and sample SD |
| Fig. 8 | [Figure 8](../figures/fig24_ieee_results_seedwise_comparison.png) | d4fad73a7e39e3e2be3f2eb23dc8a3a568915fa37cce7638d727fd88ca9ed5a4 | RQ2: three-model held-out per seed |
| Fig. 9 | [Figure 9](../figures/fig25_ieee_actual_prompt_matched_three_models.png) | 95470800684e84c618b71d7e4525ce0cf6bf954a3d5dc95e73d95bd2c09cd8dd | Actual prompt-matched seed-42 generated samples (three models; four classes) |
| Fig. 10 | [Figure 10](../figures/fig26_ieee_actual_2x2_tiling_three_models.png) | c9eafc4b905ece1e3cdf2938125337ab5f54a0ca7221f40a1724b9c8b80c29bf | Exact same generated images tiled two-by-two |
| Fig. 11 | [Figure 11](../figures/fig27_ieee_tipgan_training_examples_only.png) | 88a52138683a32689de21b03bbf2fe74c63e094547a9d7c252cc2ecdec567052 | TiPGAN one-source image-conditioned training-only outputs |

| Manuscript table | Grounded source |
|---|---|
| Table V | [24 internal runs](../results/all_24_runs.csv) and [current evidence status](../results/EVIDENCE_STATUS.json) |
| Table VI | [Arm A/B three-seed checkpoint-validation KID](../results/three_seed_checkpoint_validation_summary.csv) |
| Table VII | [Nine matched model–seed tests](../results/DFGAN_EndekGAN_real_matched_three_seeds.csv) and independent summaries |
| Table VIII | [Nine matched model–seed tests](../results/DFGAN_EndekGAN_real_matched_three_seeds.csv) |
| Table IX | [Four TiPGAN train-only class statuses](../results/TIPGAN_FOUR_CLASS_TRAINING_STATUS_20261120.csv) |

**Research integrity:** 37 independent held-out real sources. TiPGAN has no independent matched held-out FID/KID or text-adherence evidence. Fig. 9 uses fixed index 0 and verified same prompts but cannot guarantee identical latent vectors across architectures. Fig. 11 is train-only texture generation. All source photographs, private prompts, full logs and checkpoints remain outside the public repository.

**Dataset rights:** publication of composite generated-image panels does not confer any rights to underlying original motif photographs. The limited original-source samples already present in methodological Fig. 1 require public redistribution permissions to be verified.
