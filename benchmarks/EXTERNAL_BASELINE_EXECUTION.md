# External baseline execution audit — 17 November 2026

## Experimental identity and citation

**Primary paper:** He et al., *TiPGAN: High-quality tileable textures synthesis with intrinsic priors for cloth digitization applications*. Computer-Aided Design, 183, 103866 (2025). [DOI](https://doi.org/10.1016/j.cad.2025.103866). [Official code](https://github.com/VickyHEHonghong/TiPGAN).

**Additional paper:** Tao et al., *DF-GAN: A Simple and Effective Baseline for Text-to-Image Synthesis*. IEEE/CVF CVPR (2022). [DOI](https://doi.org/10.1109/CVPR52688.2022.01602). [Official code](https://github.com/tobran/DF-GAN).

**Proposed:** EndekGAN-T2I V2 — prompt- and class-conditioned GAN with a seam-regularization ablation and source-aware preprocessing. Earlier EndekGAN already included text conditioning and seam loss; V2 does not claim their invention.

## Completed actual experiment

Arm B **Transformer vs Seam Proposed × seeds 42, 123, 2025**: six held-out tests, all with 37 original real references and 256 generated images per model/seed. In this internal ablation, Seam Proposed has a 40.6% lower mean opposite-border gap; mean FID/KID and LPIPS diversity are worse. See [three-seed numerical report](THREE_SEED_TEST_REPORT.md), [Figure 12](../figures/fig12_heldout_B_test_three_seeds_tradeoff.png), [Figure 13](../figures/fig13_heldout_B_test_three_seed_mean_sd.png) and [Figure 14](../figures/fig14_actual_B_generated_motifs_3seed_comparison.png).

**All scores are exploratory**: independent test real n=37; per-class n=5–15. No three-model test ranking exists.

## TiPGAN official code: completed training smoke test ONLY

- Official source checkout and source commit are recorded in `results/REFERENCE_BASELINE_PREFLIGHT.json`.
- The official VGG weights (`vgg_conv.pth`, obtained from the research authors' Hugging Face repository) were installed locally in the private D-drive experiment workspace.
- OpenCV and TensorBoard dependencies were installed in a D-only auxiliary package directory without changing EndekGAN's main PyTorch/CUDA version.
- The official TiPGAN `train.py` was exercised with **one preselected FLORA source image from the 296-image training partition**, batch size 1, **30 optimizer iterations**. Process returned exit code 0, created its checkpoint and saved a pilot log. The first attempt failed due to a Windows log-path-length issue and was rerun using a shorter D-drive output path; the original failed log was preserved.
- **No held-out images were used as TiPGAN input**.
- The authors' default budget is **50,000 iterations per reference texture**. Thirty iterations are only compatibility evidence, **NOT a converged TiPGAN result**, and no FID/KID or RTV number is claimed.
- TiPGAN inputs a reference texture image. It **cannot receive a text prompt**. Text-to-image alignment is *not an applicable metric* for this baseline.

## DF-GAN official architecture with MiniLM384 adaptation: one-epoch pilot ONLY

- The publicly released `NetG`, `NetD`, and `NetC` from the original DF-GAN repository were imported without silently replacing them with an EndekGAN generator.
- The canonical pretrained **DAMSM** text encoder was **not available for Endek**. The study therefore uses an explicitly named **adapted DF-GAN–MiniLM**, conditioning on the exact same frozen multilingual MiniLM mean-pooled, unit-normalized 384-D features as EndekGAN V2.
- A local GPU architecture forward/backward smoke test passed with batch 2 at 256×256.
- A real, fixed one-epoch training test (seed 42, Arm B, 512 class-balanced presented images, batch size 4, **128 optimizer-update pairs**) then ran successfully, with official DF-GAN adversarial/mismatch/gradient-penalty structure. The V2 training corpus consists of **1,280 derived views from 296 independent original Endek images**.
- One-epoch pilot checkpoint, training log and status were saved privately on D. No validation-based model selection or held-out test occurred. **Do not place one-epoch pilot numbers in an IEEE performance comparison table**.
- DF-GAN's official repository specifies a minimum 12 GB GPU; the 8 GB local laptop successfully completed only the adapted feasibility training. The 12 GB threshold is a published recommendation, not a mathematical impossibility on reduced batch sizes.
- A PRIVATE Kaggle GPU kernel, dataset staging and aligned frozen embeddings were prepared for training under a larger accelerator. **Submission and completion must be independently verified**; there are no published multi-epoch DF-GAN V2 weights or test metrics at the time this document was authored.

## Data protocol, rights and leakage precautions

- Only existing verified V2 independent source split: **296 train / 37 validation / 37 held-out test** (370 total images).
- Arm A: 944 source-derived training views. Arm B: 1,280 source-derived training views including 336 heuristic crops.
- Crop captions are inherited from whole source images and have not been expert-checked for semantic fidelity.
- Reference-image TiPGAN training uses **only TRAIN** source images, not val or test as input.
- Text-conditioned models use validation KID for checkpoint selection; held-out test prompts/images only enter inference after training.
- Original images, private prompts and external baseline weights are **NOT included** in this public release. The large private cloud dataset is restricted to the owner's Kaggle account.
- The dataset does not contain 1,280 independent motifs. Cultural authenticity and feasibility of woven production remain unverified.

## Publication-readiness decision

**TiPGAN (primary reference) versus DF-GAN (additional reference) versus EndekGAN V2** remains an **incomplete external benchmark**. The completed six-test V2 ablation and two software pilots are reproducibility evidence, not proof that EndekGAN wins.

Before IEEE claims: complete TiPGAN per-image converged training at a declared reference budget and tileability protocol; complete the adapted DF-GAN training across three seeds and hold-out tests; measure matched seam gap and culturally grounded prompt fidelity using independent experts; report input modality asymmetry and limitations.
