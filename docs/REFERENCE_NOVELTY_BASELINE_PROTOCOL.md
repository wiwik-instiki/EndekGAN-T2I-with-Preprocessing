# Research Alignment: Literature -> Gap -> Novelty -> Comparison -> Evidence

**Proposed title:** **EndekGAN-T2I: Text-Conditioned Seam-Consistent Synthesis for Balinese Endek Textile Heritage**

**Document status:** Research positioning and experiment traceability, not a claim of IEEE acceptance or benchmark superiority.

## 1. Main reference (one journal article)

He, H., Sun, Z., Fan, J., & Mok, P. Y. (2025). **TiPGAN: High-quality tileable textures synthesis with intrinsic priors for cloth digitization applications.** *Computer-Aided Design*, 183, 103866. https://doi.org/10.1016/j.cad.2025.103866

Publisher: https://www.sciencedirect.com/science/article/pii/S0010448525000284

Official code: https://github.com/VickyHEHonghong/TiPGAN

**What the paper actually solves:** generating tileable cloth-oriented textures, using Patch Swapping and Patch Tiling modules, evaluated with Relative Total Variation (RTV) among other measures. These are contributions of TiPGAN, *not* ours.

**Gap specifically pertinent to EndekGAN:** the publicly released TiPGAN trains/infers from an input image texture (e.g. \`--img_path\`), not arbitrary Indonesian-language text prompts about the Endek motif, motif category, colors, layout and cultural design attributes. This is an *unaddressed capability in that work*, not evidence of a flaw in its tileability performance. TiPGAN may be superior on some tileability metrics; experiments are required.

**Proposed direction:** preserve seam-aware textile design while permitting text-driven motif generation and controlling image-source leakage in a limited-data, cultural-heritage setting.

## 2. Exactly one additional paper-based comparator

Tao, M., Tang, H., Wu, F., Jing, X.-Y., Bao, B.-K., & Xu, C. (2022). **DF-GAN: A Simple and Effective Baseline for Text-to-Image Synthesis.** *Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)*, 16515–16525. https://doi.org/10.1109/CVPR52688.2022.01602

Paper: https://mlanthology.org/cvpr/2022/tao2022cvpr-dfgan/
Official implementation: https://github.com/tobran/DF-GAN

**What it solves:** text-conditioned image synthesis using one-stage generation, deep text-image fusion and a target-aware discriminator. **Distinct scope gap:** standard DF-GAN does not explicitly target periodic textile tileability with opposing-edge continuity penalties. A comparison must establish whether such a penalty helps without harming fidelity and semantic alignment. CVPR is a major peer-reviewed *conference*, not an indexed journal article.

## 3. Existing proposed model, versus genuinely new V2 aspects

**Already in prior EndekGAN study:** structured Indonesian design prompts, multilingual text encoder, separate condition projectors, conditional generator/discriminator and a seam-aware border loss. Previous repository: https://github.com/wiwik-instiki/EndekGAN-T2I

**Additional in present V2:** documented source-aware fixed splits, class-balanced offline image augmentation, ablation of adding 336 derived crops, trained A/B controls on three seeds, checkpoints and validation logs for 24 model-seed runs.

**Conceptual model:** input prompt + class -> frozen multilingual text feature / class embedding -> condition projection -> conditional generator -> generated 256 x 256 RGB motif. Discriminator supports adversarial and feature-matching losses. Seam-aware run includes a lambda=0.5 opposite-edge consistency term; the transformer-only run sets lambda=0. Preprocessing arms do not modify the independent source-image population.

**Mandatory integrity statement:** text conditioning + seam loss are not new architectural inventions relative to the team's earlier study. The current V2 is primarily a **controlled preprocessing/retraining extension**. Do not claim a wholly new text/seam mechanism unless its architecture is actually implemented and compared with previous work.

## 4. Novelty matrix (paper-anchored, explicitly testable)

| Aspect | TiPGAN (main, CAD 2025) | DF-GAN (additional, CVPR 2022) | EndekGAN-T2I V2 (current) | Evidence / gap |
|---|---|---|---|---|
| Tileable texture purpose | Explicit, intrinsic priors and Patch modules | Not textile-specific | Opposite-border regularization | V2 seam numerical metrics exist only for one tested checkpoint |
| Prompt-based text-to-image | Not in publicly released image-driven code | Yes | Yes, frozen multilingual encoder and class | Text counterfactual adherence not verified |
| Source-aware small-data training evaluation | Per-image cloth texture training approach | Usually tested on established text-image benchmarks | Fixed 296/37/37 source grouping; A vs B ablation | Structural audit passed, but no new unique source images |
| Cultural Endek relevance | Generic cloth materials | Generic text-image synthesis | Endek Bali domain | Independent cultural/heritage validation pending |
| Public V2 Endek head-to-head | Not yet retrained/adapted | Not yet retrained | 24 training jobs; 1 held-out final test | **Cannot assert winner** |

## 5. Experiment design and comparison logic

**Exact target comparison:** *TiPGAN (main reference)* vs *EndekGAN V2 (proposed)* vs *DF-GAN (one additional comparator)*. The V2 internal Transformer without seam is an **essential ablation**, but not an additional externally cited comparator.

- **Internal same-task comparison:** V2 Transformer vs V2 Seam Proposed, same split, seed, prompt protocol and generated sample budget. This isolates border regularization. Also compare A vs B using matched seeds to isolate preprocessing/crop effects.
- **DF-GAN same-task comparison:** adapt/retrain original DF-GAN using the actual Endek training split, evaluate on the *same held-out* 37 images and associated prompts, 256-px outputs, same seed protocol and image budget. Old DF-GAN outputs from the other GitHub repo do **not** constitute a V2 head-to-head.
- **TiPGAN cross-task comparison:** TiPGAN synthesizes texture **from an image**, whereas proposed and DF-GAN are **from text**. It can support an explicitly labeled cross-task *tileability* comparison, but cannot be scored for adherence to prompts it does not consume. Prevent leakage: do not use held-out test-reference images as training/conditioning inputs and then claim unbiased distributional test results; design a separate reference-image protocol with disclosure, or restrict TiPGAN comparison to training-source textures and report it as a different task. Match output dimensions and tile visualization; implement TiPGAN's RTV accurately if cited.
- **Text semantic intervention:** fixed latent vector and class, vary prompts in color, motif source, stylization, layout; report measurable class/prompt alignment as well as qualitative changes. No expert-backed score is currently recorded.
- **Heritage assessment:** blind Endek experts evaluate recognizability, prompt fidelity, cultural appropriateness and design feasibility *after* AI image generation; do not imply expert annotation of the training dataset or verified craft reproducibility.

## 6. Current verified experimental evidence — 16 November 2026

| Fact | Verified status |
|---|---|
| Original unique images | **370** total; **296** training, **37** validation, **37** test |
| A preprocessed train records | **944** derived full/augmented files, **296** independent sources |
| B preprocessed train records | **1280** derived files = 944 full/augmented + 336 crops, **296** independent sources |
| Number of model–seed training runs completed | **24/24**, all best/last checkpoints present; 3 training seeds 42/123/2025 |
| Arm B Transformer validation KID | **0.126686** three-seed mean (checkpoint-selection validation, lower better) |
| Arm B Seam Proposed validation KID | **0.184445** three-seed mean — **worse than Transformer** |
| Completed V2 held-out test | **ONLY B / Seam Proposed / Seed 42**, 256 fake and 37 real test references |
| Single-model held-out pooled FID / pooled KID | **302.47055 / 0.1928879**, exploratory; not macro-class aggregates |
| Test metrics of other 23 V2 runs | **Not completed** |
| TiPGAN and DF-GAN on exactly matched V2 protocol | **Not completed** |
| Crop-level semantics and blinded Endek-expert scoring | **Not validated** |
| More independent Endek images from external sources | **Not incorporated** |

**Do not claim:** 1280 unique original Endek motifs, generalization proven by the small test set, full experiment-level baseline superiority, expert-endorsed cultural authenticity or novel invention of a loss already present in the team's earlier code.

## 7. Research questions and falsifiability

- **RQ1 — Tileability vs text control:** Does text-conditioned EndekGAN reduce tile discontinuity relative to an otherwise identical no-seam GAN without unacceptable FID/KID and text-adherence deterioration? This is an *hypothesis*, not an established result.
- **RQ2 — Source-aware preprocessing:** Does B outperform A on held-out quality, diversity, text fidelity and seam consistency under fixed independent-source splits and matched training budgets? Additional training files are derived; sample count alone cannot prove generalization.
- **RQ3 — External literature comparison:** How do TiPGAN, DF-GAN and the proposed GAN trade off tileability, fidelity, prompting capability and cultural relevance? The modalities differ; do not report deceptive single-score rankings.
- **RQ4 — Cultural evaluation:** Are generated designs recognizable and acceptable to Endek domain experts? Human evaluation is pending and must be separately consented and methodologically designed.

## 8. Next required work before IEEE article

1. Freeze current V2 training evidence; do not rerun 24 completed training jobs.
2. Verify image permissions, existing split overlap and one flagged cross-split visually near-similar pair; repair crop prompt semantics.
3. Run held-out generation/evaluation for V2 internal transformer and proposed across all three seeds, with identical conditions. Preserve sample indices and prompts.
4. Adapt/retrain DF-GAN for the *same* V2 text-image task. A distinct TiPGAN adaptation should be designed carefully as an image-reference texture task; disclose that asymmetry explicitly.
5. Compare quality (FID/KID cautiously with n=37), diversity (LPIPS), seam gap, 2x2 tiling, and true RTV only with faithful implementation.
6. Perform fixed-noise prompt ablations and an independent blinded Endek expert study.
7. If demanding **architectural** novelty, implement and ablate a *new* mechanism beyond the old EndekGAN design. Source-aware preprocessing alone is not proof of new generative architecture.

## Scholarly positioning

The Endek heritage motivation is a legitimate research motivation and potential societal use, **not** proof that a generator preserves intangible heritage. Avoid claims about adoption, authentication and weaving readiness without independent data. A transparent negative result (reduced seams at the cost of distributional fidelity) may be scientifically meaningful.

**Citation scope:** main TiPGAN (peer-reviewed Computer-Aided Design article, 2025); comparator DF-GAN (IEEE/CVF CVPR proceedings, 2022). Additional bibliography can be added as verified Related Work, but the designated direct external benchmark matrix remains one main + one additional method.
