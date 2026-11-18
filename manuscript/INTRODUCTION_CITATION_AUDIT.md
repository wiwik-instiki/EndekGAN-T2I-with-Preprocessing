# Endek T2I — Introduction CARS, APA 7 and novelty audit

**Target:** IEEE Access. **Method:** Endek T2I (no V2 suffix).
**Current working title:** Source-Aware Preprocessing and Seam-Regularized Text-to-Image Synthesis of Balinese Endek Motifs.

## Introduction authoring controls

The Introduction is five cohesive paragraphs (~826 words), written in natural scientific English with APA 7 author–date citations for editing through Mendeley. **There are no experimental result values or numerical performance claims in the Introduction**; values for FID, KID, LPIPS, crop counts, and independent training/test images belong in Materials and Methods and Results and Discussion.

## CARS and source–claim alignment

| Paragraph | Function | Key evidence |
|---|---|---|
| P1 | Cultural relevance, artisan regeneration, potential for educational design exploration | Arfan et al. (2024), Government of Bali (2026) |
| P2 | Existing Endek and Batik text-to-image literature; distinction between prompt realism and tileability | Hendrawati et al. (2024), Purnawan & Suparjana (2026), Wahid et al. (2025) |
| P3 | Primary paper TiPGAN (image-conditioned repeatable textiles), secondary DF-GAN (text-conditioned generation) | Rodriguez-Pardo & Garces (2023), He et al. (2025), Tao et al. (2022) |
| P4 | Specific gap, source-independent preprocessing risk, testable RQs | Karras et al. (2020), primary and secondary method contrasts |
| P5 | Endek T2I main contribution; contributions 1–3 tied explicitly to RQ1 and RQ2 | Actual experimental protocol in this repository; not externally attributed |

**RQ1** evaluates source-aware preprocessing, including training-only crops, with source-level split protection and validation criteria. **RQ2** examines seam regularization's effect on tileability and its implications for fidelity and diversity. The main contribution is a reproducible method-and-evaluation design, not a claim to be the first to invent conditional generation or seam loss.

## Review status and exceptions

The literature citations are retained in `bibliography/introduction.bib`; paper author order for the 2024 TEMATIK study follows its published PDF (Hendrawati, Wulandari, Ginantra), despite differences in Garuda metadata. DF-GAN and the Karras GAN-data-efficiency study are peer-reviewed **conference papers**, while the Bali Government notice is the primary nonjournal evidence for cultural-policy urgency.

The old EndekGAN project already contained text conditioning and seam penalties; their components are not represented as newly invented in this Introduction. The complete external TiPGAN and DF-GAN matched benchmark and Endek expert review remain pending and cannot support performance-superiority claims. APA 7 author–date citations are for research-draft/reference-manager editing; IEEE Access requires IEEE-style numbered citations at submission.

This file describes the manuscript Introduction only and does not override the experiment status audit. The original Endek dataset, prompts, checkpoints and numerical experimental outputs were not changed by this revision.
