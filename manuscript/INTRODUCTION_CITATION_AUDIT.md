# Penakluk Top Tier — Introduction CARS/APA 7/BibTeX Audit

**Study:** Source-Aware Preprocessing and Seam-Regularized Text-to-Image Synthesis of Balinese Endek Motifs. **Scientific method:** Endek T2I (no V2 suffix).

**Section status:** Working draft for academic review. APA 7 author–date citations intentionally used at this stage to connect references in Mendeley; the final IEEE Access template requires numbered IEEE references.

## CARS structure and provenance

| Paragraph | CARS function | Research source and direct support | Status |
|---|---|---|---|
| P1 | Territory: traditional weaving, sustainable heritage, generational regeneration and educational motivation | Arfan et al. (2024), DOI 10.5614/j.vad.2024.16.2.2; Bali Provincial Government (2026-05-30) official statement | Verified scope; do not claim near extinction as measured fact |
| P2 | Relevant prior art and narrowable gap: 2024 Endek Stable Diffusion; 2026 Endek LoRA with 687 *separate* images; 2025 IEEE Access batik T2I | Hendrawati et al. (2024), Purnawan & Suparjana (2026), Wahid et al. (2025) | Verified; related works exist, so no "first Endek T2I" claim |
| P3 | Primary journal anchor and image-conditioned tileability | Rodriguez-Pardo & Garces (2023), He et al. (2025) TiPGAN, DOI 10.1016/j.cad.2025.103866 | Scope gap of image versus text conditioning (not TiPGAN quality flaw) |
| P4 | Secondary external text-to-image comparator, provenance-aware data limitation and niche | Tao et al. (2022) DF-GAN; Karras et al. (2020) data-efficient GAN | Verified scope; both are conference papers, flagged explicit skill exceptions |
| P5 | Occupy niche: actual mechanism, controlled A/B preprocessing, source-independent three-seed ablation and observed trade-off | repository results/all_24_runs.csv, heldout_B_3seeds_mean_sd.csv, TRAINING_SPLITS_PREPROCESSED_V2_REPORT.json | Original empirical results traced to internal outputs, not external citations |

## Key empirical audit

- Unique original source images: **370**, partitioned **296 train / 37 validation / 37 test**.
- Primary Arm B: **1,280 derived train records** = 944 full/augmented views + 336 crop views. Not 1,280 independent originals.
- Arm A: **944 derived train records**; same 296 independent train sources.
- Independent internal trainings: **24**, configurations **8** across **three seeds 42, 123, 2025**.
- Completed internal held-out evaluations: **6** (Transformer/no-seam, Seam Proposed × three seeds).
- Opposite-edge gap: 0.0899348827 (no seam) versus 0.0533788555 (seam proposed), reduction **40.6472%**.
- Mean pooled FID (lower is better): 254.4739787 (no seam) versus 302.1274312 (seam proposed).
- Mean pooled KID (lower is better): 0.1284848278 (no seam) versus 0.1908464829 (seam proposed).
- Expert/caption semantic review: **not conducted**; avoid claims of cultural authenticity, verified caption semantics, or weaving feasibility.
- Adaptive GAN training paper by Karras is contextual evidence, not proof that the Endek T2I pipeline implements Adaptive Discriminator Augmentation (ADA); this study uses its **own source-aware fixed preprocessing**.
- External comparison: DF-GAN adaptation uses MiniLM384 instead of canonical DAMSM; TiPGAN remains a different *image-conditioned* task and is not yet a converged fair benchmark.

## Source identity & citation exceptions

1. **2024 Endek Stable Diffusion author-order discrepancy:** the published article PDF title page lists **Theresia Hendrawati, Dewa Ayu Putri Wulandari, Ni Luh Wiwik Sri Rahayu Ginantra**; some Garuda/SciSpace indexing lists Ginantra first. The present BibTeX follows the PDF version. Check the authoritative Mendeley/Crossref import before final submission, and preserve the chosen metadata consistently.
2. **Publication year versus DOI:** the relevant IEEE Access batik article is published in **2025** (vol. 13, pp. 2436–2448), even though the DOI contains 2024; BibTeX and APA references both use **2025**.
3. **Peer-reviewed proceedings exceptions:** DF-GAN (CVPR 2022) is the **user-selected second external benchmark**; Karras et al. (NeurIPS 2020) provides foundational data-efficiency evidence. Neither is misrepresented as a journal article. Official Bali government statement is a legitimate nonjournal primary cultural source.
4. **Strict sentence-level repeat restrictions:** the primary and secondary anchor references appear once in their specific paper-characterization passages. Generic literature-free analytical transitions and original project findings are marked as **no external journal-citation applicable**; fake citations are not inserted. This remains a policy exception pending later full-manuscript audit.
5. **APA versus IEEE Access:** APA 7th author–date is a **working-draft preference** for citation manager integration. Official IEEE Access submission uses an IEEE numbered citation system; convert after completing References with Mendeley, preserving keys and source identity.
6. **Novelty integrity:** multilingual conditioning and seam loss existed in the earlier EndekGAN-T2I research. The present evidence supports a controlled preprocessing and fidelity–tileability **evaluation contribution**, not a newly invented neural network module or unconditional superiority claim.
7. **Data rights:** raw/private Endek photographs and private prompt pairs are NOT included in this repository, BibTeX files, or Introduction document.

## Follow-on requirements before submission

A fully converged, fair primary TiPGAN tileability benchmark; completion of the third adapted DF-GAN held-out test; external or broader source-diverse Endek data and learning curves; independent expert assessment of cultural appropriateness and prompt semantics; inspection of the visually similar cross-split pair; a current literature search extending beyond the nine cited sources; and final IEEE Access journal template/reference conversion.

**Files:** manuscript/INTRODUCTION_APA7_ENDEK_T2I.md; bibliography/introduction.bib; bibliography/references_master.bib. All are drafts, not guaranteed final submission-ready.
