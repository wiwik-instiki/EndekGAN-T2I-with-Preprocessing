# IEEE Access: independent dataset sufficiency and strong novelty gates

## Decision
**HOLD / do not submit yet.** Earlier EndekGAN-T2I was rejected; as reported by the research team, insufficient independent Endek data was a central concern. This V2 study cannot claim that 1,280 derived training files fix that problem.

The word *EndekGAN-T2I* remains a historical model/repository identifier for reproducibility, but should **not** be the lead branding of the new IEEE Access manuscript. Proposed working title:

**Source-Aware Preprocessing and Fidelity–Tileability Trade-offs in Text-Guided Balinese Textile Motif Generation**

## Data facts from V2 manifest
- 370 independent source photos; train = 296, validation = 37, held-out test = 37.
- Primary Arm B = 1,280 derived train views (944 full-frame/augmented plus 336 crops) but **296 distinct training sources**.
- Flora 118 / fauna 42 / decorative 90 / geometric 46 independent train sources.
- Test sources per class: flora 15 / fauna 5 / decorative 11 / geometric 6. These small groups make per-class FID/KID highly unstable.
- Exact-source SHA-256 split overlap is zero. A potentially perceptually similar dHash cross-split pair still requires review.
- No preprocessed prompt was independently semantic-verified or expert-verified; crop prompts are inherited.
- No external independent dataset has been added. Do not call this a large, expert-validated or generalizable dataset.

## Mandatory pre-submission gates
1. **Data expansion:** Acquire additional **genuinely distinct, provenance-verified Endek motif photographs** under documented ethical/copyright permissions; cover meaningful variation in looms/regions/motif families/materials, and improve minority-class representation. Do not equate artificial views with independent motifs.
2. **Source governance:** Perform human-reviewed duplicate/near-duplicate and cross-label audits, group all related photos and crops together, and split strictly by origin/source before any augmentation. Freeze a new validation and held-out test protocol. Keep V2 results separate as a historical exploratory baseline.
3. **Learning curves and robustness:** Compare data-subset training curves at increasing independent source sizes, report class-level confidence/variability, at least three independent training seeds, and provenance-distinct external holdout when feasible. The needed dataset size is to be justified empirically, not by an arbitrary IEEE minimum.
4. **Retraining:** After material dataset expansion, re-run the no-seam control, proposed method, adapted DF-GAN and the relevant matched evaluation from scratch on the new splits. Old 370-source scores cannot be relabeled as new-data results.
5. **Comparator validity:** TiPGAN is image-conditioned and has only a 30-step pilot, so cannot yet be placed in the same text-prompt benchmark without an explicitly separate texture protocol. DF-GAN–MiniLM is adapted, not canonical DAMSM DF-GAN.
6. **Novelty distinction:** Existing EndekGAN already featured multilingual text and seam loss. V2 source-aware preprocessing and fidelity–tileability analysis are measurable contributions but **not yet a new architectural mechanism**. If the paper asserts strong model novelty, implement and ablate a demonstrably new module beyond the earlier model and external literature.
7. **Semantic/Endek validation:** Independently audit prompt-image fidelity, class specificity, crop labels and cultural authenticity with appropriately qualified experts; report rater agreement and sampling protocol. Do not claim expert validation prematurely.
8. **Documentation:** Update all research figures, tables, experimental manifests and reproducibility notes only from actually completed new-data runs. No invented results; no leaking original photos/prompts/weights into public GitHub.

## Editorial positioning
This study can remain focused on text-guided generation with source-aware preprocessing and seam/fidelity/diversity measurements. It must not be sold as a large-data solution until independent sample diversity and held-out generalization are demonstrated. IEEE Access acceptance cannot be guaranteed by a title change or any particular dataset size.

## Literature narrative
Primary anchor: He et al., TiPGAN (2025), image-conditioned cloth tileability. Supporting same-task anchor: Tao et al., DF-GAN (2022), text-guided generation without explicit periodic-border objective. The actual model must be supported by valid, modality-aware experiments rather than post-hoc claims.

**Document status:** research readiness gate, not a statement of completed experiments.
