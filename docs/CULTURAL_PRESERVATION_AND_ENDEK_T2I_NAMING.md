# IEEE Access cultural motivation, scarce-data positioning, and method identity

## Fixed publication terminology

- **Method name:** Endek T2I. No V2, version numbers, or EndekGAN-T2I branding in the proposed scientific method name.
- **Working article title:** Source-Aware Preprocessing and Seam-Regularized Text-to-Image Synthesis of Balinese Endek Motifs.
- **Historical original paper/repository:** EndekGAN-T2I (cited as prior work rather than misrepresented as newly invented).
- **Internal experiment labels:** V2 in directory or legacy script names is permitted only for reproducibility; do not use V2 as a method identifier or title suffix.

## Culturally grounded educational significance

**The cultural-preservation and artisan-regeneration argument belongs in the Introduction and Discussion, not in the paper title.** The proposed method name remains **Endek T2I**, without a V2 suffix. Source-aware preprocessing and seam regularization are implemented experimental components, while strong architectural novelty beyond prior EndekGAN remains unproven.

Balinese Endek weaving is a traditional Indonesian cultural expression that integrates visual motifs, specialized artisanship and local identity. The Government of Bali actively supports its protection, including the 2021 Endek usage policy and communal intellectual property registry EBT.12.2020.0000085 (recorded 22 December 2020). In an official government release dated 30 May 2026, the Balinese Provincial Government highlighted a serious decline in young people entering the weaving profession, placing the intergenerational transmission of craft knowledge under pressure. The defensible claim is **threatened continuity of craftsmanship and artisan regeneration**, not a proven assertion that the entire Endek weaving tradition is already nearly extinct.

Primary sources:
- Government of Bali (30 May 2026): https://www.baliprov.go.id/web/putri-koster-krisis-penenun-muda-ancam-masa-depan-endek-bali/
- Government of Bali (2021): https://www.baliprov.go.id/web/gubernur-bali-himbau-penggunaan-pakaian-berbahan-kain-tenun-endek-bali-kain-tenun-tradisional-bali
- Indonesia communal IP office: https://www.dgip.go.id/index.php/artikel/detail-artikel-berita/kain-tenun-endek-bali-resmi-terlindungi?kategori=ki-komunal
- Evidence of artisan regeneration difficulties: https://jurnal.uny.ac.id/index.php/economia/article/view/36218

## Educational value of AI without displacing craft expertise

Endek T2I is proposed as a **digital design-exploration and educational support tool**: it can document motif categories, translate structured descriptions into exploratory images, assist younger learners in visualizing motif attributes, and make the constraints of repeatable textile design more accessible. It is NOT evidence that generated outputs are culturally authentic, physically weaveable, or replacements for actual weavers. Qualified Endek experts must validate generated motifs and contextual explanations before strong heritage-conservation claims.

## Correct research scarcity claim

The relevant technical subfield is niche, but it is false to say no Endek text-to-image research exists. Identified studies include:
- Ginantra, N. L. W. S. R., Hendrawati, T., and Wulandari, D. A. P. (2024), Penerapan Metode Stable Diffusion Dengan Fine Tuning Untuk Pola Endek Bali, TEMATIK 11(2). https://garuda.kemdiktisaintek.go.id/documents/detail/4714265
- Suparjana, I K. G. J., and Purnawan, I K. A. (2026), Fine-Tuning Text-to-Image Diffusion Models with LoRA for Generative Synthesis of Balinese Endek Textile Patterns, JANAPATI. https://ejournal.undiksha.ac.id/index.php/janapati/article/view/111817
- Heritage design practice (2024), Crafting the Future: A Sustainable Practice of Endek Woven Cloth in Jinengdalem Village, Bali, Journal of Visual Art and Design. https://journal-itb.org/jvad/volumes/2024/16/issues/2/crafting-the-future-a-sustainable-practice-of-endek-woven-cloth-in-jinengdalem-village-buleleng-district-bali-province-i.html

Use: **“Limited domain-specific studies have addressed computational generation of Balinese Endek textile motifs, leaving important questions about source-independent evaluation, conditional motif fidelity, and periodic boundary continuity.”** This is scoped, qualified, and should be reviewed with a systematic recent-literature search before submission.

## Dataset scarcity is the setting, not a proven contribution

Current Arm B is 1,280 derived train views **from 296 independent training images**, plus 37 validation and 37 test images (370 independent source images overall). Training-only augmentation and 336 crops increase exposure but cannot be presented as an expanded independent dataset. No independent semantic expert validation has yet been completed. A 37-image real test reference set yields exploratory FID/KID; do not infer statistical generalization.

To maximize the available dataset rigorously: report source-level deduplication and splitting, balanced augmentation, three-seed training, matched ablations and learning curves by independent source count, source-aware uncertainty estimates, per-class failure analysis and expert motif evaluation. Rerun comparisons on additional independently acquired data if and when available; never relabel existing derived records as new originals.

## Relation to novelty and external references

Primary textile tileability reference: TiPGAN, He et al. (2025), image-conditioned. Supporting same-modality text-to-image model: DF-GAN, Tao et al. (2022), adapted to frozen MiniLM384 for Endek T2I comparisons. The earlier EndekGAN-T2I architecture already included multilingual text conditioning and seam loss. This study can establish a controlled fidelity–tileability and preprocessing trade-off in a culturally important limited-data domain; do not claim a newly invented module without architectural changes and ablation evidence.

**Submission state:** not yet submission-ready; third DF-GAN held-out evaluation, TiPGAN converged benchmark, independent prompt fidelity and Endek expert review remain open as of the referenced research snapshot.
