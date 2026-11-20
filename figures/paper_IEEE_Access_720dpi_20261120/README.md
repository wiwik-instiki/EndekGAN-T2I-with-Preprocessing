# IEEE Access Endek T2I: Figure package (720 dpi)

All 13 figures correspond to the November 20, 2026 manuscript and contain no private model checkpoints, raw datasets or source-linked prompt metadata.

**Image quality disclosure.** Figures 1–11 reuse existing published source image panels; most have been Lanczos-upscaled to 720 pixels per paper inch. This does not create additional optical detail. Figures 12–13 are native 720dpi matplotlib re-renders of the paper metric plots. They reproduce numeric evidence but not necessarily byte-identical styling of the manuscript embedded raster figures.

**Rights and scope.** Figure 1 shows already-public original-source textile samples; publication rights still need author confirmation. Figure 11 contains image-conditioned training-only results, not held-out text-to-image tests. Neither figure set is artisan-certified.

## Figure index

- [Fig. 1: Representative original-source-derived motifs](Fig_01_Endek_T2I_720dpi.png)
- [Fig. 2: Source-aware preprocessing arms A/B](Fig_02_Endek_T2I_720dpi.png)
- [Fig. 3: Research framework for RQ1 and RQ2](Fig_03_Endek_T2I_720dpi.png)
- [Fig. 4: Conditional GAN architecture](Fig_04_Endek_T2I_720dpi.png)
- [Fig. 5: Training seam penalty and held-out edge gap](Fig_05_Endek_T2I_720dpi.png)
- [Fig. 6: Three-seed checkpoint-validation KID](Fig_06_Endek_T2I_720dpi.png)
- [Fig. 7: Three-seed held-out FID, KID, edge gap and LPIPS](Fig_07_Endek_T2I_720dpi.png)
- [Fig. 8: Per-seed held-out FID, KID, edge gap and LPIPS](Fig_08_Endek_T2I_720dpi.png)
- [Fig. 9: Prompt-matched generated samples (seed 42)](Fig_09_Endek_T2I_720dpi.png)
- [Fig. 10: Two-by-two tiling of Figure 9](Fig_10_Endek_T2I_720dpi.png)
- [Fig. 11: TiPGAN training-only exemplars](Fig_11_Endek_T2I_720dpi.png)
- [Fig. 12: Class-wise fidelity/continuity trade-off](Fig_12_Endek_T2I_720dpi.png)
- [Fig. 13: Per-class LPIPS diversity diagnostics](Fig_13_Endek_T2I_720dpi.png)

See [FIGURE_MANIFEST.csv](FIGURE_MANIFEST.csv) for dimensions, sources and SHA-256 checksums.

**Figure 12 axis parity:** Horizontal FID/KID (lower better), vertical opposing-border gap (lower better), matching the manuscript. Arrows connect the no-seam circle with the proposed square. This is the class-wise held-out plot, not the supplementary 1/4/8-pixel check.
