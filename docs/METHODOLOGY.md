# Dataset preprocessing — method and scope

## Inputs and derived artifacts

- Independent Endek source images: 370, 4 classes, 512 × 512 pixels.
- One source-level split: train 296 / validation 37 / testing 37.
- Resize whole source images with Lanczos to 256 × 256, without cropping or distortion.
- Photometric augmentation is restricted to training; additional variants balance training file counts at 236 per class.
- Total V1 processed files: train 944 / validation 37 / test 37 = 1,018.
- Every derived file maps to its original source SHA-256 and inherits the original text prompt and class label.
- Final project data remains local; the public repository ships source-code documentation and aggregate counts only.

## Cropping comparison

The notebook compares (i) original 512 × 512, (ii) resized 256 × 256,
(iii) centered crop 384 × 384 resized to 256 × 256, (iv) edge/texture-heuristic
pilot crop 384 × 384 resized to 256 × 256, and (v) one 256 × 256
photometrically augmented variant. Pilot crop coordinates come from a local archive.
The two cropping demonstrations **are not included** in V1 training files and
were not validated as ornament-aware cropping.

## How Text-to-Image conditioning is preserved

- Existing original prompt, technique, layout, color and class metadata are inherited by all augmented images.
- The images remain paired by source identifier; no synthesized captions are invented for crops.
- For color-adjusted variants, literal color descriptions may still require visual checking.
- Experts evaluate AI-generated results after generation, not preprocessing.
- Do not claim that the number of independent motifs increased due to photometric augmentation.

## Dataset leakage and comparability

Source splitting takes place before augmenting. A close floral pair was placed
within the same train split; one different floral source was swapped to the test
split to preserve counts. Thus the split is not exactly identical to the prior
EndekGAN-T2I manuscript's source membership. All new experimental comparisons
must use the same source partition and the same evaluation protocol.

## Training status

This repository documents dataset preprocessing. It does not include newly trained
EndekGAN-T2I V2 GAN results. For generator/discriminator architecture and earlier
FID/KID and seam-loss experiments, refer to the [original research repository](https://github.com/wiwik-instiki/EndekGAN-T2I).
