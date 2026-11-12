# EndekGAN-T2I V2: Preprocessed GAN Training

Main experiment B: 1280 training, 37 validation, 37 testing FILES, including 336 heuristic crop variants.
Control A: 944 training, 37 validation, 37 testing files. Held-out sources identical.
The image source/metadata files are private and not provided in this public repository.
These are processed-file counts, not new independent Endek motif counts.
Cropping inherited global prompts and its per-crop semantic alignment has NOT been verified.

Open notebooks/05_EndekGAN_T2I_Train_Preprocessed_Crop_V2.ipynb in VS Code,
select Python 3.11 (EndekGAN-T2I V2 | D:), and execute the preflight cells.
The model runner genuinely trains EndekGAN from random weights.
To actually train, install CUDA dependencies into the D-only kernel with the setup CMD,
verify at least 6400 MiB free RTX VRAM, then change TRAIN_NOW=False to True in notebook section 7.
Experiment settings use 512 balanced sampled items per epoch, batch size 4 and 128
optimizer update pairs per epoch for comparable training A and B.
Validation KID selects checkpoints, held-out test is used only after training.

Notebook preflight execution has succeeded. CUDA PyTorch is currently absent from the D-only kernel
and free RTX VRAM is approximately 2258 MiB, so V2 full GAN training has NOT started.
The old GAN weights are not reused. Expert judgment is reserved for AI-generated images after training.
Adapted DF-GAN-MiniLM is a separate architecture and its dedicated runner is not part of this code.
