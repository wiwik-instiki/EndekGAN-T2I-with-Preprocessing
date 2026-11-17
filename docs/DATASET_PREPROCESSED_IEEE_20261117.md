# IEEE Access — 2.2 Dataset (based only on completed preprocessing)

The experiments utilized the **preprocessed Balinese Endek motif dataset** containing four categorical labels: Flora, Fauna, Decorative, and Geometric. The primary preprocessing arm (Arm B) comprised **1,354 records**, divided into **1,280 training images, 37 validation images, and 37 held-out test images**. Images were resized to **256 × 256 RGB pixels**. The training records included 944 full-frame preprocessed/augmented images and 336 additional crop-derived views; the latter were obtained exclusively from existing training sources. The 1,280 training records therefore represented **296 independent image sources**, not 1,280 distinct acquired Endek photographs.

**Table 1. Preprocessed dataset (source-group-aware partitioning).**

| Motif class | Training B | Validation | Test |
|---|---:|---:|---:|
| Flora | 320 | 14 | 15 |
| Fauna | 320 | 6 | 5 |
| Decorative | 320 | 11 | 11 |
| Geometric | 320 | 6 | 6 |
| **Total** | **1,280** | **37** | **37** |

Source identifiers derived from SHA-256 were used to audit separation across partitions. The training, validation and held-out sets contained 296, 37 and 37 independent sources, respectively, with **no identical source SHA-256 identifier appearing in more than one partition**. Validation and test images were retained as full-image samples without the training-specific augmentation or cropping. The alternate Arm A contained 944 training records from the **same 296 training sources** and was used solely to assess the effect of crop augmentation.

Each record had a structured Indonesian prompt, motif category and design attributes supplied by the existing metadata process; the prompts were used as the linguistic conditioning signal. The audited source manifest marked **none of these prompt–image associations as independently semantically verified**, and no record was marked as expert validated. In particular, the 336 cropped training views inherited prompts from their parent images. Accordingly, motif-level semantic correctness of these crops, cultural authenticity, and weaving feasibility remain outside the verified labels of this dataset. No additional externally proposed image corpus was included in the reported experiments.

**Dataset reporting rules.** This manuscript must describe Arm B as the experimental dataset and Arm A as the preprocessing ablation, distinguish derived file counts from independent source counts, and never imply that the original images or crop prompts were expert-certified. Do not describe unintegrated Mendeley/Roboflow collections as though they were used in training.
