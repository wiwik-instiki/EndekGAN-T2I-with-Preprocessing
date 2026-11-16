# -*- coding: utf-8 -*-
"""Update public evidence README + safe comparison notebook after matched test completes."""
from pathlib import Path
import json,datetime,pandas as pd
ROOT=Path(__file__).resolve().parents[1]
RES=ROOT/"results";DOCS=ROOT/"docs";TABLES=ROOT/"tables";FIG=ROOT/"figures"
assert (RES/"REFERENCE_BASELINE_PREFLIGHT.json").is_file()
assert (RES/"matched_seed42_summary_exploratory.csv").is_file()
for f in ("fig09_matched_actual_generated_samples_seed42.png",
          "fig10_matched_2x2_tile_visualization_seed42.png",
          "fig11_matched_seed42_test_KID.png"):
    assert (FIG/f).is_file()
pair=pd.read_csv(RES/"matched_seed42_summary_exploratory.csv")
assert pair.model.tolist()==["transformer","seam_proposed"]
assert (pair.n_test_real==37).all() and (pair.n_generated==256).all()
s=pair.set_index("model")
x=RES/"EVIDENCE_STATUS.json"
st=json.loads(x.read_text(encoding="utf-8"))
assert st["training_runs"]==24
st["evaluated_test_models"]=["B/transformer/seed42","B/seam_proposed/seed42"]
st["evaluated_test_run_count"]=2
st["pending_test_models"]=22
st["actual_matched_test_comparison"]=True
st["TiPGAN_V2_test"]="NOT_RUN"
st["DFGAN_V2_test"]="NOT_RUN"
st["paper_readiness"]="INCOMPLETE_EXTERNAL_BASELINES_AND_MULTI_SEED_TESTS"
x.write_text(json.dumps(st,indent=2,ensure_ascii=False),encoding="utf-8")
doc=DOCS/"TRAINING_V2_PUBLIC_AUDIT.json"
v=json.loads(doc.read_text(encoding="utf-8"))
v["V2_heldout_test_evaluated"]=["B/transformer/seed42","B/seam_proposed/seed42"]
v["other_22_V2_tests"]="NOT_COMPLETED"
v.pop("other_23_V2_tests",None)
v["actual_two_way_matched_figures"]=3
doc.write_text(json.dumps(v,indent=2,ensure_ascii=False),encoding="utf-8")

notes=(ROOT/"benchmarks")
notes.mkdir(exist_ok=True)
report=f"""# Reference-anchored three-method benchmark: what is and is not complete

## Fixed research direction and literature
- **Main reference:** TiPGAN (He et al., *Computer-Aided Design*, 2025). Official code: https://github.com/VickyHEHonghong/TiPGAN. This is **image-conditioned tileable cloth texture synthesis**; it cannot be treated as a text-to-image architecture.
- **One additional external comparator:** DF-GAN (Tao et al., *CVPR*, 2022). Official code: https://github.com/tobran/DF-GAN. This is **text-to-image** using DAMSM text embeddings.
- **Proposed:** EndekGAN-T2I V2, frozen multilingual text conditioning plus seam-aware regularization, evaluated on a controlled source-aware data split. The old EndekGAN implementation already had seam and text features, so this is not claimed as a wholly new invented architecture.

## Verified Endek data and benchmark arms
- Original independent sources: **370** (train **296**, validation **37**, test **37**).
- Arm A: **944 derived training views**, originating from 296 train source images.
- Arm B: **1280 derived training views**, originating from 296 train source images, of which **336** are crops with inherited prompts **not semantically verified**.
- All 24 V2 model-seed training jobs are complete. Test-set evaluation is now complete for just **two Arm B seed-42** models: Transformer and Seam Proposed.

## Matched EndekGAN V2 test: two genuine trained models, one seed
Evaluation uses the same held-out prompt/source references, image resolution 256x256, 64 generated images per class, and generation RNG seed 90731. Prompt/source/index alignment was verified before making paired visualization panels.

| Metric (lower is better) | B Transformer seed 42 | B Seam Proposed seed 42 |
|---|---:|---:|
| Pooled FID (37 real / 256 generated) | {s.loc['transformer','pooled_FID']:.3f} | {s.loc['seam_proposed','pooled_FID']:.3f} |
| Pooled KID | {s.loc['transformer','pooled_KID']:.6f} | {s.loc['seam_proposed','pooled_KID']:.6f} |
| Class-macro FID | {s.loc['transformer','class_macro_FID']:.3f} | {s.loc['seam_proposed','class_macro_FID']:.3f} |
| Class-macro KID | {s.loc['transformer','class_macro_KID']:.6f} | {s.loc['seam_proposed','class_macro_KID']:.6f} |
| Unweighted class-macro opposite-border gap | {s.loc['transformer','unweighted_class_macro_seam_gap']:.6f} | {s.loc['seam_proposed','unweighted_class_macro_seam_gap']:.6f} |

**Finding:** the seam penalty lowers *measured opposing-border gap* but the Transformer without seam is **better on FID/KID** under this one-seed exploratory test. These metrics do not directly measure expert-rated motif recognizability or text adherence. The test set contains only 37 independent images total, severely limiting FID/KID reliability.

### Actual GAN-generated-only comparison (no raw source images published)
![Same prompts and latent seed](../figures/fig09_matched_actual_generated_samples_seed42.png)

**Figure 9.** Generated motif examples for Transformer versus Seam Proposed; for each row the models receive the exact same held-out prompt, class label, image index and latent RNG seed. Examples are fixed indices 0 and 1, not selected for best looks.

![Repeated textile tiles](../figures/fig10_matched_2x2_tile_visualization_seed42.png)

**Figure 10.** 2 x 2 periodic repetition of *actual* generated images at fixed example index 0 across four motif classes. Tiling measurements alone cannot validate motif semantics, culture or actual weaving feasibility.

![Paired per-class test KID](../figures/fig11_matched_seed42_test_KID.png)

**Figure 11.** Exploratory test KID per class, seed 42 only. Class-specific real reference count is only 5–15.

## External benchmark execution status and modality fairness

| Method | Original modality | Local V2 preparation | On-V2 training | Independent V2 scores |
|---|---|---|---|---|
| TiPGAN | Input image / patch texture | Official git cloned; 4 train-only class-balanced source exemplars selected privately | **NOT RUN** | **NOT AVAILABLE** |
| DF-GAN | Text via DAMSM encoder | Official git cloned; train-prompt pairs collected privately | **NOT RUN** | **NOT AVAILABLE** |
| EndekGAN V2 | Text + class / generated motif | 24 pretrained local V2 checkpoints verified | **COMPLETED** | **2 of 24 final tests** |

**TiPGAN comparability:** its normal research objective is to synthesize a tileable result from a supplied image. Use train-only texture exemplars and compare tileability, edge discontinuity and faithful texture preservation; do **not** score text-prompt adherence on a model that consumes no prompt. A per-texture trained TiPGAN is not a text-to-image baseline under the same input modality. Do not use held-out test images as TiPGAN's input texture while claiming independent test fidelity.

**DF-GAN comparability:** its official code uses a pretrained DAMSM encoder and claims at least 12 GB NVIDIA VRAM; local RTX 3070 Ti offers 8 GB. No Endek DAMSM weights were present in the official clone. Any replacement with MiniLM or reduced batch-size/architecture is **adapted DF-GAN** and must be named/disclosed as such. The earlier repo's DF-GAN results belong to a *different* data protocol and cannot be copied into the V2 three-way table.

## Verified blockers and next steps
1. TiPGAN's official checkout has \`train.py\` rather than README's mentioned \`train_tipgan.py\`; VGG weights required, not present. Its source checkout does not contain an explicit LICENSE file: verify rights before redistribution.
2. DF-GAN official training requires specific caption metadata, DAMSM checkpoints and a supported 12 GB+ GPU. Never silently replace with a non-equivalent model and call it a canonical reproduction.
3. For EndekGAN, finish matching **seeds 123 and 2025** before scientific multi-seed claims. Baseline models must use source-group-matched inputs, declared conditioning modalities and fixed output metrics.
4. Conduct counterfactual text prompts and blinded Endek expert review before claiming preservation of cultural patterns.

## Official-code environment blockers verified
- TiPGAN could not import in the current V2 Python environment because cv2 (OpenCV) is absent; VGG weights were also not present. No TiPGAN-Endek training has been performed.
- DF-GAN requires DAMSM text/image encoders and officially documents at least 12 GB NVIDIA VRAM; the local RTX 3070 Ti has 8 GB. Using a different text encoder or reduced model is an explicitly adapted benchmark, not exact replication.
- Only private, training-only reference-image/caption manifests have been prepared, with no dataset photographs published.

## Research status
**This is a verified two-way V2 test + external baseline input-preparation protocol, NOT a completed TiPGAN-vs-DF-GAN-vs-EndekGAN performance ranking.**
"""
(notes/"README.md").write_text(report,encoding="utf-8")
readme=ROOT/"README.md"
r=readme.read_text(encoding="utf-8")
# Preserve all existing research context, visibly superseding the earlier one-model status.
r=r.replace("**ONE evaluated V2 model only:** Arm B / seam_proposed / seed 42",
            "**TWO evaluated V2 models at seed 42:** Arm B / transformer and Arm B / seam_proposed")
r=r.replace("**ONE evaluated V2 model only:** Arm B / seam_proposed / seed 42",
            "**TWO evaluated V2 models at seed 42:** Arm B / transformer and Arm B / seam_proposed")
r=r.replace("The only completed V2 held-out evaluation (B seam seed 42)",
            "The first completed V2 held-out evaluation (B seam seed 42)")
newblock="""## New: official-paper benchmark protocol and actual two-way generated image comparison

The official TiPGAN and DF-GAN repositories were inspected and cloned separately. They are **not** yet trained on the V2 Endek source split. The official DF-GAN requires a DAMSM encoder and documents a minimum 12 GB GPU; TiPGAN is a single-image reference texture method, not text conditioned.

**New actual evidence:** Arm B Transformer vs Arm B Seam Proposed, seed 42, on the **same held-out prompts, random latent seed and 64 images/class**. The Transformer achieved pooled FID **247.82**, pooled KID **0.11162**, with opposite-border gap **0.07952**. Seam Proposed achieved pooled FID **302.47**, pooled KID **0.19289**, and lower opposite-border gap **0.05246**. These are **exploratory** one-seed scores with only 37 real references; no FID/KID or texture-quality superiority claim is made.

Read [benchmark report with actual paired images and 2x2 tiling](benchmarks/README.md), [official code/data preflight](results/REFERENCE_BASELINE_PREFLIGHT.json), [comparison matrix](tables/REFERENCE_BENCHMARK_MATRIX.csv), and [safe VS Code notebook 06](notebooks/06_TiPGAN_DFGAN_EndekGAN_Comparison_V2.ipynb).

"""
mark="## 7. Ethics, copyright and claims"
assert mark in r
if "## New: official-paper benchmark protocol and actual two-way generated image comparison" not in r:
    r=r.replace(mark,newblock+mark)
readme.write_text(r,encoding="utf-8")

# One safe observational Jupyter notebook: no model fit, no implicit evaluation.
md1="# Experiment 06 — TiPGAN / DF-GAN / EndekGAN-T2I V2\n\nThis notebook reads only verified local-public evidence and does not train models. TiPGAN is an image-reference texture method; DF-GAN and EndekGAN are text-conditioned GANs. Do not compare incompatible text-conditioning modalities under one fictitious score."
code1="""from pathlib import Path
import pandas as pd, json
from IPython.display import display, Markdown, Image
here=Path.cwd()
REPO=next((p for p in (here,here.parent,here.parent.parent) if (p/'results/REFERENCE_BASELINE_PREFLIGHT.json').exists()),None)
assert REPO is not None, 'Open notebook from the public release repository root'
preflight=json.loads((REPO/'results/REFERENCE_BASELINE_PREFLIGHT.json').read_text(encoding='utf-8'))
display(Markdown('### Dataset and benchmark preflight'))
display(pd.DataFrame([{'method':'TiPGAN','train_on_V2':preflight['TiPGAN']['V2_dataset_training_status'],
    'input':'image-reference'}, {'method':'DF-GAN','train_on_V2':preflight['DF_GAN']['V2_dataset_training_status'],
    'input':'text/DAMSM'},{'method':'EndekGAN V2','train_on_V2':'24 completed runs','input':'text + class'}]))
display(pd.read_csv(REPO/'tables/REFERENCE_BENCHMARK_MATRIX.csv'))
"""
code2="""from IPython.display import Image, display, Markdown
display(Markdown('### Two real V2 models: seed 42, identical test protocol'))
data=pd.read_csv(REPO/'results/matched_seed42_summary_exploratory.csv')
display(data)
for f in ['fig09_matched_actual_generated_samples_seed42.png',
          'fig10_matched_2x2_tile_visualization_seed42.png',
          'fig11_matched_seed42_test_KID.png']:
    display(Image(filename=str(REPO/'figures'/f)))
display(Markdown('**Scientific caveat:** test real n=37. TiPGAN and DF-GAN test metrics are missing, not zero.'))
"""
code3="""# Safe only: no training or test evaluation is launched by running this notebook.
RUN_EXTERNAL_BASELINES=False
if RUN_EXTERNAL_BASELINES:
    raise RuntimeError('External TiPGAN and DF-GAN training needs separately reviewed adapters, dataset rights and adequate CUDA memory.')
print('Notebook review complete: official-baseline trainings were NOT executed.')
"""
def mcell(src):
    return dict(cell_type="markdown",metadata={},source=src.splitlines(keepends=True))
def ccell(src):
    return dict(cell_type="code",execution_count=None,metadata={},outputs=[],source=src.splitlines(keepends=True))
notebook=dict(cells=[mcell(md1),ccell(code1),ccell(code2),mcell(
    "## External reference experiment status\n\nTiPGAN official single-image training needs VGG weights; not run. "
    "DF-GAN original implementation needs DAMSM and >=12GB per original README; not run. "
    "Use the private preflight scripts and explicitly adapted protocol before a GPU run."),ccell(code3)],
    metadata=dict(kernelspec=dict(display_name="Python 3.11 (EndekGAN V2 D-drive)",
                                  language="python",name="endekgan-t2i-v2-d"),
                  language_info=dict(name="python",version="3.11")),nbformat=4,nbformat_minor=5)
nb=ROOT/"notebooks"/"06_TiPGAN_DFGAN_EndekGAN_Comparison_V2.ipynb"
nb.write_text(json.dumps(notebook,indent=1,ensure_ascii=False)+"\n",encoding="utf-8")

# README link integrity check; figures included.
import re
links=re.findall(r'\]\(([^)]+)\)',r)
missing=[link for link in links if not link.startswith(("http://","https://","#")) and not (ROOT/link).exists()]
assert not missing, f"README relative links missing: {missing}"
print("RELEASE_UPDATED",len(links),"links checked, image figure 9-11 and notebook 06 complete")
