# -*- coding: utf-8 -*-
"""Only synchronize verified existing six EndekGAN held-out tests into GitHub-ready report."""
from pathlib import Path
import json, pandas as pd
R=Path(__file__).resolve().parents[1]
source=R/"results"/"heldout_B_3seeds_mean_sd.csv"
assert source.is_file()
counts=pd.read_csv(R/"results"/"heldout_B_transformer_vs_seam_3seeds_per_seed.csv")
assert len(counts)==6 and set(counts.model)=={"transformer","seam_proposed"}
assert counts.groupby("model").seed.nunique().to_dict()=={"transformer":3,"seam_proposed":3}
for f in ("fig12_heldout_B_test_three_seeds_tradeoff.png",
          "fig13_heldout_B_test_three_seed_mean_sd.png",
          "fig14_actual_B_generated_motifs_3seed_comparison.png"):
    assert (R/"figures"/f).is_file()

status=R/"results"/"EVIDENCE_STATUS.json"
obj=json.loads(status.read_text(encoding="utf-8"))
obj["evaluated_test_models"]=[f"B/{row.model}/seed{row.seed}" for row in counts.itertuples(index=False)]
obj["evaluated_test_run_count"]=6
obj["pending_test_models"]=18
obj["B_transformer_and_seam_three_seed_test_complete"]=True
obj["TiPGAN_V2_test"]="NOT_RUN"
obj["DFGAN_V2_test"]="NOT_RUN"
obj["caution"]="All FID/KID test scores exploratory: just 37 independent reference images."
status.write_text(json.dumps(obj,indent=2,ensure_ascii=False),encoding="utf-8")
j=R/"docs"/"TRAINING_V2_PUBLIC_AUDIT.json"
audit=json.loads(j.read_text(encoding="utf-8"))
audit["V2_heldout_test_evaluated"]=obj["evaluated_test_models"]
audit["other_18_V2_tests"]="NOT_COMPLETED"
audit.pop("other_22_V2_tests",None)
audit.pop("other_23_V2_tests",None)
audit["last_3seed_benchmark"]="COMPLETE_INTERNAL_TWO_METHODS"
j.write_text(json.dumps(audit,indent=2,ensure_ascii=False),encoding="utf-8")

r=R/"README.md"
text=r.read_text(encoding="utf-8")
text=text.replace("**TWO evaluated V2 models at seed 42:** Arm B / transformer and Arm B / seam_proposed",
                  "**SIX evaluated V2 models/seeds:** Arm B Transformer and Seam Proposed, each at seeds 42/123/2025")
text=text.replace("**TWO evaluated V2 models at seed 42:** Arm B / transformer and Arm B / seam_proposed",
                  "**SIX evaluated V2 models/seeds:** Arm B Transformer and Seam Proposed, each at seeds 42/123/2025")
new="""
## Completed matched six-test evaluation, 3 training seeds (update 17 November 2026)

**Completed:** Arm B Transformer and Seam Proposed each at training seeds **42, 123, 2025**, all with source-identical 296/37/37 train/validation/test split, 64 generated held-out samples per class, and fixed evaluation latent seed. Images and private prompt records remain local.

| Held-out test metric | Transformer mean ± SD | Seam Proposed mean ± SD |
|---|---:|---:|
| Pooled FID ↓ | 254.474 ± 40.967 | 302.127 ± 5.459 |
| Pooled KID ↓ | 0.128485 ± 0.054584 | 0.190846 ± 0.004053 |
| Class-macro FID ↓ | 316.145 ± 41.270 | 358.653 ± 8.125 |
| Opposite-border seam gap ↓ | 0.089935 ± 0.014491 | **0.053379 ± 0.001198** |
| LPIPS diversity ↑ | 0.256295 ± 0.216808 | 0.033152 ± 0.041343 |

**Observed trade-off:** opposite-border gap reduced approximately **40.6%** in the three-seed mean, but worsened distributional fidelity and sharply reduced LPIPS diversity. Seed 2025 slightly favored Seam Proposed in pooled FID/KID. FID/KID are *exploratory* with only 37 real test images. No human-rated Endek authenticity is implied.

[Complete scientific three-seed report](benchmarks/THREE_SEED_TEST_REPORT.md) ·
[per-seed CSV](results/heldout_B_transformer_vs_seam_3seeds_per_seed.csv) ·
[mean ± SD](results/heldout_B_3seeds_mean_sd.csv) ·
[Figure 12](figures/fig12_heldout_B_test_three_seeds_tradeoff.png) ·
[Figure 13](figures/fig13_heldout_B_test_three_seed_mean_sd.png) ·
[Figure 14](figures/fig14_actual_B_generated_motifs_3seed_comparison.png)

**External main-reference TiPGAN (image-conditioned) and additional DF-GAN (text-conditioned) have NOT completed a V2 benchmark.** Neither can be represented by values borrowed from the earlier EndekGAN experiments.

"""
if "## Completed matched six-test evaluation" not in text:
    mark="## 7. Ethics, copyright and claims"
    assert mark in text
    text=text.replace(mark,new+mark)
r.write_text(text,encoding="utf-8")
p=R/"benchmarks"/"README.md"
s=p.read_text(encoding="utf-8")
if "THREE_SEED_TEST_REPORT.md" not in s:
    s += "\n## New three-seed outcome\n\nThe internal matched B-arm comparison across seeds 42/123/2025 is **complete**. See [the verified six-test results with figures 12–14](THREE_SEED_TEST_REPORT.md). The original seed-42 table remains historical detail and is superseded by the three-seed summary for scientific discussion. External TiPGAN and DF-GAN still lack V2 results.\n"
p.write_text(s,encoding="utf-8")
t=R/"tables"/"TABLES_AND_CAPTIONS.md"
s=t.read_text(encoding="utf-8")
if "Table 6. Matched three-seed" not in s:
    s+="\n## Table 6. Matched three-seed held-out test — full pre-manuscript comparison\n\n"
    s+="See [three-seed results](../benchmarks/THREE_SEED_TEST_REPORT.md). Means and sample SD computed from three completed independent training seeds, not 3 independent test datasets. Results are exploratory because the actual held-out test only includes 37 distinct reference images.\n\n"
    s+="**Figure 12.** Per-seed fidelity and seam gap trade-offs for matched B-arm controls.\n\n"
    s+="**Figure 13.** Mean ± sample SD of three independent training-seed outcomes; no significance claim.\n\n"
    s+="**Figure 14.** Generated images for two models across four classes and three training seeds, fixed sample index, not hand-picked.\n"
t.write_text(s,encoding="utf-8")
nb=R/"notebooks"/"06_TiPGAN_DFGAN_EndekGAN_Comparison_V2.ipynb"
data=json.loads(nb.read_text(encoding="utf-8"))
src="# Three-seed, truly held-out internal V2 comparison\n\nTest results from three trained seeds per model; no external baseline results are invented."
if not any("three-seed, truly held-out" in "".join(c.get("source",[])) for c in data["cells"]):
    data["cells"] += [
      dict(cell_type="markdown",metadata={},source=src.splitlines(keepends=True)),
      dict(cell_type="code",metadata={},execution_count=None,outputs=[],source=(
      "import pandas as pd\nfrom IPython.display import display, Image as NotebookImage\n"
      "summary=pd.read_csv(REPO/'results/heldout_B_3seeds_mean_sd.csv')\n"
      "per_seed=pd.read_csv(REPO/'results/heldout_B_transformer_vs_seam_3seeds_per_seed.csv')\n"
      "display(summary);display(per_seed)\n"
      "for f in ('fig12_heldout_B_test_three_seeds_tradeoff.png','fig13_heldout_B_test_three_seed_mean_sd.png','fig14_actual_B_generated_motifs_3seed_comparison.png'):\n"
      "    display(NotebookImage(filename=str(REPO/'figures'/f)))\n"
      "print('External TiPGAN/DFGAN V2 evaluation is NOT complete.')\n").splitlines(keepends=True))
    ]
nb.write_text(json.dumps(data,indent=1,ensure_ascii=False)+"\n",encoding="utf-8")
print("UPDATED_6_TEST_RELEASE",len(data["cells"]),"NOTEBOOK_CELLS")
