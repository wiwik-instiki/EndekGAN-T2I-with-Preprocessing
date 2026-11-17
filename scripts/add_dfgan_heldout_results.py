# -*- coding: utf-8 -*-
from pathlib import Path
import json,pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
R=Path(__file__).resolve().parents[1];ROOT=R.parent
candidates=[
ROOT/"REFERENCE_BASELINES/KAGGLE_DFGAN_TEST_METRICS/DFGAN_V2_heldout_seed42/EVALUATION_STATUS.json",
ROOT/"FINAL_RESULTS/DFGAN_MiniLM_Seed42/HELDOUT_TEST/DFGAN_V2_heldout_seed42/EVALUATION_STATUS.json"]
src=next((p for p in candidates if p.is_file()),None)
assert src is not None,"Kaggle evaluation JSON not found; no synthetic results allowed"
j=json.loads(src.read_text(encoding="utf-8"));d=j["summary"]
assert j["status"]=="EVALUATION_COMPLETE" and j["generated_images"]==256
assert d["seed"]==42 and d["heldout_real_n"]==37 and d["train_best_epoch"]==120
old=pd.read_csv(R/"results/matched_seed42_summary_exploratory.csv")
assert set(old.model)=={"transformer","seam_proposed"}
out=[]
for x in old.itertuples(index=False):
    out.append(dict(method=x.model,source="internal V2 GAN",seed=42,
     pooled_FID=x.pooled_FID,pooled_KID=x.pooled_KID,
     class_macro_FID=x.class_macro_FID,class_macro_KID=x.class_macro_KID,
     seam_gap=x.unweighted_class_macro_seam_gap,n_real=37,n_fake=256))
out.append(dict(method="DFGAN_MiniLM_adapted",source="CVPR DF-GAN + MiniLM384",
  seed=42,pooled_FID=d["pooled_FID"],pooled_KID=d["pooled_KID"],
  class_macro_FID=d["class_macro_FID"],class_macro_KID=d["class_macro_KID"],
  seam_gap=d["mean_seam_gap"],n_real=37,n_fake=256))
df=pd.DataFrame(out)
df.to_csv(R/"results/DFGAN_EndekGAN_heldout_seed42.csv",index=False)
order=["transformer","DFGAN_MiniLM_adapted","seam_proposed"]
f=df.set_index("method").loc[order]
fig,axes=plt.subplots(1,3,figsize=(13,4))
for ax,metric,title in zip(axes,
  ["pooled_FID","pooled_KID","seam_gap"],
  ["Pooled FID ↓","Pooled KID ↓","Opposite-border gap ↓"]):
    ax.bar(range(3),f[metric])
    ax.set_xticks(range(3),["Transformer","DF-GAN–MiniLM","Seam Proposed"],rotation=22,ha="right")
    ax.set_title(title)
    ax.grid(axis="y",alpha=.15)
fig.suptitle("Endek V2 held-out test, one seed (42), 37 real / 256 synthetic")
fig.tight_layout()
fig.savefig(R/"figures/fig15_dfgan_endekgan_seed42_test.png",dpi=180)
plt.close(fig)
md=["# Actual Kaggle V2 held-out comparison: DF-GAN–MiniLM, seed 42","",
    "Training best checkpoint chosen from validation KID 0.143294 at epoch 120; held-out testing uses 37 independent real images and 256 generated outputs. Results are sample-size-sensitive and exploratory.",
    "",
    "| Method | Pooled FID ↓ | Pooled KID ↓ | Class-macro FID ↓ | Class-macro KID ↓ | Seam gap ↓ |",
    "|---|---:|---:|---:|---:|---:|"]
for x in f.itertuples():
    md.append(f"| {x.Index} | {x.pooled_FID:.3f} | {x.pooled_KID:.6f} | {x.class_macro_FID:.3f} | {x.class_macro_KID:.6f} | {x.seam_gap:.6f} |")
md+=["",
    "![Figure 15](../figures/fig15_dfgan_endekgan_seed42_test.png)",
    "",
    "Figure 15. Source-backed, one-seed held-out comparison. Do NOT claim model superiority on these limited sample counts.",
    "",
    "This DF-GAN baseline is adapted to frozen multilingual MiniLM 384-dimensional embeddings, not the authors' original DAMSM conditioning.",
    "TiPGAN is image-conditioned cloth texture synthesis, not text-to-image, and is therefore not assigned an invented text-prompt compliance score.",
    "The two EndekGAN internal models have three completed seeds; adapted DF-GAN has one. Multi-seed DF-GAN replication and expert Endek evaluation remain pending.",
    "Perceptual quality and semantic fidelity are not established by FID, KID or a simple seam gap.",
    "",
    "Private Kaggle run: https://www.kaggle.com/code/agusperdana/endek-v2-dfgan-minilm-heldout-seed42-private"]
(R/"benchmarks/DFGAN_MiniLM_HELDOUT_SEED42.md").write_text("\n".join(md)+"\n",encoding="utf-8")
read=R/"README.md";text=read.read_text(encoding="utf-8")
if "## DF-GAN–MiniLM held-out evaluation on V2" not in text:
    block="\n## DF-GAN–MiniLM held-out evaluation on V2\n\nOne PRIVATE Kaggle run now has actual held-out results for adapted DF-GAN–MiniLM seed 42 (37 real images, 256 synthetic images). Pooled FID 250.430; KID 0.169826; seam gap 0.117372. Preliminary, single seed. See [validated evaluation report and Figure 15](benchmarks/DFGAN_MiniLM_HELDOUT_SEED42.md) and [CSV](results/DFGAN_EndekGAN_heldout_seed42.csv). TiPGAN's image-conditioned modality remains a separate comparison.\n\n"
    text=text.replace("## 7. Ethics, copyright and claims",block+"## 7. Ethics, copyright and claims")
    read.write_text(text,encoding="utf-8")
print("DONE",src,"metrics",len(df),"figure15",flush=True)
