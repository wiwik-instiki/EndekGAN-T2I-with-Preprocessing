# -*- coding: utf-8 -*-
"""Produce truthful, seed/prompt/index-matched V2 images and metrics, no external fakes."""
from pathlib import Path
import json, hashlib
import numpy as np,pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image

REPO=Path(__file__).resolve().parents[1]
ROOT=REPO.parent
BASE=ROOT/"runs_preprocessed"/"full_plus_crop_ablation_B"
FIG=REPO/"figures";RES=REPO/"results"
FIG.mkdir(exist_ok=True);RES.mkdir(exist_ok=True)
test={}
for model in ("transformer","seam_proposed"):
    run=BASE/f"{model}_seed42"/"test_evaluation"
    assert (run/"evaluation_status.json").is_file(),f"{model}: test not complete"
    status=json.loads((run/"evaluation_status.json").read_text(encoding="utf-8"))
    assert status["status"]=="EVALUATION_COMPLETE"
    images=pd.read_csv(run/"generated"/"generated_eval_index.csv")
    assert len(images)==256 and images.groupby("label").size().to_dict()=={
        "dekoratif":64,"fauna":64,"flora":64,"geometris":64}
    fid=pd.read_csv(run/"metrics"/"E1_FID_KID_corrected.csv")
    seam=pd.read_csv(run/"metrics"/"E7_seam_gap_summary.csv")
    lpips=pd.read_csv(run/"metrics"/"E2_LPIPS.csv")
    test[model]=dict(images=images,fid=fid,seam=seam,lpips=lpips)
a=test["transformer"]["images"].sort_values(["label","generated_index"]).reset_index(drop=True)
b=test["seam_proposed"]["images"].sort_values(["label","generated_index"]).reset_index(drop=True)
for col in ["label","prompt","source_filename","generated_index","latent_seed"]:
    assert a[col].astype(str).equals(b[col].astype(str)),f"Mismatch in matched key {col}"
assert a["latent_seed"].nunique()==b["latent_seed"].nunique()==1
assert int(a["latent_seed"].iloc[0])==90731
for df in (a,b):
    for fp in df.generated_path:assert Path(fp).is_file()
classes=["flora","fauna","dekoratif","geometris"]
records=[]
for cls in classes:
    for index in (0,1):
        ar=a[(a.label==cls)&(a.generated_index==index)].iloc[0]
        br=b[(b.label==cls)&(b.generated_index==index)].iloc[0]
        records.append(dict(label=cls,generated_index=index,
            matched_prompt=True,matched_latent_seed=True,
            transformer_image_sha256=hashlib.sha256(Path(ar.generated_path).read_bytes()).hexdigest(),
            proposed_image_sha256=hashlib.sha256(Path(br.generated_path).read_bytes()).hexdigest()))
pd.DataFrame(records).to_csv(RES/"matched_image_evidence_hashes.csv",index=False)

def save(filename):
    plt.tight_layout(rect=[0,0,1,.97])
    plt.savefig(FIG/filename,dpi=190,bbox_inches="tight")
    plt.close()
fig,axes=plt.subplots(4,4,figsize=(12,12))
fig.suptitle("Matched V2 test prompts and latent seed | Transformer vs Seam Proposed | seed 42")
for ri,cls in enumerate(classes):
    for cj,(model,index) in enumerate((("transformer",0),("seam_proposed",0),
                                      ("transformer",1),("seam_proposed",1))):
        df=a if model=="transformer" else b
        path=Path(df[(df.label==cls)&(df.generated_index==index)].iloc[0].generated_path)
        with Image.open(path) as im: axes[ri,cj].imshow(im.convert("RGB"))
        axes[ri,cj].set_title(f"{cls.capitalize()} | {model} | #{index}",fontsize=9)
        axes[ri,cj].axis("off")
save("fig09_matched_actual_generated_samples_seed42.png")

fig,axes=plt.subplots(4,2,figsize=(9,16))
fig.suptitle("2x2 periodic tiling of actual V2 generator outputs | seed 42")
for ri,cls in enumerate(classes):
    for cj,model in enumerate(("transformer","seam_proposed")):
        df=a if model=="transformer" else b
        path=Path(df[(df.label==cls)&(df.generated_index==0)].iloc[0].generated_path)
        with Image.open(path) as im:
            rgb=np.asarray(im.convert("RGB"))
        axes[ri,cj].imshow(np.tile(rgb,(2,2,1)))
        axes[ri,cj].set_title(f"{cls.capitalize()} | {model}")
        axes[ri,cj].axis("off")
save("fig10_matched_2x2_tile_visualization_seed42.png")

metric_records=[]
for model in ("transformer","seam_proposed"):
    q=test[model]
    for rec in q["fid"].to_dict("records"):
        cls=rec["class"]
        row=dict(model=model,seed=42,arm="B",metric_scope=("pooled" if cls=="all" else "per_class"),
                 class_label=cls,FID=float(rec["fid"]),KID=float(rec["kid"]),real_n=int(rec["n_real"]),
                 generated_n=int(rec["n_fake"]))
        if cls!="all":
            row["seam_gap"]=float(q["seam"].set_index("label").loc[cls,"mean"])
            row["LPIPS_diversity"]=float(q["lpips"].set_index("label").loc[cls,"lpips_mean"])
        metric_records.append(row)
metric=pd.DataFrame(metric_records)
metric.to_csv(RES/"matched_B_transformer_vs_seam_seed42_test.csv",index=False)
fig,ax=plt.subplots(figsize=(9,4.5))
w=.36;xx=np.arange(4)
for step,model in enumerate(("transformer","seam_proposed")):
    numbers=metric[(metric.model==model)&(metric.class_label.isin(classes))].set_index("class_label").loc[classes]
    ax.bar(xx+step*w,numbers.KID,width=w,label=model)
ax.set_xticks(xx+w/2,classes);ax.set_ylabel("Exploratory KID (lower is better)")
ax.set_title("One-seed held-out test comparison (37 independent real images)")
ax.legend()
save("fig11_matched_seed42_test_KID.png")
summary=[]
for model in ("transformer","seam_proposed"):
    d=metric[(metric.model==model)&(metric.class_label!="all")]
    pooled=metric[(metric.model==model)&(metric.class_label=="all")].iloc[0]
    summary.append(dict(model=model,training_seed=42,arm="B",
        class_macro_FID=d.FID.mean(),class_macro_KID=d.KID.mean(),
        unweighted_class_macro_seam_gap=d.seam_gap.mean(),
        pooled_FID=pooled.FID,pooled_KID=pooled.KID,
        n_test_real=37,n_generated=256))
pd.DataFrame(summary).to_csv(RES/"matched_seed42_summary_exploratory.csv",index=False)
print("MATCHED_2WAY_COMPLETE")
print(pd.DataFrame(summary).to_string(index=False))
print("NOTE: TiPGAN and DF-GAN not run on V2. Do not label this three-way study complete.")
