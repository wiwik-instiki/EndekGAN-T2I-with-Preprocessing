# -*- coding: utf-8 -*-
"""Aggregate ACTUAL 3-seed held-out EndekGAN V2 results. Never trains or reads source photos."""
from pathlib import Path
import hashlib,json
import numpy as np,pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image

REPO=Path(__file__).resolve().parents[1]
ROOT=REPO.parent
RUNS=ROOT/"runs_preprocessed"/"full_plus_crop_ablation_B"
RESULTS=REPO/"results"
FIG=REPO/"figures"
TABLE=REPO/"tables"
BENCH=REPO/"benchmarks"
for d in (RESULTS,FIG,TABLE,BENCH):d.mkdir(parents=True,exist_ok=True)
MODELS=("transformer","seam_proposed")
SEEDS=(42,123,2025)
CLASSES=("flora","fauna","dekoratif","geometris")
rows,refs=[],[]
shared_ids=None

for seed in SEEDS:
    per_seed=[]
    for model in MODELS:
        target=RUNS/f"{model}_seed{seed}"/"test_evaluation"
        status=json.loads((target/"evaluation_status.json").read_text(encoding="utf-8"))
        assert status["status"]=="EVALUATION_COMPLETE"
        assert int(status.get("raw_test_n",37))==37
        evalidx=pd.read_csv(target/"generated"/"generated_eval_index.csv")
        assert len(evalidx)==256
        assert evalidx.groupby("label").size().to_dict()=={c:64 for c in sorted(CLASSES)}
        evalidx=evalidx.sort_values(["label","generated_index"]).reset_index(drop=True)
        if "latent_seed" in evalidx:
            assert (evalidx.latent_seed==90731).all()
        for p in evalidx.generated_path:
            assert Path(p).is_file() and Path(p).stat().st_size>0
        compare=evalidx[["label","prompt","source_filename","generated_index"]].astype(str)
        if shared_ids is None:shared_ids=compare
        else:
            assert compare.equals(shared_ids),f"Input mismatch {seed}/{model}"
        vals=pd.read_csv(target/"metrics"/"E1_FID_KID_corrected.csv")
        gaps=pd.read_csv(target/"metrics"/"E7_seam_gap_summary.csv")
        lpips=pd.read_csv(target/"metrics"/"E2_LPIPS.csv")
        assert len(vals)==5 and set(vals["class"])==set(CLASSES)|{"all"}
        assert set(gaps.label)==set(lpips.label)==set(CLASSES)
        assert int(vals.loc[vals["class"]=="all","n_real"].iloc[0])==37
        assert int(vals.loc[vals["class"]=="all","n_fake"].iloc[0])==256
        for rr in vals.to_dict("records"):
            label=rr["class"]
            gaprow=gaps[gaps.label==label].iloc[0] if label!="all" else None
            lp=lpips[lpips.label==label].iloc[0] if label!="all" else None
            rows.append(dict(model=model,seed=seed,class_label=label,metric_grain="pooled" if label=="all" else "per_class",
               FID=float(rr["fid"]),KID=float(rr["kid"]),real_n=int(rr["n_real"]),fake_n=int(rr["n_fake"]),
               kid_subset_sd=float(rr["kid_sd"]),
               seam_gap=float(gaprow["mean"]) if gaprow is not None else np.nan,
               LPIPS_diversity=float(lp["lpips_mean"]) if lp is not None else np.nan))
        macro=vals[vals["class"]!="all"]
        pooled=vals[vals["class"]=="all"].iloc[0]
        per_seed.append(dict(model=model,seed=seed,
            pooled_FID=float(pooled.fid),pooled_KID=float(pooled.kid),
            class_macro_FID=float(macro.fid.mean()),class_macro_KID=float(macro.kid.mean()),
            mean_seam_gap=float(gaps["mean"].mean()),
            mean_LPIPS_diversity=float(lpips.lpips_mean.mean()),
            real_ref_n=37,generated_n=256,latent_seed=90731))
    assert len(per_seed)==2
    refs+=per_seed

ALL=pd.DataFrame(rows)
BY_SEED=pd.DataFrame(refs)
assert len(ALL)==30 and len(BY_SEED)==6
ALL.to_csv(RESULTS/"heldout_B_transformer_vs_seam_all3seeds_allclasses.csv",index=False,encoding="utf-8-sig")
BY_SEED.to_csv(RESULTS/"heldout_B_transformer_vs_seam_3seeds_per_seed.csv",index=False,encoding="utf-8-sig")
COLS=("pooled_FID","pooled_KID","class_macro_FID","class_macro_KID","mean_seam_gap","mean_LPIPS_diversity")
sum_rows=[]
for model,sub in BY_SEED.groupby("model",sort=False):
    out=dict(model=model,seeds=3)
    for metric in COLS:
        out[metric+"_mean"]=float(sub[metric].mean())
        out[metric+"_sd"]=float(sub[metric].std(ddof=1))
    sum_rows.append(out)
SUMMARY=pd.DataFrame(sum_rows)
SUMMARY.to_csv(RESULTS/"heldout_B_3seeds_mean_sd.csv",index=False,encoding="utf-8-sig")
base=BY_SEED[BY_SEED.model=="transformer"].sort_values("seed").reset_index(drop=True)
proposed=BY_SEED[BY_SEED.model=="seam_proposed"].sort_values("seed").reset_index(drop=True)
assert base.seed.equals(proposed.seed)
D=base[["seed"]].copy()
for metric in COLS:
    D["delta_proposed_minus_transformer_"+metric]=proposed[metric]-base[metric]
D.to_csv(RESULTS/"heldout_B_3seeds_paired_deltas.csv",index=False,encoding="utf-8-sig")

plt.rcParams.update({"font.size":10,"axes.titlesize":11,"savefig.dpi":180})
def save(name):
    plt.tight_layout()
    plt.savefig(FIG/name,bbox_inches="tight")
    plt.close()
fig,axs=plt.subplots(1,3,figsize=(13.5,4))
for ax,(metric,title) in zip(axs,(("pooled_FID","Pooled FID ↓"),("pooled_KID","Pooled KID ↓"),("mean_seam_gap","Opposite-border gap ↓"))):
    for model in MODELS:
        sub=BY_SEED[BY_SEED.model==model].sort_values("seed")
        ax.plot(sub.seed.astype(str),sub[metric],"-o",label=model)
    ax.set_title(title);ax.set_xlabel("Training seed");ax.grid(alpha=.17)
axs[0].legend(fontsize=8)
save("fig12_heldout_B_test_three_seeds_tradeoff.png")
fig,axs=plt.subplots(1,3,figsize=(13.5,4))
x=np.arange(2)
for ax,metric,title in zip(axs,("pooled_FID","pooled_KID","mean_seam_gap"),
                           ("Pooled FID","Pooled KID","Opposite-border gap")):
    means=[SUMMARY.set_index("model").loc[m,metric+"_mean"] for m in MODELS]
    sds=[SUMMARY.set_index("model").loc[m,metric+"_sd"] for m in MODELS]
    ax.bar(x,means,yerr=sds,capsize=4,alpha=.85)
    ax.set_xticks(x,["Transformer","Seam Proposed"],rotation=10)
    ax.set_title(title+" ↓ (mean ± SD, 3 seeds)")
    ax.grid(axis="y",alpha=.15)
save("fig13_heldout_B_test_three_seed_mean_sd.png")

# One fixed sample per class/seed/model, not hand-picked.
fig,ax=plt.subplots(len(CLASSES)*len(SEEDS),2,figsize=(6.5,3.2*len(CLASSES)*len(SEEDS)))
for si,seed in enumerate(SEEDS):
    for ci,label in enumerate(CLASSES):
        for mi,model in enumerate(MODELS):
            index=pd.read_csv(RUNS/f"{model}_seed{seed}"/"test_evaluation"/"generated"/"generated_eval_index.csv")
            first=index[(index.label==label)&(index.generated_index==0)].iloc[0]
            with Image.open(first.generated_path) as im:
                image=np.array(im.convert("RGB"))
            ax[si*len(CLASSES)+ci,mi].imshow(image)
            ax[si*len(CLASSES)+ci,mi].set_title(f"Seed {seed} · {label} · {model}",fontsize=9)
            ax[si*len(CLASSES)+ci,mi].axis("off")
save("fig14_actual_B_generated_motifs_3seed_comparison.png")

report=[
"# EndekGAN V2 — Three-Seed Held-Out Test (Verified)",
"",
"Methods: Transformer vs Seam Proposed, each seeds 42/123/2025, same grouped 296/37/37 split,",
"64 generated samples per class, 256 total per run, deterministic latent RNG seed 90731.",
"Test-based scores are reported **only after choosing best checkpoints from validation KID**.",
"**Caution:** FID/KID calculated with 37 real test sources (5–15 per class) are exploratory and sample-size biased.",
"",
"## Table A. Three-seed held-out test mean ± sample SD",
"",
"| Model | Pooled FID | Pooled KID | Class macro FID | Class macro KID | Seam gap |",
"|---|---:|---:|---:|---:|---:|"
]
for rr in SUMMARY.itertuples(index=False):
    report.append("| "+rr.model+" | "+" | ".join(f"{getattr(rr,m+'_mean'):.4f} ± {getattr(rr,m+'_sd'):.4f}" for m in COLS[:5])+" |")
report+=["","## Table B. Per-seed values","",
"| Model | Seed | Pooled FID | Pooled KID | Mean gap |",
"|---|---:|---:|---:|---:|"]
for rr in BY_SEED.itertuples(index=False):
    report.append(f"| {rr.model} | {rr.seed} | {rr.pooled_FID:.4f} | {rr.pooled_KID:.6f} | {rr.mean_seam_gap:.6f} |")
report+=["","## Figures",
"","![Per-seed tradeoffs](../figures/fig12_heldout_B_test_three_seeds_tradeoff.png)",
"","**Figure 12.** Matched-seed pooled test FID/KID and opposite-border gap for each seed. Smaller values are better; these axes measure different phenomena.",
"","![Three-seed mean and standard deviation](../figures/fig13_heldout_B_test_three_seed_mean_sd.png)",
"","**Figure 13.** Mean ± sample SD over three training seeds, not a confidence interval and not a formal significance test.",
"","![Actual generated outputs](../figures/fig14_actual_B_generated_motifs_3seed_comparison.png)",
"","**Figure 14.** Actual generated outputs at the first predetermined sample index for each class, model and seed; none cherry-picked.",
"","## Reporting limitations",
"- Only two INTERNAL V2 architectures were compared; official TiPGAN and DF-GAN have not been retrained on the new Endek V2 data.",
"- TiPGAN is image-conditioned cloth texturing; its input modality cannot be treated as prompt-based.",
"- Small genuine image counts limit FID/KID reliability and do not establish cultural authenticity.",
"- The 1280 image files are source-derived views from 296 originals, not 1280 independent Endek motifs.",
"- No independent Endek expert evaluation or quantified semantic prompt adherence has yet been performed.",
"- These values represent the original training seed effects and finite sampled generated images, not definitive generalization evidence."
]
(BENCH/"THREE_SEED_TEST_REPORT.md").write_text("\n".join(report)+"\n",encoding="utf-8")
print("AGGREGATE_3SEEDS_COMPLETE")
print(SUMMARY.to_string(index=False))
print("PAIRED_DELTAS")
print(D.to_string(index=False))
