# -*- coding: utf-8 -*-
"""Build source-backed IEEE Access aggregate figures; never export original Endek images or prompts."""
from pathlib import Path
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch,FancyArrowPatch
R=Path(__file__).resolve().parents[1];P=R.parent
F=R/"figures";T=R/"tables";D=R/"docs"
for x in (F,T,D):x.mkdir(parents=True,exist_ok=True)
C=["flora","fauna","dekoratif","geometris"];N=["Flora","Fauna","Decorative","Geometric"]
BLUE="#245785";ORANGE="#BF6B35";GREEN="#327C6D";GRAY="#526674"
plt.rcParams.update({"font.family":"DejaVu Sans","font.size":10,"axes.spines.top":False,
"axes.spines.right":False,"axes.titleweight":"bold","figure.facecolor":"white"})
arms={}
for arm,slug in [("A","full_image_augmented_A"),("B","full_plus_crop_ablation_B")]:
    base=P/"TRAINING_SPLITS_PREPROCESSED_V2"/slug
    arms[arm]={s:pd.read_csv(base/(s+"_split.csv")) for s in ("train","val","test")}
    for s in arms[arm]:
        x=arms[arm][s];assert set(["source_sha256","is_crop","label"]).issubset(x)
        for other in arms[arm]:
            if other!=s:assert not(set(x.source_sha256)&set(arms[arm][other].source_sha256))
assert [len(arms["B"][s]) for s in ("train","val","test")]==[1280,37,37]
assert [arms["B"][s].source_sha256.nunique() for s in ("train","val","test")]==[296,37,37]
assert len(arms["A"]["train"])==944
train=arms["B"]["train"];crop=train.is_crop.astype(str).str.lower().isin(["true","1"])
assert crop.sum()==336
counts=[]
for arm in ("A","B"):
    for split,x in arms[arm].items():
        for cl,name in zip(C,N):
            g=x[x.label==cl]
            counts.append(dict(arm=arm,split=split,motif_class=name,derived_records=len(g),
                independent_sources=g.source_sha256.nunique(),crop_views=int(g.is_crop.astype(str).str.lower().isin(["true","1"]).sum())))
pd.DataFrame(counts).to_csv(T/"TABLE_01_PREPROCESSED_CLASS_DISTRIBUTION.csv",index=False)
pd.DataFrame([
dict(stage="Independent source photographs",training_sources=296,derived_train_views=296,validation_sources=37,test_sources=37),
dict(stage="Arm A: resized and augmented full frames",training_sources=296,derived_train_views=944,validation_sources=37,test_sources=37),
dict(stage="Arm B: Arm A + 336 crops",training_sources=296,derived_train_views=1280,validation_sources=37,test_sources=37)
]).to_csv(T/"TABLE_02_PREPROCESSING_STAGE_COUNTS.csv",index=False)
def save(fig,name):
    fig.savefig(F/name,dpi=230,bbox_inches="tight",facecolor="white")
    plt.close(fig)
    assert (F/name).stat().st_size>15000,name
fig,ax=plt.subplots(figsize=(14,7))
ax.set_xlim(0,14);ax.set_ylim(0,8);ax.axis("off")
def box(x,y,w,h,title,sub,col=BLUE):
    ax.add_patch(FancyBboxPatch((x,y),w,h,boxstyle="round,pad=0.12",ec=col,fc="#F1F5F8",lw=1.6))
    ax.text(x+w/2,y+h*.67,title,ha="center",weight="bold",fontsize=10.4)
    ax.text(x+w/2,y+h*.26,sub,ha="center",fontsize=9.2,color=GRAY)
def ar(a,b):
    ax.add_patch(FancyArrowPatch(a,b,arrowstyle="-|>",mutation_scale=16,lw=1.6,color=GRAY))
box(4.7,6.6,4.6,1,"370 independent source images","SHA-256 source-level grouping",GREEN)
for x in (2.45,7,11.55):
    ax.plot([7,x],[6.55,5.95],lw=1.2,color=GRAY)
box(.3,4.97,4.3,.86,"Train: 296 source images","Augmentation allowed")
box(4.95,4.97,4.1,.86,"Validation: 37 images","No crop or augmentation")
box(9.4,4.97,4.3,.86,"Test: 37 held-out images","No crop or augmentation")
ar((2.45,4.87),(2.45,4.17))
box(.35,3.1,4.25,.95,"RGB resize to 256 x 256","Full-frame augmentation (train only)")
ar((2.45,3.0),(2.45,2.62))
box(.38,1.5,4.2,1,"Arm A: 944 train records","236 derived views per motif class")
ar((4.7,3.55),(6,3.55))
box(6.0,3.1,4.1,.95,"Crop augmentation","336 extra training views",ORANGE)
ar((8.05,3.0),(8.05,2.62))
box(6.0,1.5,4.2,1,"Arm B: 1,280 train records","320 derived views/class, same 296 sources",ORANGE)
ax.text(7,7.86,"Preprocessing workflow used by the V2 experiments",
    fontsize=15,ha="center",va="top",weight="bold")
ax.text(7,.65,"Derived views do not increase the number of independent source images.",
    ha="center",fontsize=10)
ax.text(7,.27,"Zero exact source-SHA256 overlap across splits; near-duplicate audit requires separate visual review.",
    ha="center",fontsize=9,color=GRAY)
save(fig,"fig16_preprocessing_source_aware_pipeline.png")
src_counts=np.array([train[train.label==c].source_sha256.nunique() for c in C])
aug=np.array([len(arms["A"]["train"][arms["A"]["train"].label==c]) for c in C])
cr=np.array([int((crop&(train.label==c)).sum()) for c in C])
assert list(src_counts)==[118,42,90,46] and list(aug)==[236]*4 and list(cr)==[84]*4
fig,(a,b)=plt.subplots(1,2,figsize=(12.6,5.1))
x=np.arange(4)
bars=a.bar(x,src_counts,color=BLUE,width=.58);a.bar_label(bars,padding=3)
a.set_xticks(x,N,rotation=15);a.set_ylim(0,135);a.set_ylabel("Independent source images")
a.set_title("(a) Source composition: train n=296")
b.bar(x,aug,color=BLUE,width=.6,label="Full-frame augmented")
b.bar(x,cr,bottom=aug,color=ORANGE,width=.6,label="Crop-derived")
for i in x:b.text(i,324,"320",ha="center",weight="bold")
b.set_xticks(x,N,rotation=15);b.set_ylim(0,370);b.set_ylabel("Derived train records")
b.set_title("(b) Arm B: 1,280 balanced views")
b.legend(frameon=False,fontsize=8.7)
fig.suptitle("Original sources vs preprocessed training exposure",weight="bold")
fig.tight_layout()
save(fig,"fig17_preprocessed_class_balance.png")
fig,ax=plt.subplots(figsize=(10.8,4.8))
bars=ax.barh([0,1,2],[296,944,1280],color=[GREEN,BLUE,ORANGE],height=.56)
ax.set_yticks([0,1,2],["Unique train sources","Arm A: full-frame views","Arm B: full-frame + crop views"])
ax.invert_yaxis();ax.set_xlim(0,1460);ax.set_xlabel("Count (source count in first row; derived records in lower rows)")
for bar,n in zip(bars,[296,944,1280]):
    ax.text(n+14,bar.get_y()+bar.get_height()/2,f"{n:,}",va="center",weight="bold")
ax.set_title("Dataset exposure across preprocessed training arms")
fig.tight_layout()
save(fig,"fig18_training_exposure_after_preprocessing.png")
held=pd.read_csv(R/"results/heldout_B_transformer_vs_seam_3seeds_per_seed.csv")
assert len(held)==6 and held.groupby("model").seed.nunique().eq(3).all()
metrics=["pooled_FID","pooled_KID","mean_seam_gap","mean_LPIPS_diversity"]
avg=held.groupby("model")[metrics].agg(["mean","std"])
rows=[]
for model in ("transformer","seam_proposed"):
    row=dict(model=model,independent_training_seeds=3,real_reference_count=37,generated_per_seed=256)
    for v in metrics:
        row[v+"_mean"]=avg.loc[model,(v,"mean")]
        row[v+"_sd"]=avg.loc[model,(v,"std")]
    rows.append(row)
pd.DataFrame(rows).to_csv(T/"TABLE_03_MATCHED_HELDOUT_3SEED_MEAN_SD.csv",index=False)
ext=pd.read_csv(R/"results/DFGAN_ACTUAL_TEST_AND_PENDING_20261117.csv")
assert len(ext)==3 and set(ext.seed)=={42,123,2025}
ext.to_csv(T/"TABLE_04_DFGAN_HELDOUT_STATUS.csv",index=False)
fig,(a,b)=plt.subplots(1,2,figsize=(12.6,4.9))
for model,color,label in [("transformer",BLUE,"No-seam control"),
                          ("seam_proposed",ORANGE,"Proposed seam-aware")]:
    d=held[held.model==model].sort_values("seed")
    a.plot(d.seed.astype(str),d.pooled_FID,color=color,marker="o",lw=2,label=label)
    b.plot(d.seed.astype(str),d.mean_seam_gap,color=color,marker="o",lw=2,label=label)
verified=ext[ext.heldout_test_status=="EVALUATION_COMPLETE"]
assert len(verified)==2 and verified.seed.isin([42,123]).all()
a.plot(verified.seed.astype(str),verified.pooled_FID,color=GREEN,marker="D",ls="--",label="DF-GAN–MiniLM adapted (2 seeds)")
b.plot(verified.seed.astype(str),verified.seam_gap,color=GREEN,marker="D",ls="--",label="DF-GAN–MiniLM adapted (2 seeds)")
a.set_title("(a) Pooled FID on held-out test ↓")
b.set_title("(b) Opposite-edge gap on generated tiles ↓")
a.set_ylabel("FID");b.set_ylabel("Edge gap")
for ax in (a,b):ax.set_xlabel("Training seed");ax.grid(axis="y",alpha=.16)
a.legend(frameon=False,fontsize=8)
fig.suptitle("Actual evaluation: seed 2025 DF-GAN test not verified",weight="bold",fontsize=12)
fig.tight_layout()
save(fig,"fig19_heldout_fidelity_vs_edge_gap_by_seed.png")
fig,axes=plt.subplots(1,3,figsize=(13.7,4.6))
for ax,key,title in zip(axes,["pooled_FID","pooled_KID","mean_seam_gap"],
                       ["Pooled FID ↓","Pooled KID ↓","Edge gap ↓"]):
    vals=[avg.loc[m,(key,"mean")] for m in ("transformer","seam_proposed")]
    stds=[avg.loc[m,(key,"std")] for m in ("transformer","seam_proposed")]
    ax.bar([0,1],vals,yerr=stds,capsize=5,color=[BLUE,ORANGE],width=.6)
    ax.set_xticks([0,1],["No-seam control","Seam proposed"],rotation=12,ha="right")
    ax.set_title(title);ax.set_ylim(bottom=0);ax.grid(axis="y",alpha=.14)
fig.suptitle("Actual internal 3-seed held-out test: mean ± sample SD (37 real images)",weight="bold",fontsize=12)
fig.tight_layout()
save(fig,"fig20_internal_3seed_heldout_mean_sd.png")
val=pd.read_csv(R/"results/three_seed_checkpoint_validation_summary.csv")
v=val[val.model.isin(["transformer","seam_proposed"])].set_index(["arm","model"])
assert len(v)==4
fig,ax=plt.subplots(figsize=(10,5))
for i,(mod,color,label) in enumerate([("transformer",BLUE,"No-seam"),("seam_proposed",ORANGE,"Proposed")]):
    scores=[v.loc[(arm,mod),"mean_KID"] for arm in ("A","B")]
    sd=[v.loc[(arm,mod),"sd_KID"] for arm in ("A","B")]
    ax.bar(np.arange(2)+(i-.5)*.33,scores,width=.33,yerr=sd,capsize=4,color=color,label=label)
ax.set_xticks([0,1],["Arm A (944 train)","Arm B (1,280 train)"])
ax.set_ylabel("Validation KID ↓ (not test KID)")
ax.set_title("A vs B preprocessing: checkpoint validation quality (3 seeds)")
ax.set_ylim(0,.33);ax.legend(frameon=False);ax.grid(axis="y",alpha=.18)
fig.tight_layout()
save(fig,"fig21_preprocessing_A_B_validation_kid.png")
lines=[
"# IEEE Access Visual Evidence: Preprocessing and Evaluation","",
"Working title: EndekGAN-T2I: Evaluating the Fidelity–Tileability Trade-off in Text-Conditioned Balinese Endek Motif Generation.",
"",
"## Preprocessing figures and tables","",
"![Source-aware pipeline](../figures/fig16_preprocessing_source_aware_pipeline.png)",
"**Figure 16:** Actual source-aware preprocessing: independent 296/37/37 partition; Arm A 944 and Arm B 1280 training views, including 336 extra train-only crops.",
"",
"![Class balance](../figures/fig17_preprocessed_class_balance.png)",
"**Figure 17:** Independent image source counts and balanced derived training views across four classes. Never conflate the units.",
"",
"![Training exposure](../figures/fig18_training_exposure_after_preprocessing.png)",
"**Figure 18:** Source count versus Arm A and Arm B derived training-file counts.",
"",
"Preprocessing tables: [Table 1, class/arm/split distribution](../tables/TABLE_01_PREPROCESSED_CLASS_DISTRIBUTION.csv) and [Table 2, workflow stage counts](../tables/TABLE_02_PREPROCESSING_STAGE_COUNTS.csv).",
"",
"## Evaluation figures and tables","",
"![Seed-wise held-out test](../figures/fig19_heldout_fidelity_vs_edge_gap_by_seed.png)",
"**Figure 19:** Actual held-out pooled FID versus mean opposite-border gap; adapted DF-GAN has test results for seeds 42 and 123, not 2025.",
"",
"![3-seed aggregate](../figures/fig20_internal_3seed_heldout_mean_sd.png)",
"**Figure 20:** Internal Transformer/no-seam control versus EndekGAN Seam Proposed; three independent training seeds, mean ± sample SD; 37 original held-out real references.",
"",
"![Preprocessing ablation validation](../figures/fig21_preprocessing_A_B_validation_kid.png)",
"**Figure 21:** Arm A vs Arm B validation KID; do not mistake validation for held-out evaluation.",
"",
"Evaluation tables: [Table 3, actual 3-seed internal test metrics](../tables/TABLE_03_MATCHED_HELDOUT_3SEED_MEAN_SD.csv) and [Table 4, external DF-GAN test status](../tables/TABLE_04_DFGAN_HELDOUT_STATUS.csv).",
"",
"Additional source-derived training figures: [validation KID](../figures/fig02_all_24_validation_KID.png), [generator loss](../figures/fig04_generator_loss.png), [discriminator loss](../figures/fig05_discriminator_loss.png), [seam loss](../figures/fig06_seam_loss.png), and [three-seed generated examples](../figures/fig14_actual_B_generated_motifs_3seed_comparison.png).",
"",
"## Confirmed post-preprocessing dataset","",
"| Class | Arm B train | Validation | Held-out test | Independent train sources |",
"|---|---:|---:|---:|---:|"]
for cl,name,src_count in zip(C,N,src_counts):
    lines.append(f"| {name} | {int((train.label==cl).sum())} | {int((arms['B']['val'].label==cl).sum())} | {int((arms['B']['test'].label==cl).sum())} | {int(src_count)} |")
lines+=["| **Total** | **1,280** | **37** | **37** | **296** |","",
"Arm A: 944 train records, same 296 independent train sources. Arm B: 944 full-frame/augmented records + 336 crop-derived records. Same validation/test split for both. Source SHA-256 overlap across splits: zero.",
"Prompt semantic and expert_verified flags remain false in the manifests. Additional source photographs from external datasets were NOT merged.",
"",
"## Three-seed outcomes and limitations","",
"| Model | Pooled FID mean ± SD ↓ | Pooled KID mean ± SD ↓ | Edge gap mean ± SD ↓ |",
"|---|---:|---:|---:|"]
for m in ("transformer","seam_proposed"):
    row=[f"{avg.loc[m,(k,'mean')]:.5f} ± {avg.loc[m,(k,'std')]:.5f}" for k in ("pooled_FID","pooled_KID","mean_seam_gap")]
    lines.append("| "+m+" | "+" | ".join(row)+" |")
lines+=["",
"Seam Proposed reduces mean edge gap by approximately 40.65% but increases mean FID/KID and reduces LPIPS diversity. FID/KID with 37 real reference images are exploratory.",
"TiPGAN is IMAGE-conditioned tileable texture synthesis, not prompt-conditioned text-to-image: only a 30-iteration one-image compatibility pilot exists; no comparable converged test scores.",
"Adapted DF-GAN–MiniLM is not canonical DAMSM-conditioned DF-GAN; its third held-out seed is pending verification, so do not compute three-seed means.",
"Neither unverified prompt-image semantic alignment nor cultural authenticity of generated Endek is established by these metrics.",
"",
"## Main references",
"1. He et al. (2025), TiPGAN, Computer-Aided Design 183, 103866, DOI: https://doi.org/10.1016/j.cad.2025.103866. Primary cloth-tileability, image-conditioned reference.",
"2. Tao et al. (2022), DF-GAN, CVPR, DOI: https://doi.org/10.1109/CVPR52688.2022.01602. Secondary text-to-image comparator.",
"",
"Implementation and data-exposure caveat: multilingual text conditioning and seam regularization existed in the earlier EndekGAN-T2I study. V2 investigates source-aware preprocessing and a measured fidelity–tileability trade-off; it does not claim to invent these components.",
"",
"## Reproduce locally",
"Execute the script scripts/build_ieee_access_visual_evidence.py from the authorized V2 project workspace; raw images, private paths, full manifests with prompts and checkpoint weights must remain private."
]
(D/"VISUAL_REPORT_PREPROCESSING_AND_EVALUATION.md").write_text("\n".join(lines)+"\n",encoding="utf-8")
(D/"VISUAL_EVIDENCE_SOURCE_AUDIT.json").write_text(json.dumps({
 "preprocessed_B":[1280,37,37],"independent_sources":[296,37,37],"Arm_A_train_views":944,
 "Arm_B_crop_views":336,"source_overlap":0,"heldout_internal_runs":6,
 "DFGAN_test_seeds_reported":[42,123],"DFGAN_2025_test_not_reported":True,
 "new_figures":[f"fig{x}" for x in range(16,22)],"raw_source_images_exported":False},indent=2)+"\n",encoding="utf-8")
print("GENERATED",6,"figures",4,"tables","TOTAL_ARM_B",len(train)+37+37,flush=True)
