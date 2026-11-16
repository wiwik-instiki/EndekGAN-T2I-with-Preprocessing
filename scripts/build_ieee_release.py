# -*- coding: utf-8 -*-
"""Compile public V2 evidence from 24 completed local training runs (no GPU training)."""
from pathlib import Path
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

REPO=Path(__file__).resolve().parents[1]
SRC=REPO.parent
RES=REPO/"results"
FIG=REPO/"figures"
TABLES=REPO/"tables"
for d in (RES,FIG,TABLES,FIG/"comparisons"):
    d.mkdir(exist_ok=True,parents=True)
ARM={"A":"full_image_augmented_A","B":"full_plus_crop_ablation_B"}
MODELS={"B":["cat","hash_text","transformer","transformer_noclass","seam_proposed","seam_noclass"],
        "A":["transformer","seam_proposed"]}
SEEDS=(42,123,2025)
batch=json.loads((SRC/"RECOVERY_BATCH_V2"/"batch_progress.json").read_text(encoding="utf-8"))
assert batch["state"]=="COMPLETED" and batch["completed"]==24
rows=[]; validation=[]; losses=[]
for arm in ("B","A"):
    for model in MODELS[arm]:
        for seed in SEEDS:
            path=SRC/"runs_preprocessed"/ARM[arm]/(model+"_seed"+str(seed))
            state=json.loads((path/"run_state.json").read_text(encoding="utf-8"))
            assert state["completed"]
            for field in ("best_checkpoint","last_checkpoint"):
                checkpoint=Path(state[field])
                assert checkpoint.is_file() and checkpoint.stat().st_size>0
            train=pd.read_csv(path/"training_log.csv")
            vals=pd.read_csv(path/"validation_selection_log.csv")
            assert len(train)==state["actual_epochs"] and int(train.epoch.max())==state["actual_epochs"]
            assert np.isclose(vals.loc[vals.epoch==state["best_epoch"],"val_KID"].iloc[0],state["best_val_KID"])
            rows.append(dict(arm=arm,model=model,seed=seed,status=state["status"],
                trained_epochs=state["actual_epochs"],best_epoch=state["best_epoch"],
                best_validation_KID=state["best_val_KID"],global_steps=state["global_steps"],
                best_and_last_checkpoint_verified=True))
            for rec in vals.to_dict("records"):
                validation.append(dict(arm=arm,model=model,seed=seed,
                    epoch=rec["epoch"],val_KID=rec["val_KID"],
                    val_KID_subset_std=rec.get("val_KID_subset_std")))
            for rec in train.to_dict("records"):
                losses.append(dict(arm=arm,model=model,seed=seed,
                    epoch=rec["epoch"],g_loss=rec["g_loss"],d_loss=rec["d_loss"],
                    seam_loss=rec.get("seam_loss"),fm_loss=rec.get("fm_loss")))
R=pd.DataFrame(rows);V=pd.DataFrame(validation);L=pd.DataFrame(losses)
assert len(R)==24
R.to_csv(RES/"all_24_runs.csv",index=False)
V.to_csv(RES/"all_24_validation_kid.csv",index=False)
L.to_csv(RES/"all_24_training_losses.csv",index=False)
G=R.groupby(["arm","model"],as_index=False).agg(
    n_seeds=("seed","size"),mean_KID=("best_validation_KID","mean"),
    sd_KID=("best_validation_KID","std"),mean_trained_epochs=("trained_epochs","mean"))
G.to_csv(RES/"three_seed_checkpoint_validation_summary.csv",index=False)
pair=[]
for arm in ("A","B"):
    for seed in SEEDS:
        base=R[(R.arm==arm)&(R.model=="transformer")&(R.seed==seed)].iloc[0]
        proposed=R[(R.arm==arm)&(R.model=="seam_proposed")&(R.seed==seed)].iloc[0]
        pair.append(dict(arm=arm,seed=seed,transformer_KID=base.best_validation_KID,
            seam_proposed_KID=proposed.best_validation_KID,
            delta_proposed_minus_transformer=proposed.best_validation_KID-base.best_validation_KID))
P=pd.DataFrame(pair)
P.to_csv(RES/"matched_validation_seam_ablation.csv",index=False)
splits=[]
for arm,folder in ARM.items():
    for split in ("train","val","test"):
        m=pd.read_csv(SRC/"TRAINING_SPLITS_PREPROCESSED_V2"/folder/(split+"_split.csv"))
        splits.append(dict(arm=arm,split=split,derived_files=len(m),
            independent_source_images=m.source_sha256.nunique(),unique_prompts=m.prompt.nunique(),
            crop_files=int((m.is_crop.astype(str).str.lower()=="true").sum())))
S=pd.DataFrame(splits)
S.to_csv(RES/"dataset_independence_audit.csv",index=False)
assert S.query("arm=='B' and split=='train'").iloc[0].independent_source_images==296

TEST=SRC/"runs_preprocessed"/ARM["B"]/"seam_proposed_seed42"/"test_evaluation"
assert json.loads((TEST/"evaluation_status.json").read_text(encoding="utf-8"))["status"]=="EVALUATION_COMPLETE"
T=pd.read_csv(TEST/"metrics"/"E1_FID_KID_corrected.csv")
T.to_csv(RES/"test_FID_KID_ONLY_B_proposed_seed42.csv",index=False)
pd.read_csv(TEST/"metrics"/"E2_LPIPS.csv").to_csv(RES/"test_LPIPS_ONLY_B_proposed_seed42.csv",index=False)
pd.read_csv(TEST/"metrics"/"E7_seam_gap_summary.csv").to_csv(RES/"test_seam_gap_ONLY_B_proposed_seed42.csv",index=False)
assert len(T)==5 and T.loc[T["class"]=="all","n_real"].iloc[0]==37

plt.rcParams.update({"font.size":10,"axes.titlesize":12,"savefig.dpi":180})
def save(name):
    plt.tight_layout()
    plt.savefig(FIG/name,bbox_inches="tight")
    plt.close()
plt.figure(figsize=(8.5,4.7))
plt.bar(["A: full + augment","B: full + augment + crop"],[944,1280],label="Derived training files")
plt.axhline(296,ls="--",color="black",label="Independent training-source images = 296")
plt.ylabel("Count");plt.title("Preprocessing expands views, not independent sources");plt.legend()
save("fig01_dataset_source_vs_derived.png")
order=G.sort_values("mean_KID")
fig,ax=plt.subplots(figsize=(10.5,5.6))
xx=np.arange(len(order))
ax.bar(xx,order.mean_KID,yerr=order.sd_KID,capsize=3,label="Three-seed mean + SD")
for j,row in enumerate(order.itertuples(index=False)):
    sub=R[(R.arm==row.arm)&(R.model==row.model)]
    ax.scatter([j]*len(sub),sub.best_validation_KID,s=18,c="black",zorder=5)
ax.set_xticks(xx,[f"{row.arm}: {row.model}" for row in order.itertuples(index=False)],
              rotation=32,ha="right")
ax.set_ylabel("Validation KID (lower is better)")
ax.set_title("24 training runs; checkpoint-selection validation, not test")
save("fig02_all_24_validation_KID.png")
plt.figure(figsize=(8.5,4.5))
for arm in ("A","B"):
    for seed in SEEDS:
        r=P.query("arm==@arm and seed==@seed").iloc[0]
        offset=0 if arm=="A" else 2
        plt.plot([0+offset,1+offset],[r.transformer_KID,r.seam_proposed_KID],"-o",
                 label=f"Arm {arm}, seed {seed}")
plt.xticks([0,1,2,3],["A transformer","A proposed","B transformer","B proposed"])
plt.ylabel("Validation KID");plt.title("Matched-seed seam ablations")
plt.legend(fontsize=8,ncol=2)
save("fig03_matched_seam_ablation.png")
for metric,filename in [
    ("g_loss","fig04_generator_loss.png"),
    ("d_loss","fig05_discriminator_loss.png"),
    ("seam_loss","fig06_seam_loss.png")]:
    plt.figure(figsize=(9,4))
    for seed in SEEDS:
        z=L.query("arm=='B' and model=='seam_proposed' and seed==@seed")
        plt.plot(z.epoch,z[metric],label=f"Seed {seed}")
    plt.xlabel("Epoch");plt.ylabel(metric);plt.title("B seam proposed: actual training "+metric)
    plt.grid(alpha=.18);plt.legend()
    save(filename)
plt.figure(figsize=(9,4))
for seed in SEEDS:
    z=V.query("arm=='B' and model=='seam_proposed' and seed==@seed")
    plt.plot(z.epoch,z.val_KID,marker="o",markersize=3,label=f"Seed {seed}")
plt.xlabel("Epoch");plt.ylabel("Validation KID");plt.title("Proposed model checkpoint-selection trajectories")
plt.legend();plt.grid(alpha=.18)
save("fig07_proposed_validation_trajectory.png")
classes=T[T["class"]!="all"]
plt.figure(figsize=(8.5,4))
plt.bar(classes["class"],classes["kid"])
plt.title("ONE evaluated model only: B seam_proposed seed42")
plt.ylabel("Exploratory test KID (37 total real references)")
save("fig08_single_model_test_KID.png")

M=["# Tables and figure legends for pre-manuscript review","",
"## Table 1. Dataset sources and derived training records",
"Table 1 distinguishes independent source images from their derived training files.","",
"| Arm | Split | Derived files | Unique sources | Unique prompts | Crops |",
"|---|---|---:|---:|---:|---:|"]
for row in S.itertuples(index=False):
    M.append(f"| {row.arm} | {row.split} | {row.derived_files} | {row.independent_source_images} | {row.unique_prompts} | {row.crop_files} |")
M+=["","**Table 1.** Source-aware split; augmentation/cropping does not create independent sources.","",
"## Table 2. Training outcomes for all 24 model-seed jobs","Table 2 lists real checkpoints selected by validation KID.","",
"| Arm | Model | Seed | Epochs | Best epoch | Best validation KID |",
"|---|---|---:|---:|---:|---:|"]
for row in R.itertuples(index=False):
    M.append(f"| {row.arm} | {row.model} | {row.seed} | {row.trained_epochs} | {row.best_epoch} | {row.best_validation_KID:.6f} |")
M+=["","**Table 2.** All best/last checkpoints and loss logs were verified.","",
"## Table 3. Three-seed validation scores","Table 3 reports checkpoint-selection KID, NOT independent test-set performance.","",
"| Arm | Model | Seeds | KID mean | KID SD |",
"|---|---|---:|---:|---:|"]
for row in G.itertuples(index=False):
    M.append(f"| {row.arm} | {row.model} | {row.n_seeds} | {row.mean_KID:.6f} | {row.sd_KID:.6f} |")
M+=["","**Table 3.** Mean ± sample SD across three training seeds; validation n=37.","",
"## Table 4. Available exploratory test scores","Table 4 contains results from only ONE completed evaluation, not all 24 models.","",
"| Test model | Class | FID | KID | Real n | Fake n |",
"|---|---|---:|---:|---:|---:|"]
for rec in T.to_dict("records"):
    M.append(f"| B seam proposed seed42 | {rec['class']} | {rec['fid']:.3f} | {rec['kid']:.6f} | {rec['n_real']} | {rec['n_fake']} |")
M+=["","**Table 4.** Class-wise test reference numbers are extremely small (5–15). FID/KID are exploratory.","",
"## Figure legends and interpretation",
"Figures 1–8 are derived from actual V2 data; qualitative comparisons (when present) must be labeled GAN-generated only.",
"- Figure 1: independent training sources vs derived image-file exposures.",
"- Figure 2: training-run validation KID (24 jobs, 3 seeds, 8 settings).",
"- Figure 3: paired transformer/seam-proposed validation ablation by seed and preprocessing arm.",
"- Figures 4–6: proposed model training generator, discriminator and seam loss.",
"- Figure 7: proposed model validation KID trajectory by seed.",
"- Figure 8: exploratory test KID for only the one evaluated V2 model.","",
"**Limitations:** 370 independent raw images total, 296 training sources, unchanged 37 val and 37 test images. The extra derived files (944 vs 1280) are not new independent motif images. Crop prompts lack independent semantic validation. External source images and independent external model evaluations were not added to this V2 training matrix. No verified expert study or independent test scores for the other 23 runs are claimed."]
(TABLES/"TABLES_AND_CAPTIONS.md").write_text("\n".join(M)+"\n",encoding="utf-8")
meta={"status":"TRAINING_COMPLETE_TEST_PARTIAL","training_runs":24,"checkpoints_verified":24,
      "unique_raw_images":370,"unique_train_sources":296,"training_rows_A":944,
      "training_rows_B":1280,"validation_sources":37,"test_sources":37,
      "evaluated_test_model":"B/seam_proposed/seed42","pending_test_models":23,
      "test_metrics_status":"EXPLORATORY_ONLY","raw_images_published":False}
(RES/"EVIDENCE_STATUS.json").write_text(json.dumps(meta,indent=2),encoding="utf-8")
print("BUILD_OK",meta)
