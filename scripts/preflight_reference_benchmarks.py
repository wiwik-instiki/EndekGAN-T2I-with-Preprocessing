# -*- coding: utf-8 -*-
"""Read-only, source-group-safe TiPGAN / DF-GAN / EndekGAN V2 benchmark preflight.

No model training, image export or private-source redistribution.
Private per-source candidate paths are saved outside the Git checkout.
"""
from pathlib import Path
import json,subprocess,datetime
import numpy as np,pandas as pd

REPO=Path(__file__).resolve().parents[1]
ROOT=REPO.parent
BASE=ROOT/"REFERENCE_BASELINES"
PRIVATE=BASE/"preflight_v2_private"
PRIVATE.mkdir(parents=True,exist_ok=True)
AUDIT=REPO/"results"
AUDIT.mkdir(exist_ok=True)
FOLDER={"A":"full_image_augmented_A","B":"full_plus_crop_ablation_B"}
data={}
for arm in ("A","B"):
    path=ROOT/"TRAINING_SPLITS_PREPROCESSED_V2"/FOLDER[arm]
    data[arm]={s:pd.read_csv(path/(s+"_split.csv")) for s in ("train","val","test")}
    for s in ("train","val","test"):
        frame=data[arm][s]
        assert frame.source_sha256.notna().all()
        assert frame.image_path.map(lambda x:Path(x).is_file()).all(),f"Missing {arm}/{s} images"
    a=data[arm]
    assert not (set(a["train"].source_sha256)&set(a["val"].source_sha256))
    assert not (set(a["train"].source_sha256)&set(a["test"].source_sha256))
    assert not (set(a["val"].source_sha256)&set(a["test"].source_sha256))
    assert {s:len(a[s]) for s in a}==({"train":944,"val":37,"test":37} if arm=="A" else {"train":1280,"val":37,"test":37})
    assert a["train"].source_sha256.nunique()==296
for s in ("train","val","test"):
    assert set(data["A"][s].source_sha256)==set(data["B"][s].source_sha256),"A/B source split mismatch"

# 296 independent training-source examples; keep private paths out of Git.
src=data["B"]["train"]
src=src[(src.augmentation_variant=="original_resized") & (src.is_crop==False)].copy()
assert len(src)==src.source_sha256.nunique()==296
assert src.prompt.nunique()==296
full=src.sort_values(["label","source_sha256"])[[
    "image_path","source_sha256","source_filename","label","prompt","prompt_semantic_verified"
]].copy()
full.to_csv(PRIVATE/"official_reference_training_sources_PRIVATE.csv",index=False)
# A transparent, class-balanced four-example TiPGAN feasibility list (not full benchmark).
pilot=full.groupby("label",sort=True,group_keys=False).head(1)
assert len(pilot)==4 and pilot.label.nunique()==4
pilot.to_csv(PRIVATE/"TIPGAN_FOUR_TRAINING_IMAGES_PRIVATE.csv",index=False)
# DF-GAN adapters require private image+prompt pairs; public only gets counts.
private_dfgan=data["B"]["train"][[
    "image_path","source_sha256","label","prompt","source_filename","is_crop",
    "prompt_semantic_verified"]].copy()
private_dfgan.to_csv(PRIVATE/"DFGAN_B_TRAIN_PAIRS_PRIVATE.csv",index=False)
test=data["B"]["test"][["image_path","source_sha256","label","prompt"]]
test.to_csv(PRIVATE/"MATCHED_HELDOUT_TEST_PRIVATE.csv",index=False)

def commit(repo):
    return subprocess.run(["git","-C",str(repo),"rev-parse","HEAD"],capture_output=True,text=True).stdout.strip()
ti=BASE/"TiPGAN"
df=BASE/"DF-GAN"
assert (ti/"train.py").is_file() and (ti/"inference.py").is_file()
assert (df/"code/src/train.py").is_file() and (df/"code/models/DAMSM.py").is_file()
ti_ckpt=any((ti/"ckpt").rglob("*.pth")) if (ti/"ckpt").is_dir() else False
df_damsm=bool(list((df/"data").rglob("text_encoder*.pth"))) if (df/"data").is_dir() else False
import torch
cuda=torch.cuda.is_available()
total_gpu_gib=round(torch.cuda.get_device_properties(0).total_memory/1024**3,2) if cuda else 0
report={
    "audit_time_local":datetime.datetime.now().isoformat(),
    "independent_original_sources":370,
    "source_split":{"train":296,"val":37,"test":37},
    "derived_train_files_A":944,
    "derived_train_files_B":1280,
    "unique_training_prompt_strings":296,
    "crop_files_B":336,
    "train_val_test_source_overlap":0,
    "A_B_source_membership_identical":True,
    "GPU": {"cuda":cuda,"total_GiB":total_gpu_gib,
            "DFGAN_official_min_GiB":12},
    "TiPGAN":{
        "official_repo":"https://github.com/VickyHEHonghong/TiPGAN",
        "official_commit":commit(ti),
        "official_input":"reference image path",
        "official_train_entrypoint":"train.py",
        "repo_readme_entrypoint_note":"README says train_tipgan.py but checkout provides train.py",
        "prerequisite_VGG_weights_available":ti_ckpt,
        "candidate_independent_train_sources":296,
        "pilot_reference_images_selected_from_training_only":4,
        "V2_dataset_training_status":"NOT_TRAINED",
        "V2_test_tileability_status":"NOT_MEASURED",
        "scientific_scope":"cross-task image-reference tileability; no text alignment scores"
    },
    "DF_GAN":{
        "official_repo":"https://github.com/tobran/DF-GAN",
        "official_commit":commit(df),
        "official_input":"text encoded using DAMSM RNN encoder",
        "available_DAMSM_checkpoint":df_damsm,
        "official_min_GPU_GiB":12,
        "V2_dataset_training_status":"NOT_TRAINED",
        "V2_test_status":"NOT_MEASURED",
        "adaptation_required":"Endek caption tokenizer and DAMSM or explicitly labeled MiniLM adaptation; 8GB VRAM workaround is not original-spec equivalence"
    },
    "EndekGAN_V2":{
        "completed_training_runs":24,
        "V2_test_evaluation_complete_run":"B/seam_proposed/seed42",
        "additional_matched_model":"B/transformer/seed42",
        "external_head_to_head_complete":False
    }
}
(REPO/"results"/"REFERENCE_BASELINE_PREFLIGHT.json").write_text(json.dumps(report,indent=2),encoding="utf-8")
# This file is safe for publication: does not contain local paths, captions, or source IDs.
matrix=[
["method","reference","input_modality","train_source_group","split_identity","trained_on_V2",
 "heldout_evaluation","text_alignment_applicable","tileability_applicable","release_rule"],
["TiPGAN","He et al., Computer-Aided Design 2025","image-reference","train-only texture sources (296 candidates)","grouped296/37/37","NO",
 "NOT_RUN","NO","YES (cross-task)","Use input textures from training split only; never test as training/conditioning"],
["DF-GAN","Tao et al., CVPR 2022","text","grouped source images with verified captions","grouped296/37/37","NO",
 "NOT_RUN","YES","YES","Requires DAMSM/image-caption adaptation; label adaptations explicitly"],
["EndekGAN-V2-SeamProposed","current experiment","Indonesian text + motif class","source-aware A,B","grouped296/37/37","YES",
 "B/seed42 only","YES (pending measurement)","YES","Held-out FID/KID n_real=37 exploratory"],
["EndekGAN-V2-Transformer","internal ablation","Indonesian text + motif class","source-aware A,B","grouped296/37/37","YES",
 "B/seed42 in progress / verify local status","YES (pending measurement)","YES","Same latent seed 90731 and 64 fake/class"]
]
pd.DataFrame(matrix[1:],columns=matrix[0]).to_csv(REPO/"tables"/"REFERENCE_BENCHMARK_MATRIX.csv",index=False)
print(json.dumps(report,indent=2))
print("PRIVATE_INPUT_MANIFESTS:",PRIVATE,"(NOT part of Git checkout)")
