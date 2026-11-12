# -*- coding: utf-8 -*-
"""Stage strictly traceable EndekGAN-T2I V2 experiments, without training models."""
from pathlib import Path
import csv, hashlib, json, sys, collections, importlib.util
ROOT=Path(r"D:\DISERTASI_S3\WIWIK\EXPERIMEN_2")
DATA=ROOT/"ENDEK_AUGMENTED_V1"
PROJECT=ROOT/"ENDEK_T2I_EXPERIMENT_V2"
META=DATA/"metadata_expert_guided_T2I_augmented_V1.csv"
ORIGINAL=Path(r"D:\DISERTASI_S3\WIWIK\DATASET_L2\EndekGAN\EndekGAN-T2I\EndekGAN_T2I_IEEE_FULL_CPU_EarlyStopping_V5.ipynb")
CLASSES=("flora","fauna","dekoratif","geometris")
SEEDS=(42,123,2025)
VARIANTS=[
 ("cat",          "cat",              0.0,False, "internal"),
 ("hash_text",    "hash_text",        0.0,False, "internal"),
 ("transformer",  "transformer_text", 0.0,False, "internal"),
 ("transformer_noclass", "transformer_text",0.0,True, "internal"),
 ("seam_proposed","transformer_text", 0.5,False, "internal"),
 ("seam_noclass","transformer_text",  0.5,True,  "internal"),
 ("dfgan_minilm", "transformer_text",  None,None,"external_requires_separate_dfgan_runner"),
]

def load(path):
    with path.open("r",encoding="utf-8-sig",newline="") as f:return list(csv.DictReader(f))
def write_csv(path,rows,cols):
    path.parent.mkdir(parents=True,exist_ok=True)
    with path.open("w",encoding="utf-8-sig",newline="") as f:
        wr=csv.DictWriter(f,fieldnames=cols,extrasaction="ignore");wr.writeheader();wr.writerows(rows)

def main():
    PROJECT.mkdir(exist_ok=True,parents=True)
    data=load(META)
    if len(data)!=1018: raise ValueError("Expected 1018 images")
    if {r["split"] for r in data}!={"train","val","test"}:raise ValueError("Bad splits")
    seen=collections.defaultdict(set)
    for r in data:
        if not Path(r["image_path"]).is_file():raise FileNotFoundError(r["image_path"])
        if not r["prompt"].strip():raise ValueError("blank text for "+r["filename"])
        if r["label"] not in CLASSES:raise ValueError("unknown class")
        if int(r["class_idx"])!=CLASSES.index(r["label"]):raise ValueError("class index mismatch")
        seen[r["source_sha256"]].add(r["split"])
    if len(seen)!=370 or any(len(x)!=1 for x in seen.values()):raise ValueError("source-level leakage")
    if collections.Counter(r["split"] for r in data)!={"train":944,"val":37,"test":37}:raise ValueError("invalid counts")
    for r in data:
        if r["split"]!="train" and r["is_augmented"]!="0":
            raise ValueError("val/test contains augmentation")
    n=collections.Counter(r["label"] for r in data if r["split"]=="train")
    if any(n[c]!=236 for c in CLASSES):raise ValueError("offline imbalance")
    baseline=[r for r in data if r["is_augmented"]=="0"]
    if len(baseline)!=370 or len({r["source_sha256"] for r in baseline})!=370:
        raise ValueError("not exactly one original image per source")
    fields=["filename","label","image_path","md5","prompt","title","motif_source","technique",
            "layout_pattern","composition","color_style","weavability_note","cultural_context",
            "expert_verified","prompt_origin","class_idx","source_filename","source_sha256",
            "augmentation_variant","split"]
    def adapt(r):
        a={k:r.get(k,"") for k in fields}
        a["md5"]=r["source_md5"] # source MD5 preserved; processed image has distinct bytes
        a["expert_verified"]="False" # neither source descriptions nor generated images expert scored yet
        a["prompt_origin"]="inherited_original_metadata_unverified_per_image"
        return a
    for label,records in [("source256",baseline),("augmented256",data)]:
        for split in ["train","val","test"]:
            part=[adapt(r) for r in records if r["split"]==split]
            write_csv(PROJECT/"splits"/label/(split+"_split.csv"),part,fields)
    old_code=json.loads(ORIGINAL.read_text(encoding="utf-8"))
    src="".join(old_code["cells"][6]["source"])
    if "def model_classes()" not in src or "def train(cfg:" not in src or "def seam_loss(" not in src:
        raise RuntimeError("Original model implementation signatures missing")
    # Preserve original GAN architecture, loss functions and optimizer verbatim.
    # Only adjust sampler: balanced offline data -> no replacement; imbalanced originals -> weighted.
    orig='''    sampler=WeightedRandomSampler(torch.DoubleTensor(weights),num_samples=len(weights),replacement=True,generator=sampler_gen)
    dl=DataLoader(dataset,batch_size=cfg.batch_size,sampler=sampler,drop_last=True,'''
    new='''    from torch.utils.data import RandomSampler
    sampler_mode=getattr(cfg,"sampler_mode","weighted")
    if sampler_mode=="uniform":
        sampler=RandomSampler(dataset,replacement=False,generator=sampler_gen)
    elif sampler_mode=="weighted":
        sampler=WeightedRandomSampler(torch.DoubleTensor(weights),num_samples=len(weights),replacement=True,generator=sampler_gen)
    else:
        raise ValueError("Unsupported sampler mode: "+str(sampler_mode))
    dl=DataLoader(dataset,batch_size=cfg.batch_size,sampler=sampler,drop_last=True,'''
    if src.count(orig)!=1:raise RuntimeError("Sampler patch anchor mismatch")
    modified=src.replace(orig,new)
    field_anchor="    mismatch_lambda: float=0.\n    save_every: int=50"
    if modified.count(field_anchor)!=1:raise RuntimeError("Config sampler field anchor mismatch")
    modified=modified.replace(field_anchor,'    mismatch_lambda: float=0.\n    sampler_mode: str="weighted"\n    save_every: int=50')
    resume_anchor="'seam_lambda','disable_class','mismatch_lambda','model_name',"
    if modified.count(resume_anchor)!=1:raise RuntimeError("Resume compatibility anchor mismatch")
    modified=modified.replace(resume_anchor,"'seam_lambda','disable_class','mismatch_lambda','model_name','sampler_mode',")
    (PROJECT/"v5_engine_dataset_adapter.py").write_text(modified,encoding="utf-8")
    (PROJECT/"v5_model_source_sha256.txt").write_text(
       hashlib.sha256(src.encode()).hexdigest()+"  cell 6 of "+str(ORIGINAL)+"\n"+
       hashlib.sha256(modified.encode()).hexdigest()+"  local sampler-adapted copy\n",encoding="utf-8")
    runs=[]
    for name,encoder,lam,noclass,kind in VARIANTS:
        for seed in SEEDS:
            runs.append(dict(run_id=f"{name}_seed{seed}",dataset="augmented256",model=name,
               encoder=encoder,seam_lambda=lam if lam is not None else "",
               disable_class=noclass if noclass is not None else "",
               seed=seed,kind=kind,status="NOT_STARTED"))
    write_csv(PROJECT/"experiment_matrix_21_runs.csv",runs,list(runs[0]))
    conditions={
        "project":"EndekGAN-T2I experimental continuation on preprocessed images, NOT original paper results",
        "repo":"https://github.com/wiwik-instiki/EndekGAN-T2I",
        "dataset":str(DATA),"split_changed_from_previous_paper":True,
        "fixed_source_images":370,"class_count":4,"train_sources":296,
        "train_augmented_files":944,"val_files":37,"test_files":37,
        "seeds":list(SEEDS),"internal_variants":6,"external_variants":1,"planned_total_runs":21,
        "model_architecture":"Exact V5 generator/discriminator + text and class conditioning; 4px seam aware loss",
        "image_size":256,"latent_dim":128,"base_channels":32,"fm_lambda":5,
        "r1_gamma":5,"r1_every":16,"ema_decay":.999,"learning_rate_g":0.0002,"learning_rate_d":0.0001,
        "seam_lambda":0.5,"max_epochs":400,"model_selection":"validation KID; not test",
        "held_out_test":"37 original-only resized images, identical across all new variants",
        "sampler_policy":"original 296 sources weighted; offline-balanced 944 files uniform without replacement",
        "train_file_transform_note":"The old V5 model additionally uses online RandomHorizontalFlip and mild ColorJitter; must review geometric validity or disable consistently before full paper experiments.",
        "expert_evaluation":"After image generation (end of protocol), no fabricated assessments.",
        "original_prompt_ground_truth_warning":"Source is Endek image collection; metadata fields were deterministically assigned and not independently image-verified.",
        "rights":"No raw images or private checkpoints uploaded to public GitHub.",
        "gpu_preflight":"RTX 3070Ti laptop used 5760MiB of 8192MiB on initial check. Do not auto-start CUDA training; require >5GiB free."
    }
    (PROJECT/"experiment_protocol.json").write_text(json.dumps(conditions,ensure_ascii=False,indent=2),encoding="utf-8")
    rep={"status":"DATASET_ADAPTER_PREPARED","input_rows":len(data),
         "source_groups":len(seen),"split_counts":dict(collections.Counter(r["split"] for r in data)),
         "train_class_counts":dict(n),"source256_train":296,
         "augmented256_train":944,"test_identical_across_variants":True,
         "experiment_configs":len(runs),"external_configs_planned":3,
         "source_engine_sha":hashlib.sha256(src.encode()).hexdigest(),
         "model_sampler_patch":"weighted for source, uniform for augmented","data_md5_required_fields":True}
    (PROJECT/"stage1_preflight_report.json").write_text(json.dumps(rep,ensure_ascii=False,indent=2),encoding="utf-8")
    print(json.dumps(rep,ensure_ascii=False,indent=2))
if __name__=="__main__": main()
