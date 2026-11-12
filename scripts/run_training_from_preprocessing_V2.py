# -*- coding: utf-8 -*-
"""Actual EndekGAN-T2I V2 training launcher for fully preprocessed datasets.

Default: dataset validation and dry run only.
--execute: train a BRAND-NEW model from random initialization. Never reuse V5 weights.
Requires isolated D CUDA environment and sufficient idle VRAM. Separate test evaluation.
"""
from __future__ import annotations
from pathlib import Path
import os,sys,argparse,csv,json,datetime,subprocess,hashlib,importlib.util
import pandas as pd

ROOT=Path(__file__).resolve().parent
DATA=ROOT.parent/"ENDEK_PREPROCESSING_V2"
BASE=Path(r"D:\VSCode-MultiAgent-AI")
HF_MODEL_CACHE=Path(r"D:\DISERTASI_S3\WIWIK\DATASET_L2\EndekGAN\EndekGAN-T2I\MODEL_CACHE\huggingface")
os.environ.setdefault("HF_HOME",str(HF_MODEL_CACHE))
os.environ.setdefault("TORCH_HOME",str(BASE/"caches"/"torch"))
os.environ.setdefault("PIP_CACHE_DIR",str(BASE/"caches"/"pip"))
os.environ.setdefault("MPLCONFIGDIR",str(BASE/"caches"/"matplotlib"))
os.environ.setdefault("TMP",str(BASE/"caches"/"temp"))
os.environ.setdefault("TEMP",str(BASE/"caches"/"temp"))
os.environ.setdefault("JUPYTER_DATA_DIR",str(BASE/"jupyter"/"data"))
os.environ.setdefault("JUPYTER_RUNTIME_DIR",str(BASE/"jupyter"/"runtime"))
os.environ.setdefault("HF_HUB_OFFLINE","1")
os.environ.setdefault("TRANSFORMERS_OFFLINE","1")
MODEL_SPECS={
    "cat":("cat",0.0,False),
    "hash_text":("hash_text",0.0,False),
    "transformer":("transformer_text",0.0,False),
    "transformer_noclass":("transformer_text",0.0,True),
    "seam_proposed":("transformer_text",0.5,False),
    "seam_noclass":("transformer_text",0.5,True)
}
SPLIT_ARMS={"B":"full_plus_crop_ablation_B","A":"full_image_augmented_A"}
CLASSES=["flora","fauna","dekoratif","geometris"]
KERNELEXE=BASE/"envs"/"endekgan-t2i-v2-py311"/"Scripts"/"python.exe"

def loadcsv(p):
    with p.open(encoding="utf-8-sig",newline="") as f:return list(csv.DictReader(f))

def gpu_memory():
    try:
        stdout=subprocess.check_output(
            ["nvidia-smi","--query-gpu=name,memory.total,memory.used,memory.free",
             "--format=csv,noheader,nounits"],text=True,timeout=9
        ).strip().splitlines()[0]
        chunks=[x.strip() for x in stdout.split(",")]
        return {"name":chunks[0],"total_mib":int(chunks[1]),"used_mib":int(chunks[2]),"free_mib":int(chunks[3])}
    except Exception as e:
        return {"error":repr(e),"free_mib":0}

def validate_splits(arm):
    folder=ROOT/"TRAINING_SPLITS_PREPROCESSED_V2"/SPLIT_ARMS[arm]
    expected={"train":1280 if arm=="B" else 944,"val":37,"test":37}
    frames={k:loadcsv(folder/(k+"_split.csv")) for k in ("train","val","test")}
    for split,rows in frames.items():
        if len(rows)!=expected[split]:raise RuntimeError(f"Bad {arm} {split}: {len(rows)} vs {expected[split]}")
        for row in rows:
            required=["image_path","label","prompt","class_idx","source_sha256","source_md5","md5","split"]
            if any(not str(row.get(k,"")).strip() for k in required):
                raise ValueError("Missing mandatory T2I field in "+str(row.get("filename")))
            if row["split"]!=split:raise ValueError("Wrong split membership")
            if row["label"] not in CLASSES or int(row["class_idx"])!=CLASSES.index(row["label"]):
                raise ValueError("Class mismatch in "+str(row.get("filename")))
            if not Path(row["image_path"]).is_file():raise FileNotFoundError(row["image_path"])
    keys={part:{r["source_sha256"] for r in rows} for part,rows in frames.items()}
    for left,right in (("train","val"),("train","test"),("val","test")):
        if keys[left]&keys[right]:raise ValueError(f"Source-group LEAKAGE between {left} {right}")
    training=frames["train"]
    class_counts={c:sum(r["label"]==c for r in training) for c in CLASSES}
    if len(set(class_counts.values()))!=1:raise ValueError("Dataset is NOT class balanced")
    crop_train=[r for r in training if str(r.get("is_crop","")).lower() in ("true","1","yes")]
    if arm=="B" and len(crop_train)!=336:raise ValueError("Expected 336 V2 crop candidates")
    if arm=="A" and crop_train:raise ValueError("Unexpected crops in control A")
    for part in ("val","test"):
        if any(str(r.get("is_crop","")).lower() in ("true","1","yes") for r in frames[part]):
            raise ValueError("Crop appeared in validation/test")
    return {"split_dir":str(folder),"files":{k:len(v) for k,v in frames.items()},
            "class_balance":class_counts,"train_crop_count":len(crop_train),
            "same_source_split_integrity":True,
            "crop_prompt_warning":"All 336 crops inherit whole-image prompts; semantic correctness not proven" if arm=="B" else None,
            "val_source_fingerprint":hashlib.sha256(("\n".join(sorted(keys["val"]))).encode()).hexdigest(),
            "test_source_fingerprint":hashlib.sha256(("\n".join(sorted(keys["test"]))).encode()).hexdigest()}

def status_modules():
    modules=("torch","torchvision","transformers","sklearn","torchmetrics","torch_fidelity","sentencepiece")
    return {module:bool(importlib.util.find_spec(module)) for module in modules}

def make_parser():
    p=argparse.ArgumentParser(description="Genuine training from preprocessed EndekGAN-T2I V2 data")
    p.add_argument("--arm",choices=("A","B"),default="B")
    p.add_argument("--model",choices=tuple(MODEL_SPECS),default="seam_proposed")
    p.add_argument("--seed",type=int,choices=(42,123,2025),default=42)
    p.add_argument("--device",choices=("cuda","cpu"),default="cuda")
    p.add_argument("--batch-size",type=int,default=4)
    p.add_argument("--samples-per-epoch",type=int,default=512)
    p.add_argument("--epochs",type=int,default=400)
    p.add_argument("--min-free-vram",type=int,default=6400)
    p.add_argument("--execute",action="store_true",help="ACTUALLY train from scratch; without this flag only preview")
    p.add_argument("--resume",action="store_true",help="Resume ONLY a locally generated V2 last checkpoint")
    p.add_argument("--smoke",action="store_true",help="Run 64px one-epoch CPU/CUDA smoke; not paper result")
    return p

def main():
    cfgargs=make_parser().parse_args()
    if cfgargs.batch_size<2:raise ValueError("Batch size needs >=2")
    if cfgargs.samples_per_epoch<cfgargs.batch_size or cfgargs.samples_per_epoch%cfgargs.batch_size:
        raise ValueError("Use a samples-per-epoch value divisible by batch size")
    report=validate_splits(cfgargs.arm)
    # Cross-arm val/test comparison must use identical real sources.
    other=validate_splits("A" if cfgargs.arm=="B" else "B")
    for fingerprint in ("val_source_fingerprint","test_source_fingerprint"):
        if report[fingerprint]!=other[fingerprint]:
            raise RuntimeError("A and B held-out sources differ; stop experiment")
    mode,seam,no_class=MODEL_SPECS[cfgargs.model]
    runid=f"{cfgargs.model}_seed{cfgargs.seed}"
    output=(ROOT/"runs_preprocessed"/SPLIT_ARMS[cfgargs.arm]/runid)
    if cfgargs.smoke:
        output=ROOT/"SMOKE_CUDA_CPU"/f"{SPLIT_ARMS[cfgargs.arm]}_{runid}"
    modules=status_modules()
    plan={
        "status":"PREFLIGHT_ONLY" if not cfgargs.execute else "REQUESTED",
        "dataset_arm":cfgargs.arm,"split_dir":report["split_dir"],"source_dataset":SPLIT_ARMS[cfgargs.arm],
        "run_id":runid,"model":cfgargs.model,"encoder":mode,"seam_lambda":seam,
        "disable_class":no_class,"seed":cfgargs.seed,
        "image_size":64 if cfgargs.smoke else 256,
        "batch_size":cfgargs.batch_size,"optimizer_updates_per_epoch":cfgargs.samples_per_epoch//cfgargs.batch_size,
        "train_samples_per_epoch":cfgargs.samples_per_epoch,"epochs":1 if cfgargs.smoke else cfgargs.epochs,
        "total_planned_optimizer_updates":(cfgargs.samples_per_epoch//cfgargs.batch_size)*(1 if cfgargs.smoke else cfgargs.epochs),
        "online_augmentation":False,
        "training_method":"New weights from random initialization; weighted class-balanced fixed number of samples per epoch",
        "validation_model_selection":"KID on held-out validation, no test for checkpoint selection" if not cfgargs.smoke else "disabled for SMOKE",
        "testing":"Only after new V2 checkpoint exists",
        "checkpoint_source":"NEW_MODEL_RANDOM_INIT",
        "output_dir":str(output),"cuda_free_memory":gpu_memory(),"modules_installed":modules,
        "dataset_integrity":report,
        "data_source_limitation":"External Mendeley/Roboflow images not integrated; crop prompt semantic accuracy unverified."
    }
    print(json.dumps(plan,indent=2,ensure_ascii=False),flush=True)
    if not cfgargs.execute:
        print("PREFLIGHT PASSED. No GAN training was started; add --execute after CUDA prerequisites.",flush=True)
        return
    if not str(sys.executable).lower().startswith(str(BASE).lower()):
        raise RuntimeError("Training must use the isolated Python environment on D:\\VSCode-MultiAgent-AI")
    needed=("torch","torchvision","sklearn","torchmetrics","torch_fidelity","transformers")
    missing=[x for x in needed if not modules[x]]
    if missing and not cfgargs.smoke:
        raise RuntimeError("Cannot train: missing modules "+str(missing)+"; install packages into D env")
    if cfgargs.smoke:
        missing=[x for x in ("torch","torchvision","sklearn") if not modules[x]]
        if missing:raise RuntimeError("Smoke prerequisites missing: "+str(missing))
    import torch
    if cfgargs.device=="cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA PyTorch not active. No training started.")
        if plan["cuda_free_memory"]["free_mib"]<cfgargs.min_free_vram:
            raise RuntimeError("Other workloads occupy GPU; free VRAM below safety threshold. No training started.")
    elif not cfgargs.smoke:
        raise RuntimeError("Full 256x256 GAN training with CPU is deliberately blocked; select CUDA.")
    if cfgargs.smoke:
        if not cfgargs.execute:
            print("Smoke preview only.")
            return
    if output.exists() and any(output.iterdir()) and not cfgargs.resume:
        raise FileExistsError("Run already contains files. To resume local V2 checkpoint use --resume.")
    if cfgargs.resume and not (output/"checkpoints"/f"checkpoint_last_{mode}.pth").exists():
        raise FileNotFoundError("No locally generated V2 checkpoint to resume")
    output.mkdir(parents=True,exist_ok=True)
    plan["status"]="STARTED"
    (output/"experiment_plan.json").write_text(json.dumps(plan,indent=2,ensure_ascii=False),encoding="utf-8")
    spec=importlib.util.spec_from_file_location("engine_v2",ROOT/"endek_v2_preprocessing_training_engine.py")
    engine=importlib.util.module_from_spec(spec)
    sys.modules[spec.name]=engine
    spec.loader.exec_module(engine)
    smoke=cfgargs.smoke
    config=engine.Config(
        split_dir=report["split_dir"],out_dir=str(output),
        encoder=mode,seed=cfgargs.seed,epochs=1 if smoke else cfgargs.epochs,
        batch_size=cfgargs.batch_size,image_size=64 if smoke else 256,
        latent_dim=128,base_ch=32,lr_g=2e-4,lr_d=1e-4,
        fm_lambda=5.0,r1_gamma=0.0 if smoke else 5.0,
        r1_every=16,ema_decay=.999,seam_lambda=seam,
        disable_class=no_class,mismatch_lambda=0.0,device=cfgargs.device,
        sampler_mode="balanced_fixed_steps",train_samples_per_epoch=cfgargs.samples_per_epoch,
        online_augment=False,num_workers=0,save_every=25,sample_every=10,
        early_stop_enabled=not smoke,early_stop_min_epoch=200,
        early_stop_every=10,early_stop_patience_epochs=80,
        early_stop_min_delta=0.002,early_stop_fixed_seed=90731,
        early_stop_kid_subsets=10,early_stop_kid_subset_size=25
    )
    local_resume=output/"checkpoints"/f"checkpoint_last_{mode}.pth" if cfgargs.resume else None
    try:
        result=engine.train(config,resume=local_resume,trusted_checkpoint=cfgargs.resume)
        state=result/"run_state.json"
        if not state.is_file():raise RuntimeError("Training ended without run_state.json")
        finished=json.loads(state.read_text(encoding="utf-8"))
        plan["status"]="COMPLETED" if finished.get("completed") else "INCOMPLETE"
        plan["trained_epochs"]=finished.get("actual_epochs")
        plan["trained_global_steps"]=finished.get("global_steps")
        plan["best_checkpoint"]=finished.get("best_checkpoint")
    except Exception as err:
        plan["status"]="FAILED";plan["error"]=repr(err)
        raise
    finally:
        (output/"experiment_plan.json").write_text(json.dumps(plan,indent=2,ensure_ascii=False),encoding="utf-8")
    print("TRAINING_RESULT="+json.dumps(plan,indent=2,ensure_ascii=False))

if __name__=="__main__":
    main()
