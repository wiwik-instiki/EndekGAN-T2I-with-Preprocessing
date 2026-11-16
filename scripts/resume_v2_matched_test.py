# -*- coding: utf-8 -*-
"""Resume matched V2 held-out evaluation after interruption (no retraining)."""
import argparse,importlib.util,sys,json,datetime
from pathlib import Path
import numpy as np,pandas as pd
from PIL import Image
import torch
from torchvision.utils import save_image

REPO=Path(__file__).resolve().parents[1];ROOT=REPO.parent
ARM={"A":"full_image_augmented_A","B":"full_plus_crop_ablation_B"}
p=argparse.ArgumentParser()
p.add_argument("--arm",choices=["A","B"],default="B")
p.add_argument("--model",choices=["cat","hash_text","transformer","transformer_noclass","seam_proposed","seam_noclass"],default="transformer")
p.add_argument("--seed",type=int,choices=[42,123,2025],default=42)
p.add_argument("--execute",action="store_true")
args=p.parse_args()
if args.arm=="A" and args.model not in ["transformer","seam_proposed"]:raise ValueError("Not trained on Arm A")
out=ROOT/"runs_preprocessed"/ARM[args.arm]/f"{args.model}_seed{args.seed}"
state=json.loads((out/"run_state.json").read_text(encoding="utf-8"))
assert state["completed"] and state["test_used_for_model_selection"] is False
checkpoint=Path(state["best_checkpoint"])
assert checkpoint.is_file() and ROOT.resolve() in checkpoint.resolve().parents
eval_dir=out/"test_evaluation";gen_dir=eval_dir/"generated"
status=eval_dir/"evaluation_status.json"
if status.exists():
    if json.loads(status.read_text(encoding="utf-8")).get("status")=="EVALUATION_COMPLETE":
        print("ALREADY_COMPLETE",eval_dir);sys.exit(0)
    raise RuntimeError("Non-complete evaluation status exists. Inspect before proceeding")
print("PREFLIGHT",args.arm,args.model,args.seed,"checkpoint",checkpoint,"output",eval_dir,flush=True)
if not args.execute:sys.exit(0)
assert torch.cuda.is_available(),"CUDA required"
assert torch.cuda.mem_get_info()[0]>3*1024**3,"GPU memory insufficient"
spec=importlib.util.spec_from_file_location("endek_v2_engine",ROOT/"endek_v2_preprocessing_training_engine.py")
engine=importlib.util.module_from_spec(spec);sys.modules[spec.name]=engine;spec.loader.exec_module(engine)
splits=ROOT/"TRAINING_SPLITS_PREPROCESSED_V2"/ARM[args.arm]
test=engine.load_splits(splits)["test"];assert len(test)==37
G,encoder,cfg,dev=engine.generator_from_checkpoint(checkpoint,device="cuda",trusted=True)
rng=np.random.default_rng(90731);gen=torch.Generator(device=dev);gen.manual_seed(90731)
rows=[];old_count=0;new_count=0
for label in engine.CLASS_NAMES:
    sub=test[test.label==label];assert len(sub)>0
    sub=sub.iloc[rng.choice(len(sub),64,replace=True)].reset_index(drop=True)
    folder=gen_dir/label;folder.mkdir(parents=True,exist_ok=True)
    for start in range(0,64,4):
        group=sub.iloc[start:start+4]
        imgs=[folder/f"{label}_{start+j:04d}.png" for j in range(len(group))]
        z=torch.randn(len(group),cfg.latent_dim,generator=gen,device=dev)
        todo=[]
        for j,path in enumerate(imgs):
            if path.is_file():
                try:
                    with Image.open(path) as im:
                        im.verify()
                    with Image.open(path) as im:
                        assert im.size==(256,256),"Wrong image dimensions"
                except (OSError,AssertionError) as err:
                    # Preserve a partial/corrupt write in a private quarantine.
                    quarantine=ROOT/"REFERENCE_BASELINES"/"preflight_v2_private"/"corrupt_interrupted_images"/args.arm/args.model/str(args.seed)/label
                    quarantine.mkdir(parents=True,exist_ok=True)
                    backup=quarantine/path.name
                    assert not backup.exists(),"Refuse to overwrite quarantine file"
                    path.replace(backup)
                    print("QUARANTINED_INTERRUPTED_IMAGE",str(backup),"reason",str(err),flush=True)
                    todo.append(j)
                else:
                    old_count+=1
            else:
                todo.append(j)
        if todo:
            e=encoder.encode(group.prompt.tolist(),group.label.tolist()).to(dev)
            y=torch.tensor([engine.CLASS_TO_IDX[label]]*len(group),device=dev)
            with torch.inference_mode():pred=(G(z,e,y).cpu()+1).div(2).clamp(0,1)
            for j in todo:
                assert not imgs[j].exists(),"Refuse overwrite"
                save_image(pred[j],imgs[j]);new_count+=1
        for j,row in enumerate(group.itertuples()):
            rows.append(dict(generated_path=str(imgs[j].resolve()),label=label,
                prompt=row.prompt,source_filename=row.filename,latent_seed=90731,
                generated_index=start+j,checkpoint=str(checkpoint),encoder=cfg.encoder))
    print("GENERATED",label,"created",new_count,"retained",old_count,flush=True)
assert len(rows)==256
index=gen_dir/"generated_eval_index.csv"
if index.exists():
    original=pd.read_csv(index);updated=pd.DataFrame(rows)
    assert len(original)==len(updated)
    for col in ("label","prompt","generated_index"):
        assert original[col].astype(str).tolist()==updated[col].astype(str).tolist()
else:pd.DataFrame(rows).to_csv(index,index=False)
config=gen_dir/"generation_config.json"
if not config.exists():
    config.write_text(json.dumps(dict(n_per_class=64,seed=90731,checkpoint=str(checkpoint),
        encoder=cfg.encoder,source="held-out prompts"),indent=2),encoding="utf-8")
del G;torch.cuda.empty_cache()
print("IMAGES_READY",old_count,new_count,flush=True)
metrics=eval_dir/"metrics"
assert not (metrics.exists() and any(metrics.iterdir())),"Metrics partly exist, no overwrites"
scores=engine.evaluate(split_dir=splits,index_csv=index,out_dir=metrics,device="cuda",
                        size=256,lpips_pairs=40,do_feature_nn=False)
done=dict(status="EVALUATION_COMPLETE",checkpoint=str(checkpoint),generated_index=str(index),
          raw_test_n=37,evaluation_metrics=str(metrics/"E1_FID_KID_corrected.csv"),
          created_images=new_count,retained_images=old_count,
          caution="Exploratory FID/KID with 37 real test reference images.",
          completed_local=datetime.datetime.now().isoformat())
status.write_text(json.dumps(done,indent=2),encoding="utf-8")
print("TEST_EVALUATION_COMPLETE",json.dumps(done),flush=True)
print(scores.to_string(index=False),flush=True)
