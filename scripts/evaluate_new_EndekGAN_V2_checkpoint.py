# -*- coding: utf-8 -*-
"""Evaluate ONLY a new fully trained V2 checkpoint on the independent held-out test split.

Never selects checkpoints using test data. FID/KID are exploratory for small test size.
"""
from pathlib import Path
import argparse,json,sys,importlib.util,datetime,os
import pandas as pd
ROOT=Path(__file__).resolve().parent
ARMS={"A":"full_image_augmented_A","B":"full_plus_crop_ablation_B"}
def main():
    p=argparse.ArgumentParser()
    p.add_argument("--arm",choices=ARMS,default="B")
    p.add_argument("--model",choices=("cat","hash_text","transformer","transformer_noclass","seam_proposed","seam_noclass"),default="seam_proposed")
    p.add_argument("--seed",type=int,choices=(42,123,2025),default=42)
    p.add_argument("--device",choices=("cuda","cpu"),default="cuda")
    p.add_argument("--samples-per-class",type=int,default=64)
    p.add_argument("--execute",action="store_true")
    args=p.parse_args()
    runid=f"{args.model}_seed{args.seed}"
    run=ROOT/"runs_preprocessed"/ARMS[args.arm]/runid
    splits=ROOT/"TRAINING_SPLITS_PREPROCESSED_V2"/ARMS[args.arm]
    state=run/"run_state.json"
    plan={"run_dir":str(run),"expected_new_checkpoint":str(run/"checkpoints"),
          "test_split_dir":str(splits),"evaluation_prompts":"HELD_OUT_TEST_ONLY",
          "samples_per_class":args.samples_per_class,"status":"PREFLIGHT_ONLY"}
    if state.is_file():
        data=json.loads(state.read_text(encoding="utf-8"))
        plan["training_finished"]=bool(data.get("completed"))
        plan["trained_epochs"]=data.get("actual_epochs")
        plan["best_checkpoint"]=data.get("best_checkpoint")
    else:
        plan["training_finished"]=False
    print(json.dumps(plan,indent=2,ensure_ascii=False))
    if not args.execute:
        print("NO TEST EVALUATION RUN: add --execute only when a completed V2 run exists")
        return
    if not plan["training_finished"]:
        raise RuntimeError("No completed NEW V2 model; never evaluate old checkpoints as V2.")
    checkpoint=Path(plan["best_checkpoint"] or "") if plan.get("best_checkpoint") else Path("")
    if not checkpoint.is_file():
        raise FileNotFoundError("Missing NEW V2 selected validation checkpoint "+str(checkpoint))
    if ROOT.resolve() not in checkpoint.resolve().parents:
        raise RuntimeError("Rejected checkpoint outside V2 project")
    if args.samples_per_class<8:raise ValueError("Need at least 8 generated samples per class for exploratory evaluation")
    if not all(importlib.util.find_spec(m) for m in ("torch","torchvision","torchmetrics","torch_fidelity")):
        raise RuntimeError("Missing torch / torchvision / torchmetrics / torch_fidelity")
    spec=importlib.util.spec_from_file_location("engine",ROOT/"endek_v2_preprocessing_training_engine.py")
    engine=importlib.util.module_from_spec(spec);sys.modules[spec.name]=engine;spec.loader.exec_module(engine)
    out=run/"test_evaluation"
    if out.exists() and any(out.iterdir()):
        raise FileExistsError("Evaluation output exists, refusing accidental overwrite")
    out.mkdir(parents=True)
    index=engine.render_images(checkpoint,split_dir=splits,out_dir=out/"generated",
                               n_per_class=args.samples_per_class,seed=90731,device=args.device,
                               trusted=True,max_batch=4)
    score=engine.evaluate(split_dir=splits,index_csv=index,out_dir=out/"metrics",
                          device=args.device,size=256,lpips_pairs=40,do_feature_nn=False)
    (out/"evaluation_status.json").write_text(json.dumps(
        {"status":"EVALUATION_COMPLETE","checkpoint":str(checkpoint),
         "generated_index":str(index),"raw_test_n":37,
         "fid_kid_interpretation":"EXPLORATORY: only 37 real test images total, much fewer per class.",
         "evaluation_metrics":str(out/"metrics"/"E1_FID_KID_corrected.csv")},indent=2),encoding="utf-8")
    print(score.to_string(index=False))
if __name__=="__main__":
    main()
