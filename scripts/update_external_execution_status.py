# -*- coding: utf-8 -*-
"""Publish only aggregate/protocol status for external baselines. Keep all Endek inputs private."""
from pathlib import Path
import json,shutil
ROOT=Path(__file__).resolve().parents[1]
LOCAL=ROOT.parent
BASE=LOCAL/"REFERENCE_BASELINES"
ti=json.loads((LOCAL.parent/"TIP_V2_SHORTPATH_PILOT"/"PILOT_STATUS.json").read_text(encoding="utf-8"))
df=json.loads((BASE/"DFGAN_V2_ADAPTED_PRIVATE"/"PILOT_1EPOCH_seed42"/"run_state.json").read_text(encoding="utf-8"))
assert ti["status"]=="PILOT_COMPLETE_NOT_BENCHMARK" and ti["checkpoint_created"]
assert df["status"]=="PILOT_1_EPOCH_COMPLETED_NOT_BENCHMARK" and df["training_steps_completed"]==128
loc=ROOT/"scripts"/"dfgan_minilm_v2_kaggle_adapter.py"
shutil.copyfile(BASE/"kaggle_dfgan_v2_private_kernel"/"dfgan_minilm_v2.py",loc)
report={
"research":"TiPGAN primary CAD 2025 reference; DF-GAN secondary CVPR 2022 reference",
"v2_training_completed":24,
"internal_heldout_test_models_completed":6,
"internal_models":"Arm B Transformer vs Seam Proposed across seeds 42/123/2025",
"real_train_sources":296,"real_val_sources":37,"real_test_sources":37,
"TiPGAN":{
 "input":"single image-conditioned cloth texture (NOT text prompt)",
 "reference_code":"https://github.com/VickyHEHonghong/TiPGAN",
 "vgg_weights_obtained":True,
 "pilot_train_reference_class":"flora",
 "pilot_completed_iterations":30,
 "authors_default_iterations_per_reference":50000,
 "pilot_checkpoint":True,
 "status":"OFFICIAL_30_ITERATION_SMOKE_ONLY_NOT_COMPARABLE",
 "heldout_test_metrics":"NOT_COMPUTED"
},
"DFGAN_MiniLM_adapted":{
 "input":"Indonesian prompt embedding (384d MiniLM, same encoder as V2)",
 "reference_code":"https://github.com/tobran/DF-GAN",
 "official_models_used":"NetG/NetD/NetC",
 "local_forward_backward_smoke":"PASS",
 "pilot_training_seed":42,"pilot_completed_epochs":1,
 "pilot_completed_optimizer_step_pairs":128,
 "gpu_max_allocated_MiB":df["max_gpu_allocated_mib"],
 "status":"ADAPTED_ONE_EPOCH_PILOT_ONLY_NOT_CONVERGED",
 "validation_checkpoint":"NOT_SELECTED",
 "heldout_test_metrics":"NOT_COMPUTED",
 "Kaggle_larger_gpu_training":"PREPARED_NOT_VERIFIED_COMPLETED"
},
"three_way_performance_ranking":"NOT_POSSIBLE_YET",
"private_data_released_to_public_GitHub":False
}
p=ROOT/"results"/"EXTERNAL_EXECUTION_STATUS.json"
p.write_text(json.dumps(report,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
r=ROOT/"results"/"REFERENCE_BASELINE_PREFLIGHT.json"
state=json.loads(r.read_text(encoding="utf-8"))
state["TiPGAN"]["prerequisite_VGG_weights_available"]=True
state["TiPGAN"]["local_pilot"]="30_TRAIN_ITERATIONS_NO_TEST"
state["DF_GAN"]["local_adapted_pilot"]="ONE_EPOCH_128_UPDATES_NO_TEST"
r.write_text(json.dumps(state,indent=2,ensure_ascii=False)+"\n",encoding="utf-8")
readme=ROOT/"README.md"
s=readme.read_text(encoding="utf-8")
new="""
## External method execution: source-anchored but not yet benchmark-complete

A source-clean TiPGAN official-code pilot was successfully run for **30 optimizer iterations on one training-only flora image**, with its original VGG weights and a saved pilot checkpoint. The authors' default per-texture training budget is 50,000 iterations, so **this is not a converged benchmark**.

The official DF-GAN NetG/NetD/NetC architecture, explicitly adapted to use **exactly the frozen V2 MiniLM384 text features instead of canonical DAMSM**, passed a local gradient test and **one 128-step training epoch** on the actual 1,280-file Arm B data. This is **not a completed DF-GAN V2 comparison**.

[Execution audit](benchmarks/EXTERNAL_BASELINE_EXECUTION.md) ·
[Machine-readable statuses](results/EXTERNAL_EXECUTION_STATUS.json) ·
[Adapted DF-GAN reproducible Kaggle training script](scripts/dfgan_minilm_v2_kaggle_adapter.py).

**Do not place pilot-only values next to the held-out three-seed EndekGAN FID/KID table.** TiPGAN is image-reference-based and cannot be evaluated for prompt adherence it does not consume.

"""
if "## External method execution: source-anchored" not in s:
    assert "## 7. Ethics, copyright and claims" in s
    s=s.replace("## 7. Ethics, copyright and claims",new+"## 7. Ethics, copyright and claims")
readme.write_text(s,encoding="utf-8")
print("EXTERNAL_STATUS_SYNC_OK")
