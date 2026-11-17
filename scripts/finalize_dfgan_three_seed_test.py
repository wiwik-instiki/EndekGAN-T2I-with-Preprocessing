# -*- coding: utf-8 -*-
"""After BOTH actual Kaggle V2 heldout tests finish and pass SHA256, aggregate 3 real seeds.
No trained weights, prompts or private photos may enter the public report.
"""
from pathlib import Path
import json,hashlib,math
import numpy as np,pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

REPO=Path(__file__).resolve().parents[1]
ROOT=REPO.parent
FOLDER=ROOT/"FINAL_RESULTS"
TRAINING={
 42: dict(best_epoch=120,validation_KID=.1432943046092987),
 123:dict(best_epoch=80,validation_KID=.16386644542217255),
 2025:dict(best_epoch=140,validation_KID=.14624154567718506)}
MEASURES=["pooled_FID","pooled_KID","class_macro_FID","class_macro_KID","mean_seam_gap","LPIPS_diversity"]
results=[]
for seed,train in TRAINING.items():
    if seed==42:
        folder=FOLDER/"DFGAN_MiniLM_Seed42"/"HELDOUT_TEST"
        proof=folder/"TRANSFER_AUDIT.json"
        assert proof.exists() and json.loads(proof.read_text(encoding="utf-8")).get("verified_complete") is True
    else:
        folder=FOLDER/"DFGAN_MiniLM_Multiseed_V2"/f"seed{seed}"/"HELDOUT_TEST"
        proof=folder/"SHA256_TRANSFER_AUDIT.json"
        assert proof.exists(),f"Private Kaggle seed{seed} results not fully transferred"
        audit=json.loads(proof.read_text(encoding="utf-8"))
        assert audit["verified_complete"] is True
        for item in audit["inventory"]:
            p=folder/item["relative_path"]
            assert p.is_file(),(seed,p)
            h=hashlib.sha256()
            with p.open("rb") as stream:
                while True:
                    block=stream.read(4*1024*1024)
                    if not block:break
                    h.update(block)
            assert h.hexdigest()==item["sha256"]
    statuses=list(folder.rglob("EVALUATION_STATUS.json"))
    assert len(statuses)==1,(seed,len(statuses))
    j=json.loads(statuses[0].read_text(encoding="utf-8"))
    assert j["status"]=="EVALUATION_COMPLETE", (seed,j["status"])
    assert j["training_seed"]==seed and j["source_train"]==296 and j["source_test"]==37
    assert j["generated_images"]==256 and j["latent_seed"]==90731
    assert j["reference_method_TiPGAN_is_image_conditioned"] is True
    assert j["test_used_for_model_selection"] is False
    m=j["summary"]
    assert m["seed"]==seed and m["train_best_epoch"]==train["best_epoch"]
    assert m["heldout_real_n"]==37 and m["generated_n"]==256
    assert abs(float(j["validation_KID"])-train["validation_KID"])<1e-8
    vals={k:float(m[k]) for k in MEASURES}
    assert all(np.isfinite(v) for v in vals.values())
    results.append(dict(seed=seed,checkpoint_best_epoch=train["best_epoch"],
                        best_validation_KID=train["validation_KID"],real_n=37,fake_n=256,**vals))
assert len(results)==3
per=pd.DataFrame(results).sort_values("seed")
means={}
for metric in MEASURES:
    means[metric+"_mean"]=float(per[metric].mean())
    means[metric+"_sd"]=float(per[metric].std(ddof=1))
summary=pd.DataFrame([{"method":"DF-GAN-MiniLM adapted","n_independent_training_seeds":3,**means}])
OUT=REPO/"results";OUT.mkdir(exist_ok=True)
per.to_csv(OUT/"DFGAN_V2_real_heldout_three_seeds.csv",index=False,encoding="utf-8-sig")
summary.to_csv(OUT/"DFGAN_V2_real_heldout_mean_sd.csv",index=False,encoding="utf-8-sig")
baseline=pd.read_csv(OUT/"heldout_B_transformer_vs_seam_3seeds_per_seed.csv")
assert len(baseline)==6 and set(baseline.model)=={"transformer","seam_proposed"}
assert baseline.groupby("model").seed.nunique().to_dict()=={"transformer":3,"seam_proposed":3}
joined=[]
for seed in sorted(TRAINING):
    x=per.loc[per.seed==seed].iloc[0]
    joined.append(dict(seed=seed,model="DF-GAN-MiniLM adapted",
        pooled_FID=x.pooled_FID,pooled_KID=x.pooled_KID,
        class_macro_FID=x.class_macro_FID,class_macro_KID=x.class_macro_KID,
        mean_seam_gap=x.mean_seam_gap,LPIPS_diversity=x.LPIPS_diversity))
for x in baseline.itertuples(index=False):
    joined.append(dict(seed=x.seed,model=x.model,
        pooled_FID=x.pooled_FID,pooled_KID=x.pooled_KID,
        class_macro_FID=x.class_macro_FID,class_macro_KID=x.class_macro_KID,
        mean_seam_gap=x.mean_seam_gap,LPIPS_diversity=x.mean_LPIPS_diversity))
jdf=pd.DataFrame(joined).sort_values(["model","seed"]).reset_index(drop=True)
assert len(jdf)==9 and jdf.groupby("model").seed.nunique().eq(3).all()
jdf.to_csv(OUT/"DFGAN_EndekGAN_real_matched_three_seeds.csv",index=False,encoding="utf-8-sig")
grp=jdf.groupby("model",sort=False)[["pooled_FID","pooled_KID","mean_seam_gap","LPIPS_diversity"]].agg(["mean","std"])
model_order=["transformer","DF-GAN-MiniLM adapted","seam_proposed"]
fig,axes=plt.subplots(1,3,figsize=(13.8,4.2))
for ax,metric,title in zip(axes,("pooled_FID","pooled_KID","mean_seam_gap"),
                            ("Pooled FID ↓","Pooled KID ↓","Opposite-border gap ↓")):
    means=[float(grp.loc[m,(metric,"mean")]) for m in model_order]
    sd=[float(grp.loc[m,(metric,"std")]) for m in model_order]
    ax.bar(range(3),means,yerr=sd,capsize=4,alpha=.83)
    ax.set_xticks(range(3),["Transformer","DF-GAN–MiniLM","Seam Proposed"],
                  rotation=19,ha="right")
    ax.set_title(title+" (mean ± sample SD)")
    ax.grid(axis="y",alpha=.16)
fig.suptitle("Endek V2 held-out test: matched 3 training seeds, 37 real references per run")
fig.tight_layout()
fig.savefig(REPO/"figures/fig16_three_seed_dfgan_endekgan_heldout.png",dpi=180)
plt.close(fig)
lines=[
 "# Three-seed genuine held-out DF-GAN–MiniLM vs EndekGAN V2",
 "",
 "**All nine model × seed held-out tests are source-backed.** This report only draws on three completed DF-GAN-MiniLM Kaggle evaluation kernels and six completed internal EndekGAN V2 test runs.",
 "",
 "Real reference set: 37 independent held-out images in all runs; generated images: 256 per run; latent RNG seed: 90731. Because the two generators have different latent dimensions, numeric Gaussian inputs are not the same vectors even with the same RNG seed.",
 "",
 "| Model | Pooled FID ↓ | Pooled KID ↓ | Seam gap ↓ | LPIPS diversity ↑ |",
 "|---|---:|---:|---:|---:|"]
for model in model_order:
    values=[]
    for metric in ("pooled_FID","pooled_KID","mean_seam_gap","LPIPS_diversity"):
        values.append(f"{grp.loc[model,(metric,'mean')]:.5f} ± {grp.loc[model,(metric,'std')]:.5f}")
    lines.append(f"| {model} | "+" | ".join(values)+" |")
lines+=["",
"![Figure 16](../figures/fig16_three_seed_dfgan_endekgan_heldout.png)",
"",
"Figure 16. Three training seed means ± sample standard deviations, not statistical significance intervals.",
"",
"**Model identity**: DF-GAN-MiniLM uses the official DF-GAN generator/discriminator family with a frozen multilingual MiniLM384 text encoder and is **not canonical DAMSM-conditioned DF-GAN**.",
"**TiPGAN** is an image-conditioned texture synthesis model. It has not received text prompts; no fabricated text-to-image scores are attributed to TiPGAN.",
"**Validity constraints**: 37 real test images (5–15 per class) make FID/KID biased and unstable. Crop captions may be heuristically inherited, and no blinded expert has yet verified cultural authenticity or caption fidelity.",
"**Further evidence required**: motif-level qualitative annotation, text prompt intervention tests, and a genuinely converged TiPGAN train-image reference texture protocol reported separately.",
"",
"**Sources**:",
"- https://www.kaggle.com/code/agusperdana/endek-v2-dfgan-minilm-heldout-seed42-private",
"- https://www.kaggle.com/code/agusperdana/endek-v2-df-gan-heldout-seed-123-private",
"- https://www.kaggle.com/code/agusperdana/endek-v2-df-gan-heldout-seed-2025-private",
]
(REPO/"benchmarks/DFGAN_3SEED_V2_HELDOUT_FINAL.md").write_text("\n".join(lines)+"\n",encoding="utf-8")
README=REPO/"README.md"
s=README.read_text(encoding="utf-8")
new="\n## Genuine three-seed matched held-out test: DF-GAN–MiniLM vs EndekGAN V2\n\nThree distinct DF-GAN–MiniLM seed 42/123/2025 models and six internal EndekGAN seed 42/123/2025 tests now have genuine, source-grouped held-out results. See [validated three-seed mean ± sample SD results](benchmarks/DFGAN_3SEED_V2_HELDOUT_FINAL.md), [per-seed values](results/DFGAN_EndekGAN_real_matched_three_seeds.csv), and [Figure 16](figures/fig16_three_seed_dfgan_endekgan_heldout.png). FID/KID remain exploratory with just 37 independent real test images. TiPGAN is excluded as it is image-conditioned, not prompt-conditioned.\n\n"
if "## Genuine three-seed matched held-out test: DF-GAN–MiniLM vs EndekGAN V2" not in s:
    s=s.replace("## 7. Ethics, copyright and claims",new+"## 7. Ethics, copyright and claims")
    README.write_text(s,encoding="utf-8")
evidence=OUT/"EVIDENCE_STATUS.json"
obj=json.loads(evidence.read_text(encoding="utf-8"))
obj["DFGAN_V2_test"]="THREE_SEEDS_HELDOUT_EVALUATION_COMPLETE"
obj["DFGAN_V2_training_seeds_verified"]=[42,123,2025]
obj["DFGAN_V2_test_seeds_verified"]=[42,123,2025]
obj["external_three_seed_comparison"]="SOURCE_BACKED_EXPLORATORY_NO_SIGNIFICANCE_CLAIM"
obj["TiPGAN_V2_test"]="NOT_RUN"
evidence.write_text(json.dumps(obj,indent=2,ensure_ascii=False)+"\n",encoding="utf-8")
print("THREE_SEED_DFGAN_AND_ENDEKGAN_VERIFIED",summary.to_string(index=False))
print(jdf.to_string(index=False))
