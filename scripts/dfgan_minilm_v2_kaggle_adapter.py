# -*- coding: utf-8 -*-
"""PRIVATE adapted DF-GAN–MiniLM V2 experiment. Official NetG/NetD/NetC architecture,
384D multilingual MiniLM conditioning; SAME Endek V2 source-grouped train/val/test.
Do not mislabel as canonical DAMSM DF-GAN. No public Endek image release.
"""
from __future__ import annotations
import os,sys,random,json,zipfile,hashlib,shutil,time
from pathlib import Path
import numpy as np,pandas as pd
import torch,torch.nn.functional as F
from torch.utils.data import Dataset,DataLoader,WeightedRandomSampler
from PIL import Image
from torchvision.transforms.functional import to_tensor

WD=Path("/kaggle/working")
WD.mkdir(parents=True,exist_ok=True)
OUT=WD/"dfgan_minilm_v2_seed42"
OUT.mkdir(exist_ok=True)
PRIVATE=WD/"endek_private_data"
METADATA=dict(
 method="ADAPTED DF-GAN–MiniLM (NOT official DAMSM)",
 baseline_reference="Tao et al., CVPR 2022",
 official_source_commit="78f14fd72cb70172d2067791c9df386f03f5124b",
 source_split="B: train 1280 derived from 296 independent train sources; val=37; test=37",
 inputs="Private zip + frozen EndekGAN V2 multilingual MiniLM 384-D vectors",
 published_min_gpu_gib=12,
 same_vs_V2={"resolution":256,"optimizer_updates_per_epoch":128,"batch_size":4,
  "train_sample_exposure_per_epoch":512,"seeds":[42],"selection":"validation KID only"},
 caveats="Private cultural motif data; no crop-prompt semantic verification; cannot claim model superiority before held-out tests",
)
(OUT/"PROTOCOL.json").write_text(json.dumps(METADATA,indent=2),encoding="utf-8")
assert torch.cuda.is_available(),"Kaggle CUDA accelerator not attached; cannot run DF-GAN"
ram=torch.cuda.get_device_properties(0).total_memory/1024**3
assert ram>=12,"Official DF-GAN suggests >=12GB; kernel must have >=12GB"

# Source architecture file and embeddings are private supplementary kernel files, not fetched from an unrelated model.
HERE=Path(__file__).resolve().parent
def resolve_aux(filename):
    for base in (HERE,WD,Path.cwd(),Path("/kaggle/input")):
        direct=base/filename
        if direct.is_file():return direct
        for fp in base.glob("**/"+filename):
            if fp.is_file():return fp
    raise FileNotFoundError("Required supplementary file absent from kernel: "+filename)
ARCH=resolve_aux("DFGAN_official_GAN.py")
sys.path.insert(0,str(ARCH.parent))
from DFGAN_official_GAN import NetG,NetD,NetC
EMB=resolve_aux("minilm_v2_source_aligned_embeddings.npz")

datafiles=list(Path("/kaggle/input").glob("**/endek_v2_private_data.zip"))
if not datafiles:raise FileNotFoundError("Private V2 dataset not attached; no synthetic fallback")
file=datafiles[0]
assert file.stat().st_size>30_000_000
if not (PRIVATE/"train_manifest.csv").exists():
    PRIVATE.mkdir(exist_ok=True)
    with zipfile.ZipFile(file) as z:
        assert z.testzip() is None
        z.extractall(PRIVATE)
d={s:pd.read_csv(PRIVATE/f"{s}_manifest.csv") for s in ("train","val","test")}
assert {k:len(v) for k,v in d.items()}=={"train":1280,"val":37,"test":37}
assert d["train"].source_group.nunique()==296
for a in d:
    for b in d:
        if a!=b:
            assert not (set(d[a].source_group)&set(d[b].source_group))
nps=np.load(EMB)
emb={s:torch.tensor(nps[s],dtype=torch.float32) for s in ("train","val","test")}
for s in d: assert emb[s].shape==(len(d[s]),384)
assert d["train"].label.value_counts().to_dict()=={
"flora":320,"fauna":320,"dekoratif":320,"geometris":320}

# Fail fast rather than discovering missing Inception dependencies after training.
import importlib.util,subprocess
missing=[]
if importlib.util.find_spec("torchmetrics") is None:missing.append("torchmetrics[image]")
if importlib.util.find_spec("torch_fidelity") is None:missing.append("torch-fidelity")
if missing:
    print("INSTALL_VALIDATION_METRICS",missing,flush=True)
    subprocess.run([sys.executable,"-m","pip","install","--quiet",*missing],check=True)
from torchmetrics.image.kid import KernelInceptionDistance

class EndekPairs(Dataset):
    def __init__(self,part):
        self.part=part;self.df=d[part];self.emb=emb[part]
    def __len__(self):return len(self.df)
    def __getitem__(self,i):
        p=PRIVATE/self.df.iloc[i]["file"]
        with Image.open(p) as im:
            x=to_tensor(im.convert("RGB"))*2-1
        return x,self.emb[i],int(self.df.iloc[i]["class_idx"])

def get_val_kid(G,device,seed_for_val=90731):
    from torchmetrics.image.kid import KernelInceptionDistance
    val=EndekPairs("val")
    real_loader=DataLoader(val,batch_size=4,num_workers=0,shuffle=False)
    kid=KernelInceptionDistance(feature=2048,normalize=False,subset_size=10,subsets=20).to(device)
    g=torch.Generator(device="cpu");g.manual_seed(seed_for_val)
    z_all=torch.randn(len(val),100,generator=g)
    G.eval();off=0
    with torch.inference_mode():
        for real,text,_ in real_loader:
            num=len(real)
            fake=G(z_all[off:off+num].to(device),text.to(device))
            off+=num
            real_u8=((real+1)*127.5).round().clamp(0,255).to(torch.uint8).to(device)
            fake_u8=((fake+1)*127.5).round().clamp(0,255).to(torch.uint8)
            kid.update(real_u8,real=True)
            kid.update(fake_u8,real=False)
    score=float(kid.compute()[0].item())
    G.train();return score

SEED=42;MAX_EPOCHS=400;SAMPLES_PER_EPOCH=512;BS=4
VAL_EVERY=20;MIN_EPOCH=200;PATIENCE=80;MIN_DELTA=.002
DEVICE=torch.device("cuda:0")
random.seed(SEED);np.random.seed(SEED);torch.manual_seed(SEED);torch.cuda.manual_seed_all(SEED)
G=NetG(32,100,384,256,3).to(DEVICE)
D=NetD(32,256,3).to(DEVICE)
C=NetC(32,384).to(DEVICE)
gopt=torch.optim.Adam(G.parameters(),lr=1e-4,betas=(0.,.9))
dopt=torch.optim.Adam(list(D.parameters())+list(C.parameters()),lr=4e-4,betas=(0.,.9))
train=EndekPairs("train")
counts=np.bincount(d["train"].class_idx.to_numpy(),minlength=4)
weight=(1.0/counts[d["train"].class_idx.to_numpy()]).tolist()
start_epoch=1
best_kid=float("inf");best_epoch=0;history=[]
last=OUT/"last_resume.pth"
best=OUT/"best_validation_KID.pth"
if last.is_file():
    state=torch.load(last,map_location="cpu",weights_only=False)
    assert state["seed"]==SEED and state["protocol"]["SAMPLES_PER_EPOCH"]==512
    G.load_state_dict(state["G"]);D.load_state_dict(state["D"]);C.load_state_dict(state["C"])
    gopt.load_state_dict(state["gopt"]);dopt.load_state_dict(state["dopt"])
    start_epoch=state["epoch"]+1;best_kid=state["best_kid"];best_epoch=state["best_epoch"];history=state["history"]
    print("RESUME",start_epoch,flush=True)

for epoch in range(start_epoch,MAX_EPOCHS+1):
    sampler=WeightedRandomSampler(weight,SAMPLES_PER_EPOCH,replacement=True,
        generator=torch.Generator().manual_seed(SEED*100000+epoch))
    loader=DataLoader(train,batch_size=BS,num_workers=0,sampler=sampler,drop_last=True)
    assert len(loader)==128
    gs=[];ds=[]
    G.train();D.train();C.train()
    for real,txt,_ in loader:
        real=real.to(DEVICE).detach().requires_grad_(True)
        txt=txt.to(DEVICE).detach().requires_grad_(True)
        feat_real=D(real)
        real_pred=C(feat_real,txt)
        mismatched=C(feat_real.roll(1,0),txt)
        z=torch.randn(len(real),100,device=DEVICE)
        fake=G(z,txt.detach())
        fake_pred=C(D(fake.detach()),txt)
        g_real=F.relu(1-real_pred).mean()
        g_mismatch=F.relu(1+mismatched).mean()
        g_fake=F.relu(1+fake_pred).mean()
        grads=torch.autograd.grad(real_pred,[real,txt],
            grad_outputs=torch.ones_like(real_pred),retain_graph=True,create_graph=True)
        gradnorm=torch.cat([grads[0].flatten(1),grads[1].flatten(1)],dim=1).norm(dim=1)
        magp=2.0*gradnorm.pow(6).mean()
        dloss=g_real+0.5*(g_mismatch+g_fake)+magp
        dopt.zero_grad(set_to_none=True);dloss.backward();dopt.step()
        for param in list(D.parameters())+list(C.parameters()):
            param.requires_grad_(False)
        gloss=-C(D(fake),txt.detach()).mean()
        gopt.zero_grad(set_to_none=True);gloss.backward();gopt.step()
        for param in list(D.parameters())+list(C.parameters()):
            param.requires_grad_(True)
        gs.append(float(gloss.detach()));ds.append(float(dloss.detach()))
    val_kid=None
    if epoch%VAL_EVERY==0 or epoch==MAX_EPOCHS:
        val_kid=get_val_kid(G,DEVICE)
        if np.isfinite(val_kid) and val_kid+MIN_DELTA<best_kid:
            best_kid=val_kid;best_epoch=epoch
            torch.save(dict(G=G.state_dict(),epoch=epoch,seed=SEED,best_val_KID=val_kid,
                text="MiniLM384 frozen V2 adapter, NOT canonical DAMSM"),best)
    history.append(dict(epoch=epoch,generator_loss=np.mean(gs),
         discriminator_loss=np.mean(ds),validation_KID=val_kid))
    # Keep only one updated resume checkpoint; per-epoch log is a regular table.
    state=dict(G=G.state_dict(),D=D.state_dict(),C=C.state_dict(),
         gopt=gopt.state_dict(),dopt=dopt.state_dict(),
         seed=SEED,epoch=epoch,best_kid=best_kid,best_epoch=best_epoch,
         history=history,protocol=dict(SAMPLES_PER_EPOCH=SAMPLES_PER_EPOCH,BS=BS,IMAGE_SIZE=256))
    torch.save(state,last)
    pd.DataFrame(history).to_csv(OUT/"training_log.csv",index=False)
    print("EPOCH",epoch,"G",round(history[-1]["generator_loss"],5),
          "D",round(history[-1]["discriminator_loss"],5),
          "val_kid",val_kid,"best",best_epoch,flush=True)
    if epoch>=MIN_EPOCH and best_epoch and epoch-best_epoch>=PATIENCE:
        print("EARLY_STOP",epoch,"BEST",best_epoch,flush=True)
        break
assert best.is_file(), "No validation-selected checkpoint; cannot claim trained"
record=dict(status="TRAINING_COMPLETE_NO_TEST",adaptation="DF-GAN original NetG/NetD/NetC + MiniLM384",
 seed=SEED,train_rows=1280,independent_train_sources=296,
 training_steps=(epoch*128),best_epoch=best_epoch,best_validation_KID=best_kid,
 last_epoch=epoch,test_used_for_selection=False,
 caveat="Do not directly equate with official DAMSM DF-GAN; held-out test pending")
(OUT/"TRAINING_COMPLETE.json").write_text(json.dumps(record,indent=2),encoding="utf-8")
print("TRAINING_COMPLETE",json.dumps(record,indent=2),flush=True)
