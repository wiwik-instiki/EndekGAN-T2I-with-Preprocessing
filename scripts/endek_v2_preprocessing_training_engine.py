"""EndekGAN-T2I v2 experimental research runner.

This extends (without overwriting) the author's original EndekGAN notebook.
Important: automatically generated design attributes are NOT expert validated.
No result is fabricated: outputs are produced only if corresponding commands run.
"""
from __future__ import annotations
import argparse
import copy
import csv
import hashlib
import json
import math
import os
import random
import re
import shutil
import time
import warnings
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image
from sklearn.model_selection import train_test_split

CLASS_NAMES = ("flora", "fauna", "dekoratif", "geometris")
CLASS_TO_IDX = {k: i for i, k in enumerate(CLASS_NAMES)}
EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".tif", ".tiff", ".bmp"}
TRANSFORMER_NAME = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"

# The wording and deterministic design knowledge below follows the ORIGINAL notebook.
KNOWLEDGE = {
    'technique': {
        'stilasi': 'teknik stilasi, yaitu penggayaan bentuk asli menjadi ornamen dekoratif dengan tetap mempertahankan ciri utama objek sumber',
        'deformasi': 'teknik deformasi, yaitu penyederhanaan struktur dan proporsi bentuk asli menjadi bentuk baru yang lebih dekoratif',
        'distorsi': 'teknik distorsi, yaitu perubahan bentuk dengan penekanan atau pelebaran proporsi untuk menghasilkan karakter visual yang lebih ekspresif'
    },
    'layout_pattern': ['pola tunggal','pola acak terkontrol','pola ulang himpunan','pola ulang menyeluruh'],
    'composition': ['proporsi seimbang','keseimbangan visual','irama repetitif','kontras bentuk dan warna','klimaks visual yang jelas','kesatuan ornamen'],
    'weavability_note': ['layak diterapkan pada proses ikat tenun Endek','mempertimbangkan jarak antar ikatan minimal 0,5 cm agar warna celupan dapat masuk','motif tetap terbaca pada skala benang','struktur ornamen tidak terlalu rapat sehingga dapat diendekkan','mempertahankan karakter tekstur tenun ikat tradisional'],
    'color_style': ['warna merah dan emas yang kontras','warna hijau dan kuning yang harmonis','warna biru, hitam, dan putih dengan kontras kuat','warna coklat, krem, dan emas bernuansa tradisional','warna cerah khas kain upacara Bali'],
    'cultural_context': ['digunakan dalam konteks budaya Bali','menjaga kesakralan dan kepantasan motif','memadukan unsur lama dan unsur baru secara kreatif','mengutamakan nilai estetika, fungsi, dan identitas lokal Endek Bali']
}
SOURCES = {
 'flora':['bunga kamboja sebagai simbol kesucian','daun dan sulur tanaman tropis Bali','bunga teratai sebagai simbol keindahan dan keseimbangan','motif bunga lokal dengan kelopak berulang'],
 'fauna':['kupu-kupu sebagai simbol keindahan dan transformasi','burung sebagai simbol harmoni alam','ikan atau hewan air sebagai inspirasi alam tropis','bentuk fauna yang distilasi menjadi ornamen dekoratif'],
 'dekoratif':['ornamen tradisional Bali dengan unsur hias klasik','paduan garis, titik, sulur, dan ornamen pinggir','kombinasi unsur lama dan unsur baru dalam ragam hias Endek','hiasan dekoratif dengan komposisi simetris'],
 'geometris':['susunan garis dan bidang berulang','pola segitiga, segi empat, dan bentuk simetris','komposisi geometris ritmis dengan pengulangan teratur','ornamen geometris berbasis struktur ikat tradisional']
}

def md5_file(path):
    h = hashlib.md5()
    with open(path,'rb') as f:
        for b in iter(lambda:f.read(2**20),b''):h.update(b)
    return h.hexdigest()

def choose(items, key):
    return items[int(hashlib.md5(str(key).encode('utf-8')).hexdigest(),16) % len(items)]

def legacy_prompt(label, filename):
    """Reproduce author's hashing scheme; NOT ground-truth semantic labels."""
    label = str(label).lower().strip()
    key = f"{label}_{filename}"
    src = choose(SOURCES[label],key+'_source')
    tech = choose(list(KNOWLEDGE['technique']),key+'_technique')
    layout = choose(KNOWLEDGE['layout_pattern'],key+'_layout')
    comp1 = choose(KNOWLEDGE['composition'],key+'_comp1')
    comp2 = choose(KNOWLEDGE['composition'],key+'_comp2')
    color = choose(KNOWLEDGE['color_style'],key+'_color')
    weaving = choose(KNOWLEDGE['weavability_note'],key+'_weave')
    cultural = choose(KNOWLEDGE['cultural_context'],key+'_cultural')
    title=f'Motif Endek {label.capitalize()} berbasis {src}'
    prompt = (f'{title}. Desain kain Endek Bali bertema {label} yang terinspirasi dari {src}. '
              f'Motif dibuat menggunakan {KNOWLEDGE["technique"][tech]}. '
              f'Tata letak menggunakan {layout}, dengan {comp1}, {comp2}, dan kesatuan ornamen. '
              f'Pewarnaan memakai {color}. Desain harus {weaving}, tetap mempertahankan tekstur tenun ikat tradisional, dan {cultural}.')
    return dict(prompt=prompt,title=title,motif_source=src,technique=tech,layout_pattern=layout,
                composition=f'{comp1}; {comp2}; kesatuan ornamen',color_style=color,
                weavability_note=weaving,cultural_context=cultural)

def dump_json(path, obj):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    with open(path,'w',encoding='utf-8') as f:json.dump(obj,f,ensure_ascii=False,indent=2,default=str)

def dataset_prepare(dataset_root, out_dir, source_metadata=None, split_seed=3407, allow_invalid=False):
    """Create reproducible common stratified split. No implicit non-stratified fallback."""
    root=Path(dataset_root).expanduser().resolve()
    out=Path(out_dir).expanduser().resolve();out.mkdir(parents=True,exist_ok=True)
    if not root.is_dir(): raise FileNotFoundError(f'Dataset directory missing: {root}')
    records=[];broken=[]
    for label in CLASS_NAMES:
        folder=root/label
        if not folder.is_dir(): raise FileNotFoundError(f'Missing class directory: {folder}')
        for p in sorted(folder.rglob('*')):
            if p.suffix.lower() not in EXTENSIONS or not p.is_file():continue
            try:
                with Image.open(p) as im: im.verify()
                with Image.open(p) as im: im.convert('RGB').load()
                hashval=md5_file(p)
                records.append(dict(filename=p.relative_to(root).as_posix(),label=label,image_path=str(p),md5=hashval))
            except Exception as exc:broken.append(dict(path=str(p),error=str(exc)))
    pd.DataFrame(broken,columns=['path','error']).to_csv(out/'invalid_images.csv',index=False)
    if broken and not allow_invalid: raise ValueError(f'{len(broken)} invalid image(s). See {out / "invalid_images.csv"}. Fix before continuing, or explicitly --allow-invalid.')
    if not records:raise ValueError('No usable images in class folders.')
    raw=pd.DataFrame(records)
    raw.loc[raw.duplicated('md5',keep=False)].to_csv(out/'exact_duplicates.csv',index=False)
    # Identical bytes with conflicting labels cannot be silently deduplicated.
    conflicting_hashes=raw.groupby('md5')['label'].nunique()
    conflicting_hashes=conflicting_hashes[conflicting_hashes>1].index
    if len(conflicting_hashes):
        raw[raw.md5.isin(conflicting_hashes)].to_csv(out/'DUPLICATE_LABEL_CONFLICTS.csv',index=False)
        raise ValueError('Label conflicts for duplicate image hashes; see DUPLICATE_LABEL_CONFLICTS.csv')
    df=raw.drop_duplicates('md5',keep='first').reset_index(drop=True)
    if source_metadata:
        meta=pd.read_csv(source_metadata)
        if 'filename' not in meta or 'prompt' not in meta:raise ValueError('Metadata must contain filename and prompt columns')
        if meta['filename'].duplicated().any():raise ValueError('Duplicate filename values in metadata')
        meta['filename']=meta['filename'].astype(str).str.replace('\\','/',regex=False)
        existing=set(df.filename)
        if not set(meta.filename).issubset(existing):raise ValueError('External metadata has unknown filenames')
        # Keep only selected metadata rows, but every image must have a prompt.
        keep=[c for c in meta.columns if c not in ('label','image_path','md5','class_idx')]
        df=df.merge(meta[keep],on='filename',how='left',validate='one_to_one')
        if df['prompt'].isna().any():raise ValueError('Missing prompt for some images in supplied metadata')
        if 'expert_verified' not in df:df['expert_verified']=False
        df['prompt_origin']='imported_not_automatically_verified'
    else:
        gen=[legacy_prompt(row.label,row.filename) for row in df.itertuples()]
        df=pd.concat([df,pd.DataFrame(gen)],axis=1)
        df['expert_verified']=False
        df['prompt_origin']='deterministic_legacy_unverified'
    df['class_idx']=df.label.map(CLASS_TO_IDX).astype(int)
    df['expert_verified']=df.expert_verified.fillna(False).astype(str).str.lower().isin(['true','1','yes','y'])
    if df.groupby('label').size().min() < 10:raise ValueError('Insufficient samples for 80/10/10 stratification in one or more classes')
    # hash audit: identical binary same image never crosses split
    train,tmp=train_test_split(df,test_size=.20,stratify=df['label'],random_state=split_seed)
    val,test=train_test_split(tmp,test_size=.50,stratify=tmp['label'],random_state=split_seed)
    dfs={'train':train,'val':val,'test':test}
    for name,part in dfs.items():
        part=part.sort_values(['label','filename']).reset_index(drop=True)
        dfs[name]=part
        part.to_csv(out/f'{name}_split.csv',index=False,encoding='utf-8')
    df.to_csv(out/'metadata_full.csv',index=False,encoding='utf-8')
    # template for actual expert or annotator review; blank decisions must be filled by humans.
    review_cols=['filename','label','prompt','motif_source','technique','layout_pattern','color_style',
                 'expert_verified','reviewer_id','image_prompt_correct','corrected_prompt','notes']
    review=df.copy()
    for c in review_cols:
        if c not in review:review[c]=''
    review['reviewer_id']='';review['image_prompt_correct']='';review['corrected_prompt']='';review['notes']=''
    review[review_cols].to_csv(out/'prompt_annotation_review_TEMPLATE.csv',index=False,encoding='utf-8')
    report=dict(raw_images=len(raw),unique_md5_images=len(df),exact_duplicate_removed=len(raw)-len(df),
                invalid_images=len(broken),unique_prompts=df.prompt.nunique(),
                split_seed=split_seed,split_counts={n:dict(p.label.value_counts()) for n,p in dfs.items()},
                expert_verified_rows=int(df.expert_verified.sum()),
                warning='Hash-generated attributes are NOT guaranteed to describe image content. Expert verify before claiming semantic alignment.')
    dump_json(out/'dataset_audit.json',report)
    return report

def load_splits(split_dir):
    base=Path(split_dir)
    dfs={k:pd.read_csv(base/f'{k}_split.csv').fillna({'prompt':''}) for k in ('train','val','test')}
    ids=[set(df.md5) for df in dfs.values()]
    if any(ids[i]&ids[j] for i in range(3) for j in range(i+1,3)):
        raise ValueError('DATA LEAKAGE: duplicate MD5 between splits')
    for name,df in dfs.items():
        if not {'image_path','label','prompt','class_idx','md5'}.issubset(df):raise ValueError(f'Invalid split CSV: {name}')
        if (df.prompt.astype(str).str.strip()=='').any():raise ValueError(f'Blank prompts: {name}')
        if not df.image_path.map(lambda x:Path(x).is_file()).all():raise FileNotFoundError(f'Images missing in {name}; regenerate split with correct dataset path')
    return dfs

# -- Torch only imported for neural training/evaluation, not metadata preparation --
def torch_import():
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torchvision import transforms
    return torch,nn,F,transforms

def seed_all(seed):
    torch,_,_,_=torch_import()
    random.seed(seed);np.random.seed(seed);torch.manual_seed(seed)
    if torch.cuda.is_available():torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic=True
    torch.backends.cudnn.benchmark=False

def device_auto(request='auto'):
    torch,_,_,_=torch_import()
    return torch.device(('cuda' if torch.cuda.is_available() else 'cpu') if request=='auto' else request)

class Encoder:
    def __init__(self, mode, device, model_name=TRANSFORMER_NAME, out_dim=384):
        self.mode=mode;self.device=device;self.out_dim=out_dim;self.model_name=model_name
        if mode=='transformer_text':
            try:
                from transformers import AutoModel, AutoTokenizer
                self.tokenizer=AutoTokenizer.from_pretrained(model_name)
                self.model=AutoModel.from_pretrained(model_name).to(device).eval()
                self.out_dim=int(self.model.config.hidden_size)
                if self.out_dim != out_dim:raise ValueError(f'Transformer size {self.out_dim} != expected {out_dim}')
            except Exception as exc:
                raise RuntimeError('Transformer encoder unavailable. STOPPING; never silently replace with hash_text.') from exc
        elif mode not in ['cat','hash_text']:
            raise ValueError(f'Invalid encoder: {mode}')

    def encode(self,prompts,labels,batch_size=24):
        torch,_,F,_=torch_import()
        if len(prompts)!=len(labels):raise ValueError('prompts and labels length mismatch')
        if self.mode=='cat':
            arr=np.zeros((len(labels),self.out_dim),dtype='float32')
            for i,label in enumerate(labels):
                j=int(hashlib.md5(str(label).encode()).hexdigest(),16)%self.out_dim
                arr[i,j]=1.
            return torch.from_numpy(arr)
        if self.mode=='hash_text':
            arr=np.zeros((len(prompts),self.out_dim),dtype='float32')
            for i,text in enumerate(prompts):
                tokens=str(text).lower().replace(',',' ').replace('.',' ').replace(';',' ').split() or ['endek']
                for word in tokens:
                    j=int(hashlib.md5(word.encode('utf-8')).hexdigest(),16)%self.out_dim
                    arr[i,j]+=1.
                arr[i]/=max(np.linalg.norm(arr[i]),1e-8)
            return torch.from_numpy(arr)
        out=[]
        with torch.inference_mode():
            for start in range(0,len(prompts),batch_size):
                batch=prompts[start:start+batch_size]
                toks=self.tokenizer(batch,padding=True,truncation=True,max_length=160,return_tensors='pt')
                toks={k:v.to(self.device) for k,v in toks.items()}
                last=self.model(**toks).last_hidden_state
                mask=toks['attention_mask'].unsqueeze(-1)
                vec=(last*mask).sum(dim=1)/mask.sum(dim=1).clamp(min=1)
                out.append(F.normalize(vec,dim=1).cpu())
        return torch.cat(out) if out else torch.empty((0,self.out_dim))

# Model architecture: reproduces naming/state_dict structure of original notebook.
def model_classes():
    torch,nn,F,_=torch_import()
    from torch.nn.utils import spectral_norm
    class ConditionProjector(nn.Module):
        def __init__(self,text_dim,num_classes,class_dim=64,cond_dim=256):
            super().__init__();self.class_embed=nn.Embedding(num_classes,class_dim)
            self.net=nn.Sequential(nn.Linear(text_dim+class_dim,cond_dim),nn.LeakyReLU(.2,inplace=True),
                nn.Linear(cond_dim,cond_dim),nn.LeakyReLU(.2,inplace=True))
        def forward(self,text_embed,labels,disable_class=False):
            class_e=self.class_embed(labels)
            if disable_class:class_e=torch.zeros_like(class_e)
            return self.net(torch.cat([text_embed,class_e],dim=1))
    class Generator(nn.Module):
        def __init__(self,latent_dim=128,text_dim=384,num_classes=4,image_size=256,base_ch=32,cond_dim=256,disable_class=False):
            super().__init__();assert image_size in [64,128,256,512]
            self.image_size=image_size;self.latent_dim=latent_dim;self.disable_class=disable_class
            self.cond=ConditionProjector(text_dim,num_classes,64,cond_dim)
            num_ups=int(math.log2(image_size//4));self.start_ch=min(base_ch*2**num_ups,base_ch*16)
            self.fc=nn.Sequential(nn.Linear(latent_dim+cond_dim,self.start_ch*4*4),
                nn.BatchNorm1d(self.start_ch*4*4),nn.ReLU(True))
            blocks=[];in_ch=self.start_ch
            for i in range(num_ups):
                out_ch=max(base_ch,in_ch//2)
                if i==num_ups-1:out_ch=base_ch
                blocks.extend([nn.ConvTranspose2d(in_ch,out_ch,4,2,1,bias=False),nn.BatchNorm2d(out_ch),nn.ReLU(True)])
                in_ch=out_ch
            blocks.extend([nn.Conv2d(in_ch,3,3,1,1),nn.Tanh()])
            self.net=nn.Sequential(*blocks)
        def forward(self,z,text_embed,labels):
            cond=self.cond(text_embed,labels,self.disable_class)
            x=self.fc(torch.cat([z,cond],dim=1))
            return self.net(x.view(x.size(0),self.start_ch,4,4))
    class Discriminator(nn.Module):
        def __init__(self,text_dim=384,num_classes=4,image_size=256,base_ch=32,cond_dim=256,disable_class=False):
            super().__init__();assert image_size in [64,128,256,512]
            self.disable_class=disable_class
            self.cond=ConditionProjector(text_dim,num_classes,64,cond_dim)
            num_down=int(math.log2(image_size//4));layers=[];in_ch=3
            for i in range(num_down):
                out_ch=min(base_ch*(2**i),base_ch*16)
                layers.extend([spectral_norm(nn.Conv2d(in_ch,out_ch,4,2,1)),nn.LeakyReLU(.2,inplace=True)])
                in_ch=out_ch
            self.features=nn.Sequential(*layers);self.final_ch=in_ch
            self.adv=spectral_norm(nn.Linear(in_ch,1));self.proj=spectral_norm(nn.Linear(cond_dim,in_ch))
        def forward(self,img,text_embed,labels,return_features=False):
            feat=self.features(img).sum(dim=(2,3));cond=self.cond(text_embed,labels,self.disable_class)
            score=self.adv(feat).squeeze(1)+torch.sum(feat*self.proj(cond),dim=1)
            return (score,feat) if return_features else score
    return Generator,Discriminator


def seam_loss(x, band=4):
    """Explicitly targets boundaries of tileable motifs; not proof of motif structure.

    Smoothness consistency across left/right and top/bottom borders using near-edge strips,
    with fixed matching coordinate pairs. Parameter band controls edge-strip width.
    """
    torch,_,F,_=torch_import();k=min(band,x.shape[-1]//8,x.shape[-2]//8)
    if k<1:raise ValueError('Image too small for seam loss')
    left=x[:,:,:,:k];right=x[:,:,:,-k:]
    top=x[:,:,:k,:];bottom=x[:,:,-k:,:]
    return F.l1_loss(left,right)+F.l1_loss(top,bottom)

def seam_metric(images):
    """Lower is better boundary equality, NOT necessarily better Endek ornament quality."""
    torch,_,_,_=torch_import()
    with torch.no_grad():
        return ((images[:,:,:,:1]-images[:,:,:,-1:]).abs().mean()+
                (images[:,:,:1,:]-images[:,:,-1:,:]).abs().mean()).item()/2

@dataclass
class Config:
    split_dir: str
    out_dir: str
    encoder: str='transformer_text'
    seed: int=42
    split_seed: int=3407
    epochs: int=400
    batch_size: int=16
    image_size: int=256
    latent_dim: int=128
    base_ch: int=32
    lr_g: float=2e-4
    lr_d: float=1e-4
    fm_lambda: float=5.0
    r1_gamma: float=5.0
    r1_every: int=16
    ema_decay: float=.999
    seam_lambda: float=0.
    disable_class: bool=False
    mismatch_lambda: float=0.
    sampler_mode: str="weighted"
    online_augment: bool=False
    train_samples_per_epoch: int=0
    save_every: int=50
    sample_every: int=25
    device: str='auto'
    model_name: str=TRANSFORMER_NAME
    num_workers: int=0
    early_stop_enabled: bool=False
    early_stop_min_epoch: int=200
    early_stop_every: int=10
    early_stop_patience_epochs: int=80
    early_stop_min_delta: float=0.002
    early_stop_fixed_seed: int=90731
    early_stop_kid_subsets: int=10
    early_stop_kid_subset_size: int=25


def make_dataset(df,encoder,image_size=256,augment=False):
    torch,_,_,transforms=torch_import()
    from torch.utils.data import Dataset
    e=encoder.encode(df.prompt.astype(str).tolist(),df.label.astype(str).tolist()).float()
    trans=[transforms.Resize((image_size,image_size))]
    if augment:trans += [transforms.RandomHorizontalFlip(.5), transforms.RandomApply([
        transforms.ColorJitter(brightness=.08,contrast=.08,saturation=.08,hue=.02)],p=.3)]
    trans += [transforms.ToTensor(),transforms.Normalize([.5]*3,[.5]*3)]
    tf=transforms.Compose(trans)
    class EndekData(Dataset):
        def __len__(self):return len(df)
        def __getitem__(self,index):
            row=df.iloc[index]
            with Image.open(row.image_path) as im:
                im=im.convert('RGB');x=tf(im)
            return x,e[index],int(row.class_idx),str(row.prompt),str(row.image_path)
    return EndekData(),e


def init_models(cfg, device):
    torch,nn,_,_=torch_import()
    G_class,D_class=model_classes()
    args=dict(text_dim=384,num_classes=len(CLASS_NAMES),image_size=cfg.image_size,base_ch=cfg.base_ch,
              cond_dim=256,disable_class=cfg.disable_class)
    G=G_class(latent_dim=cfg.latent_dim,**args).to(device)
    D=D_class(**args).to(device)
    # Initialize as in original author's code; spectral_norm weight is usually weight_orig.
    def init(m):
        name=m.__class__.__name__
        if 'Conv' in name or 'Linear' in name:
            w=getattr(m,'weight_orig',None)
            if w is None:w=getattr(m,'weight',None)
            if w is not None:nn.init.normal_(w,0.,.02)
        if 'BatchNorm' in name:
            if m.weight is not None:nn.init.normal_(m.weight,1.,.02)
            if m.bias is not None:nn.init.zeros_(m.bias)
    G.apply(init);D.apply(init)
    Gema=copy.deepcopy(G).to(device).eval()
    for p in Gema.parameters():p.requires_grad_(False)
    return G,D,Gema

def checkpoint_load(path, device, trusted=False):
    torch,_,_,_=torch_import()
    if not trusted:raise ValueError('Checkpoint contains pickle data. Only load author-owned trusted files with trusted=True.')
    return torch.load(path,map_location=device,weights_only=False)

class FixedValidationKID:
    """Deterministic, pooled validation-only KID monitor.

    This is model selection, NOT claim of semantic alignment or reliable small-sample FID/KID.
    Fixed image-prompt conditions and z are shared at every validation check for a run.
    """
    def __init__(self,cfg,encoder,device):
        torch,_,_,_=torch_import()
        from torchmetrics.image.kid import KernelInceptionDistance
        from torch.utils.data import DataLoader
        self.cfg,self.device=cfg,device
        validation=load_splits(cfg.split_dir)['val'].reset_index(drop=True)
        if len(validation)<cfg.early_stop_kid_subset_size:
            raise ValueError('Validation set too small for selected KID subset size')
        self.dataset,self.embs=make_dataset(validation,encoder,cfg.image_size,augment=False)
        self.batch_size=min(cfg.batch_size,4) if device.type=='cpu' else cfg.batch_size
        self.loader=DataLoader(self.dataset,batch_size=self.batch_size,shuffle=False,num_workers=0)
        g=torch.Generator(device='cpu');g.manual_seed(cfg.early_stop_fixed_seed)
        self.z=torch.randn((len(validation),cfg.latent_dim),generator=g)
        # real features persist after .reset(), fake features are newly calculated each check
        try:
            self.metric=KernelInceptionDistance(
                feature=2048,normalize=False,subsets=cfg.early_stop_kid_subsets,
                subset_size=cfg.early_stop_kid_subset_size,
                reset_real_features=False).to(device)
        except Exception as exc:
            raise RuntimeError('Tidak dapat menyiapkan pretrained Inception untuk validation KID. Cek internet / torch-fidelity. Training FULL TIDAK dimulai.') from exc
        self.conditions=[]
        with torch.inference_mode():
            for images,txt,labels,_,_ in self.loader:
                self.metric.update(((images.to(device)+1)*127.5).clamp(0,255).to(torch.uint8),real=True)
                self.conditions.append((txt.contiguous(),labels.contiguous()))
        self.real_count=len(validation)

    def score(self,ema):
        torch,_,_,_=torch_import()
        was_training=ema.training
        ema.eval()
        cursor=0
        with torch.inference_mode():
            for txt,labels in self.conditions:
                bs=labels.shape[0]
                generated=ema(self.z[cursor:cursor+bs].to(self.device),txt.to(self.device),labels.to(self.device))
                pixels=((generated+1)*127.5).clamp(0,255).to(torch.uint8)
                self.metric.update(pixels,real=False)
                cursor+=bs
            # KID uses RNG for subsets. Fix seed without changing training RNG states.
            with torch.random.fork_rng(devices=[self.device.index or 0] if self.device.type=='cuda' else []):
                torch.manual_seed(self.cfg.early_stop_fixed_seed)
                if self.device.type=='cuda':torch.cuda.manual_seed_all(self.cfg.early_stop_fixed_seed)
                val_mean,val_std=self.metric.compute()
            self.metric.reset()
        if was_training:ema.train()
        score=float(val_mean.detach().cpu())
        if not np.isfinite(score):raise FloatingPointError('Validation KID is non-finite; STOP without selecting checkpoint')
        return score,float(val_std.detach().cpu())


def train(cfg:Config, resume=None, trusted_checkpoint=False):
    torch,_,F,_=torch_import()
    from torch.utils.data import WeightedRandomSampler, DataLoader
    seed_all(cfg.seed)
    device=device_auto(cfg.device)
    if cfg.image_size not in (64,128,256,512):raise ValueError('Unsupported image size')
    if cfg.epochs<1 or cfg.batch_size<2:raise ValueError('Invalid epochs/batch size')
    splits=load_splits(cfg.split_dir)
    train_df=splits['train'];out=Path(cfg.out_dir);out.mkdir(parents=True,exist_ok=True)
    if cfg.mismatch_lambda>0:
        if cfg.encoder=='cat':raise ValueError('Mismatched-prompt loss does not apply to category-only encoder')
        if not train_df['expert_verified'].astype(str).str.lower().isin(['true','1','yes']).all():
            raise ValueError('Matched/mismatched prompt discriminator REQUIRES expert-verified annotations')
    encoder=Encoder(cfg.encoder,device,cfg.model_name)
    dataset,_=make_dataset(train_df,encoder,cfg.image_size,augment=cfg.online_augment)
    weights=train_df['label'].map(lambda y:1.0/(train_df.label==y).sum()).values
    sampler_gen=torch.Generator();sampler_gen.manual_seed(cfg.seed)
    from torch.utils.data import RandomSampler
    sampler_mode=getattr(cfg,"sampler_mode","weighted")
    if sampler_mode=="balanced_fixed_steps":
        target=cfg.train_samples_per_epoch
        if target<cfg.batch_size or target%cfg.batch_size:
            raise ValueError("Fixed-steps mode requires a positive multiple of batch size")
        sampler=WeightedRandomSampler(torch.DoubleTensor(weights),num_samples=target,
                                      replacement=True,generator=sampler_gen)
    elif sampler_mode=="uniform":
        sampler=RandomSampler(dataset,replacement=False,generator=sampler_gen)
    elif sampler_mode=="weighted":
        sampler=WeightedRandomSampler(torch.DoubleTensor(weights),num_samples=len(weights),replacement=True,generator=sampler_gen)
    else:
        raise ValueError("Unsupported sampler mode: "+str(sampler_mode))
    dl=DataLoader(dataset,batch_size=cfg.batch_size,sampler=sampler,drop_last=True,
                  pin_memory=device.type=='cuda',num_workers=cfg.num_workers)
    if not len(dl):raise ValueError('Dataset smaller than batch size with drop_last')
    G,D,ema=init_models(cfg,device)
    optg=torch.optim.Adam(G.parameters(),lr=cfg.lr_g,betas=(0.0,.999))
    optd=torch.optim.Adam(D.parameters(),lr=cfg.lr_d,betas=(0.0,.999))
    validator=FixedValidationKID(cfg,encoder,device) if cfg.early_stop_enabled else None
    start_epoch=1;step=0;best_score=float('inf');best_epoch=0;bad_epochs=0
    monitor_path=out/'validation_selection_log.csv'
    monitor_rows=[]
    if resume:
        ck=checkpoint_load(resume,device,trusted=trusted_checkpoint)
        for k in ['split_dir','encoder','seed','epochs','batch_size','image_size','latent_dim',
                  'base_ch','lr_g','lr_d','fm_lambda','r1_gamma','r1_every','ema_decay',
                  'seam_lambda','disable_class','mismatch_lambda','model_name','sampler_mode',
                   'online_augment','train_samples_per_epoch',
                  'early_stop_enabled','early_stop_min_epoch','early_stop_every',
                  'early_stop_patience_epochs','early_stop_min_delta','early_stop_fixed_seed',
                  'early_stop_kid_subsets','early_stop_kid_subset_size']:
            prev=ck.get('config',{}).get(k)
            if prev is None or prev != getattr(cfg,k):
                raise ValueError(f'Checkpoint V5 incompatible with configuration {k}: {prev} != {getattr(cfg,k)}. Do not mix protocols.')
        G.load_state_dict(ck['G']);D.load_state_dict(ck['D']);ema.load_state_dict(ck['G_ema'])
        optg.load_state_dict(ck['optimizer_G']);optd.load_state_dict(ck['optimizer_D'])
        start_epoch=ck['epoch']+1;step=ck.get('global_step',0)
        es=ck.get('early_stop_state',{})
        best_score=float(es.get('best_KID',float('inf')))
        best_epoch=int(es.get('best_epoch',0));bad_epochs=int(es.get('epochs_since_improvement',0))
        if cfg.early_stop_enabled and not es:
            raise ValueError('Checkpoint lacks early_stop_state; cannot silently resume V4 as V5')
        if 'torch_rng_state' in ck:torch.set_rng_state(ck['torch_rng_state'])
        if 'numpy_rng_state' in ck:np.random.set_state(ck['numpy_rng_state'])
        if 'python_rng_state' in ck:random.setstate(ck['python_rng_state'])
        if 'sampler_rng_state' in ck:sampler_gen.set_state(ck['sampler_rng_state'])
        if torch.cuda.is_available() and ck.get('cuda_rng_state') is not None:
            torch.cuda.set_rng_state_all(ck['cuda_rng_state'])
        if monitor_path.exists():
            monitor_rows=pd.read_csv(monitor_path).query('epoch < @start_epoch').to_dict('records')
    dump_json(out/'run_config.json',asdict(cfg))
    history=[];start=time.time();log_path=out/'training_log.csv'
    if resume and log_path.exists():
        history=pd.read_csv(log_path).query('epoch < @start_epoch').to_dict('records')
    cp_dir=out/'checkpoints';cp_dir.mkdir(exist_ok=True)
    best_path=cp_dir/f'checkpoint_best_val_KID_{cfg.encoder}.pth'
    last_path=cp_dir/f'checkpoint_last_{cfg.encoder}.pth'
    if cfg.early_stop_enabled and resume and best_epoch>0 and not best_path.is_file():
        raise FileNotFoundError('Best validation checkpoint missing during resume: '+str(best_path))

    def save_checkpoint(path,epoch):
        ck=dict(epoch=epoch,global_step=step,G=G.state_dict(),D=D.state_dict(),
                G_ema=ema.state_dict(),optimizer_G=optg.state_dict(),optimizer_D=optd.state_dict(),
                config=asdict(cfg),early_stop_state=dict(best_KID=best_score,best_epoch=best_epoch,
                epochs_since_improvement=bad_epochs,selected_on='validation_only'),
                torch_rng_state=torch.get_rng_state(),numpy_rng_state=np.random.get_state(),
                python_rng_state=random.getstate(),sampler_rng_state=sampler_gen.get_state(),
                cuda_rng_state=torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None)
        tmp=str(path)+'.tmp';torch.save(ck,tmp);os.replace(tmp,path)

    print('TRAIN:',cfg.encoder,'seam=',cfg.seam_lambda,'no_class=',cfg.disable_class,
          'seed=',cfg.seed,'device=',device,'train=',len(dataset),'val=',len(splits['val']),
          'steps per epoch=',len(dl),'early_stop=',cfg.early_stop_enabled,flush=True)
    final_status='max_epochs';actual_end=start_epoch-1
    for epoch in range(start_epoch,cfg.epochs+1):
        G.train();D.train();epoch_stats=[]
        for images,text_vec,labels,prompts,_ in dl:
            real=images.to(device);e=text_vec.to(device);y=labels.to(device);bs=len(y)
            for p in D.parameters():p.requires_grad_(True)
            optd.zero_grad(set_to_none=True)
            fake=G(torch.randn(bs,cfg.latent_dim,device=device),e,y).detach()
            dr=D(real,e,y);dfake=D(fake,e,y)
            loss_d=F.relu(1-dr).mean()+F.relu(1+dfake).mean()
            if cfg.mismatch_lambda:
                shifted=e.clone();valid=torch.zeros(bs,dtype=torch.bool,device=device)
                for cls in y.unique():
                    ix=(y==cls).nonzero(as_tuple=True)[0].tolist()
                    for ii in ix:
                        alternatives=[jj for jj in ix if jj!=ii and prompts[jj]!=prompts[ii]]
                        if alternatives:
                            shifted[ii]=e[alternatives[0]];valid[ii]=True
                if valid.any():
                    mismatch=D(real[valid],shifted[valid],y[valid])
                    loss_d=loss_d+cfg.mismatch_lambda*F.relu(1+mismatch).mean()
            loss_d.backward();optd.step()
            r1_value=0.
            if cfg.r1_every>0 and step%cfg.r1_every==0 and cfg.r1_gamma>0:
                optd.zero_grad(set_to_none=True)
                x=real.detach().requires_grad_(True)
                dr1=D(x,e,y)
                grad=torch.autograd.grad(dr1.sum(),x,create_graph=True)[0]
                penalty=grad.square().reshape(bs,-1).sum(1).mean()
                ((cfg.r1_gamma/2)*cfg.r1_every*penalty).backward();optd.step()
                r1_value=penalty.item()
            for p in D.parameters():p.requires_grad_(False)
            optg.zero_grad(set_to_none=True)
            fg=G(torch.randn(bs,cfg.latent_dim,device=device),e,y)
            score,feat=D(fg,e,y,return_features=True)
            with torch.no_grad():_,fr=D(real,e,y,return_features=True)
            fm=F.l1_loss(feat.mean(0),fr.mean(0))
            base_g=-score.mean()+cfg.fm_lambda*fm
            seam=seam_loss(fg) if cfg.seam_lambda>0 else fg.new_zeros(())
            loss_g=base_g+cfg.seam_lambda*seam
            loss_g.backward();optg.step()
            with torch.no_grad():
                for ea,p in zip(ema.parameters(),G.parameters()):ea.lerp_(p,1-cfg.ema_decay)
                for ea,p in zip(ema.buffers(),G.buffers()):
                    if ea.dtype.is_floating_point:ea.lerp_(p,1-cfg.ema_decay)
                    else:ea.copy_(p)
            epoch_stats.append([loss_g.item(),loss_d.item(),fm.item(),seam.item(),r1_value])
            step+=1
        m=np.array(epoch_stats).mean(axis=0)
        history.append(dict(epoch=epoch,g_loss=m[0],d_loss=m[1],fm_loss=m[2],
                            seam_loss=m[3],r1=m[4],steps=step,minutes=(time.time()-start)/60))
        pd.DataFrame(history).to_csv(log_path,index=False)
        actual_end=epoch;checked=False;improved=False;stop_now=False
        if validator and epoch >= cfg.early_stop_min_epoch and epoch % cfg.early_stop_every == 0:
            # Never select on training losses or on test images.
            val_kid,val_kid_std=validator.score(ema)
            checked=True
            improved=val_kid < (best_score-cfg.early_stop_min_delta)
            if improved:
                best_score=val_kid;best_epoch=epoch;bad_epochs=0
                save_checkpoint(best_path,epoch)
            else:
                bad_epochs=epoch-best_epoch
            monitor_rows.append(dict(epoch=epoch,val_KID=val_kid,val_KID_subset_std=val_kid_std,
                                     best_KID=best_score,best_epoch=best_epoch,
                                     improved=bool(improved),epochs_since_improvement=bad_epochs,
                                     n_val_real=validator.real_count,n_val_fake=validator.real_count,
                                     purpose='checkpoint_selection_validation_only'))
            pd.DataFrame(monitor_rows).to_csv(monitor_path,index=False)
            stop_now=bad_epochs>=cfg.early_stop_patience_epochs
            print(f'[VALIDATION] epoch={epoch} KID={val_kid:.6f} +/- {val_kid_std:.6f}'
                  f' best={best_score:.6f}@{best_epoch} no_improvement={bad_epochs}/{cfg.early_stop_patience_epochs}',
                  flush=True)
        if epoch%max(1,cfg.sample_every)==0 or epoch==1 or epoch==cfg.epochs:
            print(f'epoch={epoch} G={m[0]:.4f} D={m[1]:.4f} seam={m[3]:.4f}',flush=True)
        if epoch%cfg.save_every==0 or epoch==cfg.epochs or checked or stop_now:
            save_checkpoint(last_path,epoch)
            if epoch%cfg.save_every==0 or epoch==cfg.epochs or stop_now:
                save_checkpoint(cp_dir/f'checkpoint_epoch_{epoch:04d}_{cfg.encoder}.pth',epoch)
        if stop_now:
            final_status='early_stopped';print('EARLY STOP reached at epoch',epoch,flush=True);break
    completed=(actual_end>=cfg.epochs or final_status=='early_stopped')
    dump_json(out/'run_state.json',dict(status=final_status if completed else 'incomplete',
         completed=completed,actual_epochs=actual_end,max_epochs=cfg.epochs,
         global_steps=step,best_val_KID=(best_score if np.isfinite(best_score) else None),
         best_epoch=best_epoch,best_checkpoint=str(best_path) if best_path.exists() else None,
         last_checkpoint=str(last_path),model_selection='validation_KID_only' if validator else 'last_SMOKE',
         validation_n=len(splits['val']),test_used_for_model_selection=False,
         min_epoch=cfg.early_stop_min_epoch,patience_epochs=cfg.early_stop_patience_epochs,
         validation_interval=cfg.early_stop_every,delta=cfg.early_stop_min_delta))
    return out

def generator_from_checkpoint(checkpoint,device='auto',trusted=False,config_override=None):
    torch,_,_,_=torch_import();device=device_auto(device)
    ck=checkpoint_load(checkpoint,device,trusted=trusted)
    conf=ck.get('config',{})
    # legacy original checkpoint format: config keys original names; default params faithfully old.
    if config_override: conf={**conf,**config_override}
    cfg=Config(split_dir=str(conf.get('split_dir','')),out_dir=str(conf.get('out_dir','')),
               image_size=int(conf.get('image_size',256)),latent_dim=int(conf.get('latent_dim',128)),
               base_ch=int(conf.get('base_ch',32)),disable_class=bool(conf.get('disable_class',False)),
               encoder=conf.get('encoder',ck.get('experiment_mode',conf.get('experiment_mode','transformer_text'))),
               model_name=conf.get('model_name',TRANSFORMER_NAME),device=str(device))
    if cfg.encoder not in ('cat','hash_text','transformer_text'):raise ValueError('Invalid checkpoint encoder: '+cfg.encoder)
    G_class,_=model_classes();G=G_class(latent_dim=cfg.latent_dim,image_size=cfg.image_size,base_ch=cfg.base_ch,disable_class=cfg.disable_class).to(device)
    G.load_state_dict(ck['G_ema'],strict=True);G.eval()
    encoder=Encoder(cfg.encoder,device,cfg.model_name)
    return G,encoder,cfg,device


def render_images(checkpoint,split_dir,out_dir,n_per_class=128,seed=9001,device='auto',trusted=False,
                  max_batch=12):
    torch,_,_,_=torch_import()
    from torchvision.utils import save_image
    splits=load_splits(split_dir);test=splits['test']
    generator,encoder,cfg,device=generator_from_checkpoint(checkpoint,device=device,trusted=trusted)
    if device.type == 'cpu': max_batch=min(max_batch,4)  # Generate images in smaller chunks
    out=Path(out_dir);out.mkdir(parents=True,exist_ok=True)
    rng=np.random.default_rng(seed);gen=torch.Generator(device=device);gen.manual_seed(seed)
    records=[]
    for label in CLASS_NAMES:
        subset=test[test.label==label]
        if len(subset)==0:raise ValueError(f'No held-out test prompts for {label}')
        # repeat prompts, but independent latent vectors: n_per_class is the final sample count.
        indices=rng.choice(len(subset),n_per_class,replace=True)
        df=subset.iloc[indices].reset_index(drop=True)
        xfolder=out/label;xfolder.mkdir(exist_ok=True)
        for start in range(0,len(df),max_batch):
            group=df.iloc[start:start+max_batch]
            e=encoder.encode(group.prompt.tolist(),group.label.tolist()).to(device)
            y=torch.tensor([CLASS_TO_IDX[label]]*len(group),device=device)
            z=torch.randn(len(group),cfg.latent_dim,generator=gen,device=device)
            with torch.inference_mode():imgs=(generator(z,e,y).cpu()+1).div(2).clamp(0,1)
            for i,row in enumerate(group.itertuples()):
                path=xfolder/f'{label}_{start+i:04d}.png';save_image(imgs[i],path)
                records.append(dict(generated_path=str(path.resolve()),label=label,prompt=row.prompt,
                     source_filename=row.filename,latent_seed=seed,generated_index=start+i,
                     checkpoint=str(checkpoint),encoder=cfg.encoder))
    pd.DataFrame(records).to_csv(out/'generated_eval_index.csv',index=False)
    dump_json(out/'generation_config.json',dict(n_per_class=n_per_class,seed=seed,checkpoint=str(checkpoint),
                                                encoder=cfg.encoder,source='held-out prompts'))
    return out/'generated_eval_index.csv'


def load_img(path,size=256):
    torch,_,_,_=torch_import()
    with Image.open(path) as im:ar=np.asarray(im.convert('RGB').resize((size,size)),dtype=np.uint8).copy()
    return torch.from_numpy(ar).permute(2,0,1)

def calc_fid_kid(real_paths,fake_paths,device='auto',size=256,batch=16,kid_subsets=50):
    """Do NOT truncate fake count to n_real. Small real n => exploratory scores only."""
    torch,_,_,_=torch_import()
    from torchmetrics.image.fid import FrechetInceptionDistance
    from torchmetrics.image.kid import KernelInceptionDistance
    if len(real_paths)<2 or len(fake_paths)<2:raise ValueError('Need at least 2 real and fake samples')
    device=device_auto(device)
    if device.type == 'cpu': batch=min(batch,4)  # Avoid CPU RAM spikes, metric unchanged
    try:
        f=FrechetInceptionDistance(feature=2048,normalize=False).to(device)
        k=KernelInceptionDistance(feature=2048,normalize=False,subset_size=min(50,len(real_paths),len(fake_paths)),subsets=kid_subsets).to(device)
    except Exception as exc:raise RuntimeError('Install torchmetrics and torch-fidelity for FID/KID. No silent NaNs.') from exc
    for paths,isreal in [(real_paths,True),(fake_paths,False)]:
        for start in range(0,len(paths),batch):
            x=torch.stack([load_img(p,size) for p in paths[start:start+batch]]).to(device)
            f.update(x,real=isreal);k.update(x,real=isreal)
    fid=float(f.compute().cpu());kid_mu,kid_sd=k.compute()
    return dict(fid=fid,kid=float(kid_mu.cpu()),kid_sd=float(kid_sd.cpu()),
                n_real=len(real_paths),n_fake=len(fake_paths),kid_subset_size=min(50,len(real_paths),len(fake_paths)),
                warning=('EXPLORATORY: <50 real reference images; FID is strongly biased/unstable.' if len(real_paths)<50 else ''))

def evaluate(split_dir,index_csv,out_dir,device='auto',size=256,lpips_pairs=0,do_feature_nn=True):
    """E1 FID/KID + E2 optional LPIPS, memorization check via feature distances."""
    splits=load_splits(split_dir);test=splits['test'];gen=pd.read_csv(index_csv)
    if not {'generated_path','label'}.issubset(gen.columns):raise ValueError('Missing generated_path or label')
    if not gen.generated_path.map(lambda p:Path(p).is_file()).all():raise FileNotFoundError('Generated images missing')
    out=Path(out_dir);out.mkdir(parents=True,exist_ok=True)
    rows=[]
    for c in ['all']+list(CLASS_NAMES):
        real=test if c=='all' else test[test.label==c]
        fake=gen if c=='all' else gen[gen.label==c]
        met=calc_fid_kid(real.image_path.tolist(),fake.generated_path.tolist(),device=device,size=size)
        rows.append({'class':c,**met})
    metrics=pd.DataFrame(rows);metrics.to_csv(out/'E1_FID_KID_corrected.csv',index=False)
    dump_json(out/'E1_protocol.json',dict(note='n_fake NOT truncated to n_real. Small real reference => exploratory only.',
                                            split_dir=str(split_dir),index_csv=str(index_csv),device=device,size=size))
    if lpips_pairs>0:eval_lpips(gen,out,device=device,pairs=lpips_pairs)
    if do_feature_nn:mem_feature_check(splits['train'],gen,out,device=device,size=size)
    # E7: seam discontinuity is only a specific tiling proxy, not ornament coherence.
    seam_rows=[]
    for row in gen.itertuples():
        with Image.open(row.generated_path) as im:
            x=np.asarray(im.convert('RGB'),dtype=np.float32)/255.
        gap=(np.abs(x[:,0,:]-x[:,-1,:]).mean()+np.abs(x[0,:,:]-x[-1,:,:]).mean())/2
        seam_rows.append(dict(label=row.label,generated_path=row.generated_path,seam_gap=float(gap)))
    pd.DataFrame(seam_rows).to_csv(out/'E7_seam_gap_by_image.csv',index=False)
    pd.DataFrame(seam_rows).groupby('label').seam_gap.agg(['mean','std','count']).to_csv(out/'E7_seam_gap_summary.csv')
    return metrics

def eval_lpips(gen,out,device='auto',pairs=50):
    torch,_,_,_=torch_import();device=device_auto(device)
    try:import lpips
    except ImportError as e:raise RuntimeError('Install lpips explicitly to run LPIPS diversity') from e
    dist=lpips.LPIPS(net='alex').to(device).eval()
    rng=np.random.default_rng(12456)
    rows=[]
    for c,sub in gen.groupby('label'):
        # LPIPS pair diversity must be within identical prompt; not arbitrary same-class pair.
        if 'prompt' not in sub:raise ValueError('LPIPS needs prompt column for within-prompt diversity')
        groups=[g for _,g in sub.groupby('prompt') if len(g)>=2]
        if not groups:rows.append(dict(label=c,lpips_mean=float('nan'),n_pairs=0,warning='Insufficient repeated identical prompts'));continue
        scores=[]
        for _ in range(pairs):
            group=groups[int(rng.integers(len(groups)))];i,j=rng.choice(len(group),2,replace=False)
            a=load_img(group.iloc[i]['generated_path']).float().unsqueeze(0).to(device)/127.5-1
            b=load_img(group.iloc[j]['generated_path']).float().unsqueeze(0).to(device)/127.5-1
            assert a.min() >= -1.01 and a.max() <= 1.01 and b.min() >= -1.01 and b.max() <= 1.01, 'LPIPS input harus dalam rentang [-1,1]'
            with torch.inference_mode():scores.append(float(dist(a,b).item()))
        rows.append(dict(label=c,lpips_mean=np.mean(scores),lpips_sd=np.std(scores,ddof=1) if len(scores)>1 else np.nan,n_pairs=len(scores),warning='LPIPS perceptual variability, not semantic compliance'))
    pd.DataFrame(rows).to_csv(Path(out)/'E2_LPIPS.csv',index=False)

def mem_feature_check(train,gen,out,device='auto',size=256):
    """Image feature nearest-neighbor diagnostic only; NOT formal memorization proof."""
    torch,_,_,_=torch_import();device=device_auto(device)
    try:
        from torchvision.models import resnet18,ResNet18_Weights
        weights=ResNet18_Weights.DEFAULT
        model=resnet18(weights=weights)
        model.fc=torch.nn.Identity();model=model.eval().to(device)
        transform=weights.transforms()
    except Exception as exc:
        warnings.warn('Pretrained ResNet-18 weights unavailable; skipping feature NN diagnostic: '+str(exc))
        return
    def extract(paths):
        out=[]
        for path in paths:
            with Image.open(path) as im: img=transform(im.convert('RGB')).unsqueeze(0).to(device)
            with torch.inference_mode():feat=model(img)
            out.append(feat.flatten().cpu())
        return torch.nn.functional.normalize(torch.stack(out),dim=1)
    rows=[]
    for c in CLASS_NAMES:
        real=train[train.label==c];fake=gen[gen.label==c]
        rf=extract(real.image_path.tolist());ff=extract(fake.generated_path.tolist())
        sim=ff@rf.T
        best,ix=sim.max(dim=1)
        for i,row in enumerate(fake.itertuples()):
            rows.append(dict(label=c,generated_path=row.generated_path,
                             nearest_train=real.iloc[int(ix[i])]['image_path'],cosine_similarity=float(best[i])))
    pd.DataFrame(rows).to_csv(Path(out)/'E2_NN_feature_diagnostic.csv',index=False)


def counterfactual(checkpoint,out_dir,label='flora',device='auto',trusted=False,n_latents=12,seed=5678):
    """E3 pairs SAME z and SAME class with only a single controlled attribute changed.

    These are deliberately simple diagnostic prompts, not automatically verified annotations.
    A changing output is NOT proof of following a semantic instruction.
    """
    torch,_,_,_=torch_import();from torchvision.utils import save_image
    G,encoder,cfg,device=generator_from_checkpoint(checkpoint,device=device,trusted=trusted)
    out=Path(out_dir);out.mkdir(parents=True,exist_ok=True)
    pairs={
      'color': ('Motif Endek Bali kategori '+label+' dengan warna merah dan emas yang kontras, ornamen simetris dan pola ulang menyeluruh.',
                'Motif Endek Bali kategori '+label+' dengan warna biru, hitam, dan putih dengan kontras kuat, ornamen simetris dan pola ulang menyeluruh.'),
      'layout': ('Motif Endek Bali kategori '+label+' dengan pola tunggal, warna merah dan emas yang kontras, ornamen simetris.',
                 'Motif Endek Bali kategori '+label+' dengan pola ulang menyeluruh, warna merah dan emas yang kontras, ornamen simetris.'),
      'technique': ('Motif Endek Bali kategori '+label+' dengan teknik stilasi, warna merah dan emas yang kontras, pola ulang menyeluruh.',
                    'Motif Endek Bali kategori '+label+' dengan teknik distorsi, warna merah dan emas yang kontras, pola ulang menyeluruh.')
    }
    rng=torch.Generator(device=device);rng.manual_seed(seed)
    z=torch.randn(n_latents,cfg.latent_dim,generator=rng,device=device)
    y=torch.full((n_latents,),CLASS_TO_IDX[label],dtype=torch.long,device=device)
    records=[]
    for aspect,(a,b) in pairs.items():
        outputs=[]
        for j,prompt in enumerate((a,b)):
            e=encoder.encode([prompt]*n_latents,[label]*n_latents).to(device)
            with torch.inference_mode():images=G(z,e,y).detach().cpu()
            outputs.append(images)
            for i,img in enumerate(images):
                path=out/f'{label}_{aspect}_{j}_z{i:03d}.png';save_image((img+1)/2,path)
                records.append(dict(label=label,attribute=aspect,prompt_variant=j,latent_index=i,
                                    image_path=str(path.resolve()),prompt=prompt))
        dif=((outputs[0]-outputs[1])/2).square().mean(dim=(1,2,3)).sqrt().tolist()
        for i,v in enumerate(dif):records.append(dict(label=label,attribute=aspect,prompt_variant='paired_difference',latent_index=i,
                    image_path='',prompt='',pixel_RMSE=float(v)))
    pd.DataFrame(records).to_csv(out/'E3_counterfactual_manifest.csv',index=False)
    # A falsifiable but limited color intervention test (red/gold vs blue); do NOT
    # extrapolate hue scores to ornament/layout/cultural compliance.
    with open(out/'E3_color_diagnostic.csv','w',newline='',encoding='utf-8') as f:
        w=csv.DictWriter(f,fieldnames=['latent_index','redgold_target_fraction','blue_target_fraction',
            'redgold_pref_on_red_prompt','blue_pref_on_blue_prompt','both_directions_pass'])
        w.writeheader()
        for i in range(n_latents):
            scores=[]
            for j in [0,1]:
                with Image.open(out/f'{label}_color_{j}_z{i:03d}.png') as im:
                    from PIL import ImageColor
                    import colorsys
                    hsv=np.asarray(im.convert('HSV')).astype('float32') /255.
                h,sat,val=hsv[:,:,0],hsv[:,:,1],hsv[:,:,2]
                # Red and gold approximated as visible hue/saturation regions;
                # threshold is a probe chosen a priori, not a validated color classifier.
                rg=((((h < .055)|(h > .955))|((h > .10)&(h < .19)))&(sat>.32)&(val>.22)).mean()
                blue=((h>.52)&(h<.73)&(sat>.3)&(val>.2)).mean()
                scores.append((float(rg),float(blue)))
            pass_rg=scores[0][0]>scores[1][0];pass_blue=scores[1][1]>scores[0][1]
            w.writerow(dict(latent_index=i,redgold_target_fraction=scores[0][0],
                 blue_target_fraction=scores[1][1],redgold_pref_on_red_prompt=pass_rg,
                 blue_pref_on_blue_prompt=pass_blue,both_directions_pass=pass_rg and pass_blue))
    # Image-pair RMSE alone is a responsiveness indicator, NOT semantic accuracy.
    return out/'E3_counterfactual_manifest.csv'


def expert_template(index_csv,out_dir,n_per_class=12,seed=42):
    """E4 creates BLANK evaluation sheets; human must fill. Never generate fake expert scores."""
    data=pd.read_csv(index_csv)
    needed={'generated_path','label','prompt'}
    if not needed.issubset(data):raise ValueError('index CSV needs '+str(needed))
    out=Path(out_dir);out.mkdir(parents=True,exist_ok=True)
    selected=[]
    for label,group in data.groupby('label'):
        subset=group.sample(min(n_per_class,len(group)),random_state=seed)
        selected.extend(subset.to_dict('records'))
    rng=np.random.default_rng(seed);rng.shuffle(selected)
    items=[];key=[];blind_folder=out/'expert_images_BLINDED';blind_folder.mkdir(exist_ok=True)
    for row in selected:
        sample_id=f'E{len(items)+1:04d}'
        blind_path=blind_folder/f'{sample_id}.png'
        with Image.open(row['generated_path']) as im:im.convert('RGB').save(blind_path)
        items.append({'sample_id':sample_id,'image_path':str(blind_path.resolve()),
             'label':row['label'],'prompt':row['prompt'],'expert_id':'',
             'ornament_clarity_1to5':'','repeat_regular_1to5':'','color_prompt_match_1to5':'',
             'visual_cultural_plausibility_1to5':'','weaving_feasibility_1to5':'',
             'overall_1to5':'','notes':''})
        key.append({'sample_id':sample_id,'original_image_path':row['generated_path']})
    pd.DataFrame(items).to_csv(out/'E4_expert_BLANK_form.csv',index=False)
    pd.DataFrame(key).to_csv(out/'E4_PRIVATE_blinding_key.csv',index=False)
    dump_json(out/'E4_expert_protocol.json',dict(instructions=[
        'Recruit qualified Endek designers/weavers; report expert count and expertise.',
        'Blind model names during evaluation; independently rate each same set of images.',
        '1=very poor, 5=excellent; scores must come from real human raters.',
        'Report inter-rater agreement and disagreement; do not fabricate ratings.',
        'Weaving feasibility and cultural acceptability remain unproven until rated.',
        'If evaluating multiple models, balance/rerandomize their image order and sample IDs.']))
    return out/'E4_expert_BLANK_form.csv'


def export_ada_data(split_dir,out_dir):
    """E6 prepare labels for official StyleGAN2-ADA class-conditioned baseline.

    Independent external baseline; official code must be installed/run separately.
    """
    dfs=load_splits(split_dir);out=Path(out_dir);out.mkdir(parents=True,exist_ok=True)
    split_dir_names={}
    for split,frame in dfs.items():
        dest=out/split;dest.mkdir(exist_ok=True)
        labels=[]
        for ix,row in enumerate(frame.itertuples()):
            name=f'{ix:05d}.png'
            with Image.open(row.image_path) as im: im.convert('RGB').resize((256,256)).save(dest/name)
            labels.append([name,int(row.class_idx)])
        dump_json(dest/'dataset.json',dict(labels=labels))
        split_dir_names[split]=str(dest)
    dump_json(out/'E6_export_protocol.json',dict(source='official StyleGAN2-ADA: https://github.com/NVlabs/stylegan2-ada-pytorch',
        baseline='class-conditional only; NOT matched prompt-conditioned T2I baseline',
        status='DATA EXPORTED; TRAINING NOT PERFORMED',
        note='Prepare separate train.zip via official dataset_tool.py --source=<train_folder> --dest=train.zip. '
             'Use official train.py with --cond=1 and report compute budget, checkpoints and metrics. '
             'Do not use test images for training.',folders=split_dir_names))
    return out


def legacy_analysis(summary_folder,out_dir):
    """E0 summary; accepts extracted Resume directory, never generates new results."""
    root=Path(summary_folder);out=Path(out_dir);out.mkdir(parents=True,exist_ok=True)
    rows=[]
    for p in sorted(root.rglob('fid_kid_metrics_*.csv')):
        f=pd.read_csv(p)
        v=p.stem.replace('fid_kid_metrics_','')
        seed_match=re.search(r'(?:cat|hash_text|transformer_text)_(42|123|2025)',p.parent.name)
        seed=int(seed_match.group(1)) if seed_match else None
        class_df=f[f['scope']=='per_class']
        if not len(class_df):continue
        rows.append(dict(variant=v,seed=seed,macro_fid=float(class_df.fid.mean()),macro_kid=float(class_df.kid_mean.mean()),
                         source_file=str(p),n_real_classwise=int(class_df.n_real.sum()),
                         warning='Original FID/KID truncated n_fake to n_real; historical values ONLY'))
    result=pd.DataFrame(rows)
    result.to_csv(out/'E0_legacy_audit.csv',index=False)
    if len(result):
        result.groupby('variant')[['macro_fid','macro_kid']].agg(['mean','std']).to_csv(out/'E0_legacy_summary.csv')
    return result