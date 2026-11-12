# -*- coding: utf-8 -*-
"""EndekGAN source-safe TRAIN augmentation. No changes to original photographs."""
from pathlib import Path
import csv, json, hashlib, random, collections, math
from PIL import Image, ImageOps, ImageEnhance, ImageFilter, ImageDraw, ImageChops, ImageStat

ROOT = Path(r"D:\DISERTASI_S3\WIWIK\EXPERIMEN_2")
SRC = ROOT / "ENDEK_Dataset"
OLD_MANIFEST = ROOT / "ENDEK_AUGMENTED_V1" / "source_split_pre_revision.csv"
OUT = ROOT / "ENDEK_AUGMENTED_V1"
CLASSES = ("flora", "fauna", "dekoratif", "geometris")
TARGET_TRAIN_PER_CLASS = 236
OUT_SIZE=256
SEED=20261112
CLOSE_THRESHOLD=4
NEAR_PAIR_TO_GROUP = ("F-015-flora_133.jpg", "F-018-flora_122.jpg")

def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()
def dhash(path):
    with Image.open(path) as image:
        g=ImageOps.grayscale(image).resize((9,8),Image.Resampling.LANCZOS)
        v=list(g.getdata())
    val=0
    for j in range(8):
        for i in range(8):
            val=(val<<1)|int(v[9*j+i] > v[9*j+i+1])
    return val
def difference(x,y):
    return (x^y).bit_count()
def save_jpg(image,path):
    path.parent.mkdir(parents=True,exist_ok=True)
    image.save(path,"JPEG",quality=94,subsampling=0,optimize=True)
def prepare_base(src):
    with Image.open(src) as im:
        return ImageOps.exif_transpose(im).convert("RGB").resize((OUT_SIZE,OUT_SIZE),Image.Resampling.LANCZOS)
def augment(base, rng):
    # VERY SMALL photometric-only transformation, avoids any affine/rotation/flip/crop
    params={
        "brightness":round(rng.uniform(.935,1.065),6),
        "contrast":round(rng.uniform(.945,1.055),6),
        "color":round(rng.uniform(.965,1.035),6),
        "sharpness":round(rng.uniform(.88,1.16),6)
    }
    result=ImageEnhance.Brightness(base).enhance(params["brightness"])
    result=ImageEnhance.Contrast(result).enhance(params["contrast"])
    result=ImageEnhance.Color(result).enhance(params["color"])
    result=ImageEnhance.Sharpness(result).enhance(params["sharpness"])
    return result,params
def main():
    if not OLD_MANIFEST.exists(): raise RuntimeError("Missing audited source split manifest")
    with OLD_MANIFEST.open(encoding="utf-8-sig",newline="") as f:
        rows=list(csv.DictReader(f))
    if len(rows)!=370: raise RuntimeError("Expected 370 source rows")
    rows.sort(key=lambda r:(CLASSES.index(r["class"]),r["file_name"]))
    dh={}
    sha_seen={}
    for r in rows:
        src=Path(r["source_path"])
        if not src.is_file() or src.parent!=SRC/r["class"]:
            raise RuntimeError("Source path changed: "+str(src))
        h=digest(src)
        if h!=r["sha256"]:raise RuntimeError("Original image modified since last audit: "+str(src))
        if h in sha_seen:raise RuntimeError("Unexpected exact duplicate: "+str(src))
        sha_seen[h]=src
        dh[src.name]=dhash(src)
    # Fix the clear visually reviewed near-identical floral pair crossing train/test
    rmap={r["file_name"]:r for r in rows}
    a,b=(rmap[x] for x in NEAR_PAIR_TO_GROUP)
    if (a["split"],b["split"]) != ("train","test"):
        raise RuntimeError("Unexpected previous pair split, need manual audit")
    if difference(dh[a["file_name"]],dh[b["file_name"]])>CLOSE_THRESHOLD:
        raise RuntimeError("Floral pair similarity has changed; abort")
    b["split"]="train"
    eligible=[r for r in rows if r["class"]=="flora" and r["split"]=="train" and r["file_name"] not in NEAR_PAIR_TO_GROUP]
    # choose a deterministic replacement that is not visually near-duplicated with other train sources
    rng=random.Random(3407)
    rng.shuffle(eligible)
    replacement=None
    for candidate in eligible:
        candidate_hash=dh[candidate["file_name"]]
        other_train=[r for r in rows if r["class"]=="flora" and r["split"]=="train" and r is not candidate]
        if min(difference(candidate_hash,dh[r["file_name"]]) for r in other_train)>6:
            replacement=candidate
            break
    if replacement is None:raise RuntimeError("Cannot find visually separated replacement")
    replacement["split"]="test"
    by_split={sp:collections.Counter(r["class"] for r in rows if r["split"]==sp) for sp in ("train","val","test")}
    expected={"train":(118,42,90,46),"val":(14,6,11,6),"test":(15,5,11,6)}
    for sp,counts in expected.items():
        if tuple(by_split[sp][c] for c in CLASSES)!=counts:
            raise RuntimeError("Class split count mismatch: "+str((sp,by_split[sp])))
    # Same-class dHash screening across partitions: threshold is a flag, not proof of identity
    flagged=[]
    for cls in CLASSES:
        subset=[r for r in rows if r["class"]==cls]
        for i,x in enumerate(subset):
            for y in subset[i+1:]:
                if x["split"]==y["split"]:continue
                dist=difference(dh[x["file_name"]],dh[y["file_name"]])
                if dist<=CLOSE_THRESHOLD:
                    flagged.append(dict(class_name=cls,a=x["file_name"],b=y["file_name"],
                                        split_a=x["split"],split_b=y["split"],dhash_distance=dist))
    if flagged:
        raise RuntimeError("Same-class dHash near-pairs across splits; abort augmentation: "+json.dumps(flagged))
    source_columns=["class","file_name","source_path","sha256","split"]
    with (OUT/"group_aware_source_split.csv").open("w",encoding="utf-8-sig",newline="") as f:
        w=csv.DictWriter(f,fieldnames=source_columns);w.writeheader();w.writerows([{k:r[k] for k in source_columns} for r in rows])
    manifest=[]
    # Source names are retained, aug file names carry parent stem / variant
    training={c:[r for r in rows if r["split"]=="train" and r["class"]==c] for c in CLASSES}
    counts_aug={}
    for cls,pool in training.items():
        target=TARGET_TRAIN_PER_CLASS-len(pool)
        if target< len(pool): raise RuntimeError("Target does not augment every source")
        counts_aug[cls]=target
    for r in rows:
        src=Path(r["source_path"])
        base=prepare_base(src)
        dst=OUT/r["split"]/r["class"]/(src.stem+"__original_resized.jpg")
        save_jpg(base,dst)
        manifest.append(dict(class_name=r["class"],split=r["split"],origin=src.name,source_sha256=r["sha256"],variant="original_resized",generated=0,recipe="full_frame_resize_lanczos_256",brightness=1,contrast=1,color=1,sharpness=1,path=str(dst)))
    for cls,pool in training.items():
        # all originals receive at least one variant, then round-robin for minority classes
        pool=sorted(pool,key=lambda r:r["sha256"])
        remaining=counts_aug[cls]
        variant_idx=collections.Counter()
        i=0
        while remaining>0:
            r=pool[i%len(pool)]
            variant_idx[r["sha256"]]+=1
            k=variant_idx[r["sha256"]]
            seed=int(hashlib.sha256((r["sha256"]+"#"+str(k)+"#"+str(SEED)).encode()).hexdigest()[:16],16)
            rng=random.Random(seed)
            base=prepare_base(Path(r["source_path"]))
            im,params=augment(base,rng)
            name=Path(r["file_name"]).stem+f"__aug{k:02d}.jpg"
            dst=OUT/"train"/cls/name
            save_jpg(im,dst)
            manifest.append(dict(class_name=cls,split="train",origin=r["file_name"],source_sha256=r["sha256"],variant=f"aug{k:02d}",generated=1,recipe="mild_photometric_preserve_layout",path=str(dst),**params))
            i+=1
            remaining-=1
    # Check data balance, no parent is assigned across train/val/test
    counts=collections.defaultdict(collections.Counter)
    groups=collections.defaultdict(set)
    for r in manifest:
        counts[r["split"]][r["class_name"]]+=1
        groups[r["source_sha256"]].add(r["split"])
    assert all(len(g)==1 for g in groups.values())
    assert len(groups)==370
    assert len(manifest)==(TARGET_TRAIN_PER_CLASS*4+37+37)
    assert all(counts["train"][c]==TARGET_TRAIN_PER_CLASS for c in CLASSES)
    assert counts["val"]==by_split["val"] and counts["test"]==by_split["test"]
    out_hashes=collections.defaultdict(list)
    for r in manifest:
        p=Path(r["path"])
        if not p.is_file():raise RuntimeError("Missing output "+str(p))
        with Image.open(p) as im:
            im.verify()
        out_hashes[digest(p)].append(r["path"])
    identical=[v for v in out_hashes.values() if len(v)>1]
    if identical:raise RuntimeError(f"Augmented exact duplicates detected: {identical[:2]}")
    cols=["class_name","split","origin","source_sha256","variant","generated","recipe","brightness","contrast","color","sharpness","path"]
    with (OUT/"augmentation_manifest.csv").open("w",encoding="utf-8-sig",newline="") as f:
        w=csv.DictWriter(f,fieldnames=cols);w.writeheader();w.writerows(manifest)
    # Visual inspection: 2 examples per class, three augmentations each
    tile=160; label_h=24; cols_n=4;rows_n=8
    sheet=Image.new("RGB",(cols_n*tile,rows_n*(tile+label_h)),(252,252,252))
    draw=ImageDraw.Draw(sheet)
    for ci,cls in enumerate(CLASSES):
        sample=sorted(training[cls],key=lambda r:r["sha256"])[:2]
        for si,r in enumerate(sample):
            entries=[m for m in manifest if m["split"]=="train" and m["source_sha256"]==r["sha256"]]
            entries=sorted(entries,key=lambda x:(x["generated"],x["variant"]))[:4]
            for col,m in enumerate(entries):
                image=Image.open(m["path"]).convert("RGB").resize((tile,tile),Image.Resampling.LANCZOS)
                row=ci*2+si
                x=col*tile; y=row*(tile+label_h)
                sheet.paste(image,(x,y))
                draw.text((x+4,y+tile+3),cls+" "+m["variant"],fill=(10,10,10))
    sheet.save(OUT/"augmentation_preview.jpg",quality=90,optimize=True)
    report={
      "original_source_folder":str(SRC),"dataset_modified":False,
      "source_images":len(rows),"unique_source_sha":len(sha_seen),
      "out_size":[OUT_SIZE,OUT_SIZE],"train_originals":sum(by_split["train"].values()),
      "train_augmented":sum(counts_aug.values()),"train_total":sum(counts["train"].values()),
      "val_total":sum(counts["val"].values()),"test_total":sum(counts["test"].values()),
      "all_files":len(manifest),"train_class_counts":dict(counts["train"]),
      "val_class_counts":dict(counts["val"]),"test_class_counts":dict(counts["test"]),
      "new_variants_by_class":counts_aug,
      "exact_duplicate_output_groups":len(identical),
      "parent_split_leakage":sum(len(g)>1 for g in groups.values()),
      "source_split_revision":{"grouped_flora_pair":list(NEAR_PAIR_TO_GROUP),"moved_test_to_train":b["file_name"],"moved_train_to_test":replacement["file_name"],"purpose":"keep visually near-identical source pair in the same split, while preserving class counts", "warning":"new experimental split; not identical to previous manuscript"},
      "same_class_dhash_close_cross_split":len(flagged),
      "supported_claim":"class-balanced photometric augmentation of TRAIN ONLY, not new independent motif designs",
      "not_executed":["SMOTE image-pixel interpolation","automated stylization/deformation/distortion","layout-specific cropping","GAN training","domain-expert cultural validation"]
    }
    (OUT/"augmentation_report.json").write_text(json.dumps(report,indent=2,ensure_ascii=False),encoding="utf-8")
    print(json.dumps(report,indent=2,ensure_ascii=False))
if __name__=="__main__":
    main()
