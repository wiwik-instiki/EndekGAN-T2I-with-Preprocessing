# -*- coding: utf-8 -*-
"""Validate CSV prompt conditioning and create leakage-safe T2I augmented mapping.
Never changes the user source images or source CSV.
"""
from pathlib import Path
from PIL import Image
import csv, hashlib, json, collections, os

ORIGINAL_META=Path(r"D:\DISERTASI_S3\WIWIK\DATASET_L2\EndekGAN\EndekGAN_Dataset\metadata_expert_guided.csv")
DATASET=Path(r"D:\DISERTASI_S3\WIWIK\EXPERIMEN_2\ENDEK_Dataset")
OUTPUT=Path(r"D:\DISERTASI_S3\WIWIK\EXPERIMEN_2\ENDEK_AUGMENTED_V1")
AUG=OUTPUT/"augmentation_manifest.csv"
SPLIT=OUTPUT/"group_aware_source_split.csv"

def load(path):
    with path.open(encoding="utf-8-sig",newline="") as f:
        return list(csv.DictReader(f))
def md5(path):
    h=hashlib.md5()
    with path.open("rb") as fh:
        for chunk in iter(lambda:fh.read(1<<20),b""):h.update(chunk)
    return h.hexdigest()
def sha256(path):
    h=hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda:fh.read(1<<20),b""):h.update(chunk)
    return h.hexdigest()

meta=load(ORIGINAL_META);augs=load(AUG);splits=load(SPLIT)
assert len(meta)==370 and len(splits)==370 and len(augs)==1018, (len(meta),len(splits),len(augs))
by_name={}
issues=[]
relinked=[]
files=[p for p in DATASET.rglob("*.jpg") if p.is_file()]
available_md5=collections.defaultdict(list)
for p in files:
    available_md5[md5(p).lower()].append(p)
assert len(files)==370
for m in meta:
    name=m["filename"].replace("\\","/")
    source=DATASET/name
    if not source.exists():
        matches=available_md5.get(m["md5"].lower(),[])
        if len(matches)!=1:
            issues.append("unresolved_missing_source:"+name)
            continue
        replacement=matches[0]
        newname=replacement.relative_to(DATASET).as_posix()
        relinked.append({"csv_filename":name,"current_dataset_filename":newname,"same_md5":True})
        name=newname
        source=replacement
    assert name not in by_name, ("duplicate metadata image",name)
    by_name[name]=m
    if md5(source).lower()!=m["md5"].lower():issues.append("source_md5_mismatch:"+name)
    if m["label"]!=name.split("/")[0]:issues.append("label_mismatch:"+name)
    if not m["prompt"].strip():issues.append("empty_prompt:"+name)
    if not m["layout_pattern"].strip():issues.append("empty_layout:"+name)
    if not m["technique"].strip():issues.append("empty_technique:"+name)
    if source.stat().st_size==0:issues.append("empty_file:"+name)
    if (m["image_path"].replace("/","\\").lower() != str(source).replace("/","\\").lower()):
        # Old path is expected because the research project was moved.
        pass
assert not issues, issues[:30]

meta_key_digest=hashlib.sha256("\n".join(sorted(m["filename"]+"\t"+m["md5"] for m in meta)).encode()).hexdigest()
source_split_by_sha={s["sha256"]:s for s in splits}
if len(source_split_by_sha)!=370: raise RuntimeError("Duplicate SHA in split")
mapped=[]
for a in augs:
    src_rel=(a["class_name"]+"/"+a["origin"]).replace("\\","/")
    item=by_name.get(src_rel)
    if item is None:issues.append("metadata_not_found:"+src_rel);continue
    split=source_split_by_sha.get(a["source_sha256"])
    if not split:issues.append("split_not_found:"+src_rel);continue
    if split["class"]!=a["class_name"] or split["file_name"]!=a["origin"] or split["split"]!=a["split"]:
        issues.append("manifest_source_split_mismatch:"+src_rel)
    original_image=DATASET/src_rel
    if sha256(original_image)!=a["source_sha256"]:
        issues.append("sha256_source_mismatch:"+src_rel)
    filepath=Path(a["path"])
    if not filepath.is_file():
        issues.append("output_missing:"+str(filepath));continue
    if filepath.parent.name != a["class_name"] or filepath.parent.parent.name != a["split"]:
        issues.append("output_path_mismatch:"+str(filepath))
    with Image.open(filepath) as image:
        w,h=image.size
        if (w,h)!=(256,256):
            issues.append("bad_output_shape:"+str(filepath))
    mapped.append({
        "image_path":str(filepath),
        "filename":str(filepath.relative_to(OUTPUT)).replace("\\","/"),
        "label":item["label"],
        "class_idx":item["class_idx"],
        "split":a["split"],
        "source_filename":src_rel,
        "source_md5":item["md5"],
        "source_sha256":a["source_sha256"],
        "augmentation_variant":a["variant"],
        "is_augmented":a["generated"],
        "augmentation_recipe":a["recipe"],
        "brightness":a["brightness"],
        "contrast":a["contrast"],
        "color_factor":a["color"],
        "sharpness":a["sharpness"],
        "width":w,
        "height":h,
        "title":item["title"],
        "motif_source":item["motif_source"],
        "technique":item["technique"],
        "layout_pattern":item["layout_pattern"],
        "composition":item["composition"],
        "color_style":item["color_style"],
        "weavability_note":item["weavability_note"],
        "cultural_context":item["cultural_context"],
        "prompt":item["prompt"],
        "prompt_provenance":"inherited_from_original_expert_guided_metadata_not_independently_expert_verified",
        "prompt_semantic_verified":"false"
    })
assert not issues,issues[:30]
assert len(mapped)==1018, len(mapped)
by_source=collections.defaultdict(set)
for m in mapped:by_source[m["source_sha256"]].add(m["split"])
assert len(by_source)==370 and all(len(x)==1 for x in by_source.values())
by_split=collections.Counter(m["split"] for m in mapped)
assert by_split=={"train":944,"val":37,"test":37}
assert all((m["is_augmented"]=="0") for m in mapped if m["split"]!="train")
out_file=OUTPUT/"metadata_expert_guided_T2I_augmented_V1.csv"
with out_file.open("w",encoding="utf-8-sig",newline="") as f:
    writer=csv.DictWriter(f,fieldnames=list(mapped[0]))
    writer.writeheader()
    writer.writerows(mapped)
summary={
"source_csv":str(ORIGINAL_META),
"source_csv_row_count":len(meta),
"source_csv_md5_key_digest":meta_key_digest,
"source_filename_relinks":relinked,
"source_filename_relink_count":len(relinked),
"source_image_rows_matched_to_current_folder":len(meta),
"updated_output_metadata_csv":str(out_file),
"generated_csv_rows":len(mapped),
"unmatched_rows":len(issues),
"split_counts":dict(by_split),
"unique_source_images":len(by_source),
"unique_prompts_original":len(set(m["prompt"] for m in meta)),
"prompt_inheritance_preserved":True,
"prompt_text_changed_by_preprocessing":False,
"classes":dict(collections.Counter(m["label"] for m in meta)),
"layout_patterns":dict(collections.Counter(m["layout_pattern"] for m in meta)),
"techniques":dict(collections.Counter(m["technique"] for m in meta)),
"no_validation_or_test_augmentation":True,
"warning":"Text-image pairing is inherited from metadata and filename-based prompts; correctness of all motif descriptions, technique, layout, cultural statements, and colors has NOT been independently validated. Cropping is NOT applied in V1.",
"legacy_image_path_in_source_csv":"Original CSV points to D:\\DISERTASI_S3\\WIWIK\\DATASET_L2\\EndekGAN\\EndekGAN_Dataset. Use NEW CSV image_path for processed images."
}
(OUTPUT/"metadata_T2I_verification_report.json").write_text(json.dumps(summary,indent=2,ensure_ascii=False),encoding="utf-8")
print(json.dumps(summary,indent=2,ensure_ascii=False))
