from pathlib import Path
from PIL import Image, ImageChops, ImageStat
import csv, json, statistics, collections
ROOT=Path(r"D:\DISERTASI_S3\WIWIK\EXPERIMEN_2\ENDEK_AUGMENTED_V1")
with (ROOT/"augmentation_manifest.csv").open(newline="",encoding="utf-8-sig") as f:
    rows=list(csv.DictReader(f))
variants=[r for r in rows if r["generated"]=="1"]
original_by_source={r["source_sha256"]:r for r in rows if r["generated"]=="0" and r["split"]=="train"}
stats=[]; misses=[]
for a in variants:
    src=original_by_source.get(a["source_sha256"])
    if src is None: raise AssertionError("Missing original for "+a["origin"])
    with Image.open(src["path"]) as ia,Image.open(a["path"]) as ib:
        xa=ia.convert("RGB")
        xb=ib.convert("RGB")
        if xa.size !=(256,256) or xb.size !=(256,256):
            raise AssertionError("Unexpected image resolution")
        delta=ImageChops.difference(xa,xb)
        avg=sum(ImageStat.Stat(delta).mean)/3/255
        stats.append(avg)
        if avg==0:misses.append(a["path"])
source_stats=collections.Counter(r["source_sha256"] for r in variants)
byclass={c:collections.Counter(r["source_sha256"] for r in variants if r["class_name"]==c) for c in ("flora","fauna","dekoratif","geometris")}
output={
 "manifest_rows":len(rows),
 "generated_variants":len(variants),
 "train_sources_with_augmentation":len(source_stats),
 "train_sources_total":len(original_by_source),
 "variant_per_source_min":min(source_stats.values()),
 "variant_per_source_max":max(source_stats.values()),
 "variant_count_by_class":{k:sum(v.values()) for k,v in byclass.items()},
 "augmented_variants_per_source_range_by_class":{k: [min(v.values()),max(v.values())] for k,v in byclass.items()},
 "mean_abs_rgb_difference_ratio_average":round(statistics.mean(stats),5),
 "mean_abs_rgb_difference_ratio_min":round(min(stats),5),
 "mean_abs_rgb_difference_ratio_max":round(max(stats),5),
 "identical_to_resized_source":len(misses),
 "note":"The pixel difference measures altered colors/contrast/sharpness, not acquisition of new motifs or meaningful design novelty."
}
(ROOT/"augmentation_quality_audit.json").write_text(json.dumps(output,indent=2,ensure_ascii=False),encoding="utf-8")
print(json.dumps(output,indent=2,ensure_ascii=False))
