# -*- coding: utf-8 -*-
"""Build engine-compatible V2 split files WITHOUT modifying master preprocessing output."""
from pathlib import Path
import pandas as pd,json,hashlib
ROOT=Path(r"D:\DISERTASI_S3\WIWIK\EXPERIMEN_2\ENDEK_T2I_EXPERIMENT_V2")
BASE=ROOT.parent/"ENDEK_PREPROCESSING_V2"/"arms"
ARMS={"A":"full_image_augmented_A","B":"full_plus_crop_ablation_B"}
STAGED=ROOT/"TRAINING_SPLITS_PREPROCESSED_V2"
CLASSES=("flora","fauna","dekoratif","geometris")
def main():
    result={}
    for key,arm in ARMS.items():
        folder=BASE/arm
        target=STAGED/arm
        target.mkdir(parents=True,exist_ok=True)
        data={}
        for split in ("train","val","test"):
            frame=pd.read_csv(folder/(split+"_split.csv"),keep_default_na=False,encoding="utf-8-sig")
            assert {"source_md5","source_sha256","prompt","label","image_path","class_idx","split"}.issubset(frame.columns)
            if "md5" not in frame.columns:frame["md5"]=""
            frame["md5"]=frame["source_md5"].where(frame["md5"].astype(str).str.strip()=="",frame["md5"])
            assert frame["md5"].astype(str).str.strip().ne("").all()
            assert frame["prompt"].astype(str).str.strip().ne("").all()
            assert frame["label"].isin(CLASSES).all()
            assert frame["image_path"].map(lambda s:Path(str(s)).is_file()).all()
            assert frame["split"].eq(split).all()
            if "expert_verified" not in frame.columns:frame["expert_verified"]=False
            data[split]=frame
            frame.to_csv(target/(split+"_split.csv"),index=False,encoding="utf-8-sig")
        assert {x:len(data[x]) for x in data}=={"train":1280 if key=="B" else 944,"val":37,"test":37}
        sha_sets={sp:set(data[sp].source_sha256) for sp in data}
        md5_sets={sp:set(data[sp].md5) for sp in data}
        for x,y in (("train","val"),("train","test"),("val","test")):
            assert not (sha_sets[x]&sha_sets[y]),("SHA leak",key,x,y)
            assert not (md5_sets[x]&md5_sets[y]),("MD5 leak",key,x,y)
        class_count=data["train"].label.value_counts()
        assert set(class_count.index)==set(CLASSES)
        assert class_count.nunique()==1
        result[key]={"path":str(target),"train":len(data["train"]),"val":len(data["val"]),
                     "test":len(data["test"]),"class_counts":class_count.to_dict(),
                     "crop_rows":int(data["train"].is_crop.astype(str).str.lower().isin(["true","1"]).sum()),
                     "source_group_leakage":False}
    for sp in ("val","test"):
        a=data
        x=pd.read_csv(STAGED/ARMS["A"]/(sp+"_split.csv"))
        y=pd.read_csv(STAGED/ARMS["B"]/(sp+"_split.csv"))
        assert set(x.source_sha256)==set(y.source_sha256)
    (ROOT/"TRAINING_SPLITS_PREPROCESSED_V2_REPORT.json").write_text(
        json.dumps(result,indent=2,ensure_ascii=False),encoding="utf-8")
    print(json.dumps(result,indent=2,ensure_ascii=False))
if __name__=="__main__":main()
