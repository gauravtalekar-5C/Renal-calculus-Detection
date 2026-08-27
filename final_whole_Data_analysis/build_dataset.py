#!/usr/bin/env python
"""Pool every candidate with a label, once, and cache it.

Separated from training because reading ~1000 per-study CSVs on a box at load 23
takes minutes, and every retrain was paying that cost again.
"""
import os
for _v in ("OMP_NUM_THREADS","OPENBLAS_NUM_THREADS","MKL_NUM_THREADS"):
    os.environ.setdefault(_v,"2")
import glob, re, numpy as np, pandas as pd
HERE=os.path.dirname(os.path.abspath(__file__))
COMPART={"renal":r"kidney|renal|calyx|calyce|pole|nephrolith|concretion",
         "ureteric":r"ureter|vuj|uvj|puj|vesico[- ]?ureter",
         "bladder":r"bladder|vesical|cystolith"}
def nums(t):
    t=str(t); sizes=[]
    for m in re.finditer(r"((?:\d+(?:\.\d+)?\s*x\s*)*\d+(?:\.\d+)?)\s*(mm|cm)\b",t,re.I):
        k=10.0 if m.group(2).lower()=="cm" else 1.0
        sizes+=[float(v)*k for v in re.findall(r"\d+(?:\.\d+)?",m.group(1))]
    hus=[int(g) for m in re.finditer(r"(\d{2,4})\s*(?:HU|hu)\b|(?:HU|attenuation)[:\s]*(\d{2,4})",t) for g in m.groups() if g]
    return sizes,hus
coh=pd.read_csv(f"{HERE}/cohort.csv"); coh["study_id"]=coh.study_id.astype(str)
done={os.path.basename(f)[:-5] for f in glob.glob(f"{HERE}/json/*.json")}
coh=coh[coh.study_id.isin(done)]
out=[]
for t in coh.itertuples():
    sid=t.study_id; txt=str(t.calculus_line).lower()
    named={k for k,p in COMPART.items() if re.search(p,txt)}
    sizes,hus=nums(t.calculus_line)
    for comp,fn,ex in (("renal",f"{sid}_candidates.csv",None),
                       ("ureteric",f"{sid}_ureter_candidates.csv","report_this"),
                       ("bladder",f"{sid}_bladder_candidates.csv",None)):
        f=f"{HERE}/runs/{sid}/csv/per_study/{fn}"
        if not os.path.exists(f): continue
        d=pd.read_csv(f)
        if "is_stone" in d: d=d[d.is_stone.astype(bool)]
        if ex and ex in d: d=d[d[ex].fillna(True).astype(bool)]
        if comp=="renal" and "compartment" in d:
            d=d[~d.compartment.astype(str).str.startswith("bladder")]
        if not len(d): continue
        d=d.copy(); d["comp"]=comp; d["study_id"]=sid; d["expected"]=t.expected
        if t.expected=="normal": d["label"]=0
        else:
            lab=[]
            for r in d.itertuples():
                if comp not in named: lab.append(-1); continue
                mm=float(getattr(r,"max_diameter_mm",np.nan)); hu=float(getattr(r,"hu_max",np.nan))
                c=False
                if sizes and np.isfinite(mm): c|=min(abs(mm-s) for s in sizes)<=3.0
                if hus and np.isfinite(hu):   c|=min(abs(hu/h-1) for h in hus)<=0.30
                if not sizes and not hus: c=True
                lab.append(1 if c else -1)
            d["label"]=lab
        out.append(d)
D=pd.concat(out,ignore_index=True)
D.to_csv(f"{HERE}/candidates_pooled.csv",index=False)
print(f"pooled {len(D)} candidates from {D.study_id.nunique()} studies -> candidates_pooled.csv")
print(f"  negatives (clean study) {int((D.label==0).sum())}")
print(f"  positives (matched)     {int((D.label==1).sum())}")
print(f"  unknown (dropped)       {int((D.label==-1).sum())}")
