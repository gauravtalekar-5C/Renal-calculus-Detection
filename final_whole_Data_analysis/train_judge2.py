#!/usr/bin/env python
"""Train the per-candidate judge and report the number that matters.

AUC on labelled candidates is not the deliverable. The deliverable is what
happens to SENSITIVITY and the FALSE-POSITIVE RATE per study, because that is
what a radiologist experiences. So:

  * studies are split into folds, never candidates -- candidates from one study
    share a patient, a scanner and a kernel, and splitting them lets the model
    memorise the study instead of the finding
  * each fold's model scores EVERY candidate of its held-out studies, including
    the ones with no label. In production the judge sees everything; evaluating
    only on labelled rows would measure a system that does not exist
  * a study is Abnormal if any surviving candidate scores above the threshold,
    which is exactly the existing rule with the judge inserted
"""
import os
for _v in ("OMP_NUM_THREADS","OPENBLAS_NUM_THREADS","MKL_NUM_THREADS","NUMEXPR_NUM_THREADS"):
    os.environ[_v]="2"
import json, numpy as np, pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score, average_precision_score
from sklearn.model_selection import GroupKFold

HERE=os.path.dirname(os.path.abspath(__file__))
DROP={"study_id","stone_id","candidate_id","reject_reason","is_stone","report_this",
      "review_flag","measurement_flag","centroid_vox","shape_source","measure_stage",
      "expected","label","hu_rank_side","oversized_review"}

D=pd.read_csv(f"{HERE}/candidates_pooled.csv")
D["study_id"]=D.study_id.astype(str)
num=[c for c in D.columns if c not in DROP and D[c].dtype.kind in "fib"]
cats=[c for c in ("comp","zone","compartment","side","location") if c in D.columns]
X=pd.concat([D[num].astype(float)]+
            [pd.get_dummies(D[c].astype(str),prefix=c) for c in cats],axis=1)
feats=list(X.columns); X=X.astype(float).values
y=D.label.values.astype(int); g=D.study_id.values
print(f"{len(D)} candidates, {len(feats)} features, "
      f"{(y==1).sum()} pos / {(y==0).sum()} neg / {(y==-1).sum()} unlabelled")

studies=D.study_id.unique()
rng=np.random.RandomState(0); rng.shuffle(studies)
fold_of={s:i%5 for i,s in enumerate(studies)}
D["fold"]=D.study_id.map(fold_of)
p=np.full(len(D),np.nan)
for k in range(5):
    tr=(D.fold.values!=k)&(y>=0)      # labelled rows from other studies only
    te=(D.fold.values==k)             # ALL rows of held-out studies
    clf=HistGradientBoostingClassifier(max_iter=250,learning_rate=0.06,
        max_leaf_nodes=15,l2_regularization=1.0,min_samples_leaf=20,random_state=0)
    clf.fit(X[tr],y[tr])
    p[te]=clf.predict_proba(X[te])[:,1]
D["p"]=p
lab=y>=0
auc=roc_auc_score(y[lab],p[lab]); ap=average_precision_score(y[lab],p[lab])
print(f"\nout-of-fold on labelled candidates:  AUC {auc:.3f}   AP {ap:.3f}")

# ---- the number that matters: study level -----------------------------------
meta=D.groupby("study_id").expected.first()
print(f"\n{'thresh':>7}{'sens':>9}{'FP':>9}{'spec':>9}   (studies: "
      f"{(meta=='abnormal').sum()} pos / {(meta=='normal').sum()} neg)")
best=None
for thr in (0.0,0.05,0.10,0.15,0.20,0.30,0.40,0.50,0.60,0.70,0.80):
    keep=D[D.p>=thr]
    flagged=set(keep.study_id)
    pos=[s for s in meta.index if meta[s]=="abnormal"]
    neg=[s for s in meta.index if meta[s]=="normal"]
    sens=100*sum(s in flagged for s in pos)/len(pos)
    fp=100*sum(s in flagged for s in neg)/len(neg)
    mark=""
    if sens>=90 and (best is None or fp<best[2]): best=(thr,sens,fp); mark=" <-"
    print(f"{thr:>7.2f}{sens:>8.1f}%{fp:>8.1f}%{100-fp:>8.1f}%{mark}")
if best: print(f"\nbest point with sensitivity >= 90%:  threshold {best[0]:.2f}  "
               f"sens {best[1]:.1f}%  FP {best[2]:.1f}%")
print("\nCOMPARISON")
print("  as it shipped            sens 97.6%   FP 87.4%")
print("  + the two rule fixes     sens 94.1%   FP 52.4%  (simulated)")
D.to_csv(f"{HERE}/judge_scored.csv",index=False)
json.dump({"auc":float(auc),"ap":float(ap)},open(f"{HERE}/judge_report.json","w"),indent=2)
