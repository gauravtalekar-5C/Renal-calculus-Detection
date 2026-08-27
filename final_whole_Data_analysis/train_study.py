#!/usr/bin/env python
"""Stage 2: a STUDY-level model over the candidate scores.

WHY THIS STAGE EXISTS. The candidate judge reaches AUC 0.912, but the rule that
turns candidates into an answer -- "Abnormal if ANY candidate scores high" -- is
a maximum, and a maximum over eight candidates is eight chances to be wrong. At
90% sensitivity that rule still gives 53.6% false positives.

A clean study and a stone-bearing study differ in the SHAPE of their score
distribution, not just its maximum: a clean study has many mediocre candidates,
a positive study has one convincing one. "Many mediocre" and "one convincing"
have the same max when a mimic scores well by chance, and they look completely
different in the second-highest score, the count above threshold, and how the
scores split across compartments. This stage gets to see all of that.

Trained and evaluated with the same grouped folds as stage 1, so a study's
candidates never inform its own study-level score.
"""
import os
for _v in ("OMP_NUM_THREADS","OPENBLAS_NUM_THREADS","MKL_NUM_THREADS","NUMEXPR_NUM_THREADS"):
    os.environ[_v]="2"
import json, numpy as np, pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

HERE=os.path.dirname(os.path.abspath(__file__))
D=pd.read_csv(f"{HERE}/judge_scored.csv"); D["study_id"]=D.study_id.astype(str)

def agg(g):
    p=np.sort(g.p.values)[::-1]
    out={"n_cand":len(p),
         "p1":p[0], "p2":p[1] if len(p)>1 else 0.0,
         "p3":p[2] if len(p)>2 else 0.0,
         "p_mean":p.mean(), "p_sum":p.sum(),
         "n_gt50":int((p>=.5).sum()), "n_gt20":int((p>=.2).sum()),
         "gap12":p[0]-(p[1] if len(p)>1 else 0.0)}
    for c in ("renal","ureteric","bladder"):
        s=g[g.comp==c]
        out[f"n_{c}"]=len(s)
        out[f"p1_{c}"]=s.p.max() if len(s) else 0.0
    # the measurement of the best-scoring candidate: a convincing score on a
    # 2 mm 200 HU object means something different from the same score on a
    # 9 mm 1200 HU one
    b=g.loc[g.p.idxmax()]
    for k in ("max_diameter_mm","hu_max","volume_mm3","fill_fraction"):
        out[f"best_{k}"]=float(b[k]) if k in g.columns and pd.notna(b[k]) else np.nan
    out["fold"]=int(g.fold.iloc[0]); out["expected"]=g.expected.iloc[0]
    return pd.Series(out)

S=D.groupby("study_id").apply(agg, include_groups=False).reset_index()
y=(S.expected=="abnormal").astype(int).values
feats=[c for c in S.columns if c not in ("study_id","expected","fold")]
X=S[feats].astype(float).values
print(f"studies {len(S)}  ({y.sum()} positive / {(y==0).sum()} negative), "
      f"{len(feats)} study-level features")

p=np.full(len(S),np.nan)
for k in range(5):
    tr=S.fold.values!=k; te=S.fold.values==k
    clf=HistGradientBoostingClassifier(max_iter=250,learning_rate=0.05,
        max_leaf_nodes=7,l2_regularization=2.0,min_samples_leaf=15,random_state=0)
    clf.fit(X[tr],y[tr]); p[te]=clf.predict_proba(X[te])[:,1]
auc=roc_auc_score(y,p)
print(f"\nSTUDY-LEVEL out-of-fold AUC {auc:.3f}")

print(f"\n{'thresh':>7}{'sens':>9}{'FP':>9}{'spec':>9}")
rows=[]
for thr in np.arange(0.05,0.96,0.05):
    pred=p>=thr
    sens=100*pred[y==1].mean(); fp=100*pred[y==0].mean()
    rows.append((thr,sens,fp))
    print(f"{thr:>7.2f}{sens:>8.1f}%{fp:>8.1f}%{100-fp:>8.1f}%")
print()
for target in (95,93,90,85):
    ok=[r for r in rows if r[1]>=target]
    if ok:
        b=min(ok,key=lambda r:r[2])
        print(f"  sensitivity >= {target}%  ->  thr {b[0]:.2f}   "
              f"sens {b[1]:.1f}%   FP {b[2]:.1f}%")
print("\nCOMPARISON (same 326 studies)")
print("  as it shipped                     sens 97.6%   FP 87.4%")
print("  two rule fixes                    sens 94.1%   FP 52.4%")
print("  candidate judge, any-above-thresh  sens 90.2%   FP 53.6%")
S["p_study"]=p
S.to_csv(f"{HERE}/study_scored.csv",index=False)
json.dump({"study_auc":float(auc)},open(f"{HERE}/study_report.json","w"),indent=2)
