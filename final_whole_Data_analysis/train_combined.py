#!/usr/bin/env python
"""Rules first, then the judge on what survives.

The judge in train_judge2 was trained on the candidate pool as it existed BEFORE
the two rule fixes, so it spent much of its capacity re-learning "streaks are not
stones" -- a thing now settled by a rejection that costs no sensitivity. Removing
that population first changes the judge's job from "separate stones from streaks
and mimics" to "separate stones from MIMICS", which is the hard part and the only
part left.

Stage 2 (a study-level model over aggregated scores) is dropped: at matched
sensitivity it was worse than taking the maximum candidate score, which is what
326 training studies against 19 aggregate features should have predicted.
"""
import os
for _v in ("OMP_NUM_THREADS","OPENBLAS_NUM_THREADS","MKL_NUM_THREADS","NUMEXPR_NUM_THREADS"):
    os.environ[_v]="2"
import json, numpy as np, pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score, average_precision_score

HERE=os.path.dirname(os.path.abspath(__file__))
DROP={"study_id","stone_id","candidate_id","reject_reason","is_stone","report_this",
      "review_flag","measurement_flag","centroid_vox","shape_source","measure_stage",
      "expected","label","hu_rank_side","oversized_review","fold","p"}

D=pd.read_csv(f"{HERE}/candidates_pooled.csv"); D["study_id"]=D.study_id.astype(str)
# THE DENOMINATOR IS EVERY STUDY, fixed here after getting it wrong. The rules
# strip every candidate from 58 clean studies, which is the rules working: those
# studies are then correctly Normal. Taking the study list AFTER filtering
# dropped them from the negative denominator (153 -> 95) and reported a false
# positive rate computed only over the clean studies that still had something to
# flag. A study that survives with nothing is a true negative and must be
# counted as one.
ALL = D.groupby("study_id").expected.first()
ALL_POS=[s for s in ALL.index if ALL[s]=="abnormal"]
ALL_NEG=[s for s in ALL.index if ALL[s]=="normal"]
before=len(D)
# the committed rules, applied to the pool exactly as the detectors now do
ff=D.fill_fraction.fillna(1.0)
rule_ok=~(((D.comp=="renal")   & (ff<0.05)) |
          ((D.comp=="bladder") & (ff<0.04)))
D=D[rule_ok].copy()
print(f"rule fixes remove {before-len(D)} of {before} candidates "
      f"({100*(before-len(D))/before:.0f}%), leaving {len(D)}")
print(f"  remaining: {int((D.label==1).sum())} pos, {int((D.label==0).sum())} neg, "
      f"{int((D.label==-1).sum())} unlabelled")

num=[c for c in D.columns if c not in DROP and D[c].dtype.kind in "fib"]
cats=[c for c in ("comp","zone","compartment","side","location") if c in D.columns]
X=pd.concat([D[num].astype(float)]+
            [pd.get_dummies(D[c].astype(str),prefix=c) for c in cats],axis=1)
feats=list(X.columns); X=X.astype(float).values
y=D.label.values.astype(int)

studies=list(pd.unique(D.study_id))
rng=np.random.RandomState(0); rng.shuffle(studies)
fold={s:i%5 for i,s in enumerate(studies)}
D["fold"]=D.study_id.map(fold)
p=np.full(len(D),np.nan)
for k in range(5):
    tr=(D.fold.values!=k)&(y>=0); te=D.fold.values==k
    clf=HistGradientBoostingClassifier(max_iter=300,learning_rate=0.06,
        max_leaf_nodes=15,l2_regularization=1.0,min_samples_leaf=20,random_state=0)
    clf.fit(X[tr],y[tr]); p[te]=clf.predict_proba(X[te])[:,1]
D["p"]=p
lab=y>=0
print(f"\ncandidate judge on the filtered pool:  AUC {roc_auc_score(y[lab],p[lab]):.3f}"
      f"   AP {average_precision_score(y[lab],p[lab]):.3f}")

pos, neg = ALL_POS, ALL_NEG
print(f"\nRULES + JUDGE   ({len(pos)} positive / {len(neg)} negative studies)")
print(f"{'thresh':>7}{'sens':>9}{'FP':>9}{'spec':>9}")
rows=[]
for thr in [0.0,0.02,0.05,0.08,0.10,0.15,0.20,0.25,0.30,0.40,0.50,0.60,0.70]:
    fl=set(D[D.p>=thr].study_id)
    s=100*sum(x in fl for x in pos)/len(pos); f=100*sum(x in fl for x in neg)/len(neg)
    rows.append((thr,s,f)); print(f"{thr:>7.2f}{s:>8.1f}%{f:>8.1f}%{100-f:>8.1f}%")
print()
for t in (95,94,92,90,88,85):
    ok=[r for r in rows if r[1]>=t]
    if ok:
        b=min(ok,key=lambda r:r[2])
        print(f"  sens >= {t}%  ->  thr {b[0]:.2f}   sens {b[1]:.1f}%   FP {b[2]:.1f}%")
print("\nWHERE WE STARTED / WHERE WE ARE (same studies)")
print("  as it shipped                sens 97.6%   FP 87.4%")
print("  rules only                   sens 94.1%   FP 52.4%")
print("  judge only (unfiltered pool) sens 90.2%   FP 53.6%")
D.to_csv(f"{HERE}/combined_scored.csv",index=False)
