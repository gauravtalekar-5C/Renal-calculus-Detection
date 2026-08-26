"""Diff two runs of the same cohort, case by case. Read-only.

WHY
---
Seven fixes went in between the first validation run and the second. A summary
number ("recall improved") hides which case moved and why, and a fix that helps
four studies while quietly breaking a fifth looks identical to a fix that helps
four. So this prints the per-case delta and, for every study whose output
changed, what changed.

It reports in BOTH directions on purpose. A row where the new run finds FEWER
calculi is not automatically an improvement -- it is an improvement if the
removed rows were the pelvic-mimic false positives, and a regression if a real
stone went with them. Only the report text can settle that, so the report's own
findings are printed alongside.

Usage:
    python -m calculus.evaluate.before_after --before case_analysis \\
                                             --after final_check_deployment
"""
import argparse
import os

import pandas as pd


def load(run):
    """Per-study calculus tables from one run, keyed by study id."""
    out = {}
    d = os.path.join(run, "reports")
    if not os.path.isdir(d):
        return out
    for f in sorted(os.listdir(d)):
        if not f.endswith("_calculi.csv"):
            continue
        sid = f[:-len("_calculi.csv")]
        # reports/ also holds the cohort roll-ups all_calculi.csv etc, which
        # would otherwise appear as a study called "all"
        if sid.startswith("all") or not sid.isdigit():
            continue
        try:
            t = pd.read_csv(os.path.join(d, f))
        except Exception:
            continue
        out[sid] = t
    return out


def summ(run, sid):
    """Selected counters that explain a delta."""
    keep = ("n_branched_kept_whole", "n_fragments_merged", "n_touching_split")
    got = {}
    for name in (f"{sid}_summary.csv", f"{sid}_ureter_summary.csv"):
        p = os.path.join(run, "csv", "per_study", name)
        if not os.path.exists(p):
            continue
        try:
            t = pd.read_csv(p)
        except Exception:
            continue
        if not len(t):
            continue
        pre = "k." if "ureter" not in name else "u."
        for k in keep:
            if k in t.columns and pd.notna(t.iloc[0][k]):
                got[pre + k] = int(t.iloc[0][k])
    return got


def verdicts(run, sid):
    """Rejection tallies, so a delta can be traced to a specific rule."""
    p = os.path.join(run, "csv", "per_study", f"{sid}_ureter_candidates.csv")
    if not os.path.exists(p):
        return {}
    try:
        d = pd.read_csv(p)
    except Exception:
        return {}
    if not len(d):
        return {}
    rr = d.reject_reason.fillna("ACCEPTED").astype(str).str.strip()
    return rr.replace("", "ACCEPTED").value_counts().to_dict()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--before", required=True)
    ap.add_argument("--after", required=True)
    ap.add_argument("--cases", default="validation_cases.csv")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()

    B, A = load(a.before), load(a.after)
    cases = pd.read_csv(a.cases) if os.path.exists(a.cases) else pd.DataFrame()
    meta = {}
    if len(cases):
        for r in cases.itertuples():
            meta[str(r.study_id)] = (r.category, r.pick, r.finding_reported)

    rows = []
    for sid in sorted(set(B) | set(A)):
        nb = len(B[sid]) if sid in B else None
        na = len(A[sid]) if sid in A else None
        cat, pick, rep = meta.get(sid, ("", "", ""))
        rows.append({"study_id": sid, "category": cat, "pick": pick,
                     "n_before": nb, "n_after": na,
                     "delta": (None if nb is None or na is None else na - nb)})
    t = pd.DataFrame(rows)

    print("=" * 92)
    print(f"BEFORE  {a.before}          AFTER  {a.after}")
    print("=" * 92)
    print(t.to_string(index=False))

    print("\n" + "=" * 92)
    print("PER-CASE DETAIL for every study whose output changed")
    print("=" * 92)
    for r in t.itertuples():
        sid = r.study_id
        if sid not in B or sid not in A:
            print(f"\n{sid}  {r.category} -- present in only one run")
            continue
        b, aa = B[sid], A[sid]
        same = (len(b) == len(aa)
                and b.to_csv(index=False) == aa.to_csv(index=False))
        if same:
            continue
        cat, pick, rep = meta.get(sid, ("", "", ""))
        print(f"\n{'-'*92}\n{sid}   {cat} ({pick})   {len(b)} -> {len(aa)} calculi")
        if rep:
            print("  REPORT:")
            for c in str(rep).split("|"):
                c = c.strip()
                if c and c.lower() != "nan":
                    print(f"    - {c[:110]}")
        cols = [c for c in ("Organ", "Side", "Size (in mm)", "Density (HU)",
                            "Location") if c in b.columns and c in aa.columns]
        print("  BEFORE:")
        print("    " + (b[cols].to_string(index=False).replace("\n", "\n    ")
                        if len(b) else "(none)"))
        print("  AFTER:")
        print("    " + (aa[cols].to_string(index=False).replace("\n", "\n    ")
                        if len(aa) else "(none)"))
        sb, sa = summ(a.before, sid), summ(a.after, sid)
        moved = {k: (sb.get(k), sa.get(k)) for k in set(sb) | set(sa)
                 if sb.get(k) != sa.get(k)}
        if moved:
            print("  counters that moved: "
                  + ", ".join(f"{k} {v[0]}->{v[1]}" for k, v in sorted(moved.items())))
        vb, va = verdicts(a.before, sid), verdicts(a.after, sid)
        vm = {k: (vb.get(k, 0), va.get(k, 0)) for k in set(vb) | set(va)
              if vb.get(k, 0) != va.get(k, 0)}
        if vm:
            print("  ureteric verdicts that moved: "
                  + ", ".join(f"{k} {v[0]}->{v[1]}" for k, v in sorted(vm.items())))

    print("\n" + "=" * 92)
    ch = t.dropna(subset=["delta"])
    print(f"studies compared          {len(ch)}")
    print(f"unchanged count           {int((ch.delta == 0).sum())}")
    print(f"fewer calculi reported    {int((ch.delta < 0).sum())}")
    print(f"more calculi reported     {int((ch.delta > 0).sum())}")
    print("\nA smaller count is an improvement only if what disappeared was a")
    print("false positive. Read the per-case detail above against the report text;")
    print("the direction alone does not settle it.")

    if a.out:
        t.to_csv(a.out, index=False)
        print(f"\nwrote {a.out}")


if __name__ == "__main__":
    main()
