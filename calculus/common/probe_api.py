"""Probe the 5C download API: which studies are actually fetchable, and how fast?

The API prepares a zip server-side and streams it. Studies that take >300 s to
prepare are killed by the gateway with HTTP 504 (observed on older studies,
probably cold storage). This script measures, per study:
    http status, time-to-first-byte, throughput on the first few MB

It aborts each request right after the first bytes arrive, so it costs almost
no bandwidth and tells us the success rate before we commit to a bulk download.

Usage:
    python probe_api.py --n 12 --workers 6
    python probe_api.py --n 30 --workers 8 --out probe_results.csv
"""
import argparse
import os
import sys
import time
import concurrent.futures as cf

import pandas as pd
import requests

# this file lives in utils/, so the project root is one level up. All data
# directories hang off ROOT, never off the script's own folder.
HERE = os.path.dirname(os.path.abspath(__file__))
# package root -> project root: calculus/<sub>/x.py is two levels down
ROOT = os.path.dirname(os.path.dirname(HERE))
API = "https://api.5cnetwork.com/dicom/download/{iuid}"
PROBE_BYTES = 4 << 20      # read this much, then hang up


def probe(rec, timeout):
    iuid = str(rec["study_iuid"]).strip()
    t0 = time.time()
    out = {"study_id": rec["study_id"], "study_iuid": iuid,
           "tier": rec.get("tier", ""), "variant": rec.get("variant", ""),
           "report_created_at": rec.get("report_created_at", "")}
    try:
        with requests.get(API.format(iuid=iuid), stream=True,
                          timeout=(30, timeout)) as r:
            out["http"] = r.status_code
            if r.status_code != 200:
                out["ttfb_s"] = round(time.time() - t0, 1)
                out["note"] = r.text[:100]
                return out
            got = 0
            ttfb = None
            for chunk in r.iter_content(1 << 20):
                if ttfb is None:
                    ttfb = round(time.time() - t0, 1)
                got += len(chunk)
                if got >= PROBE_BYTES:
                    break
            dt = time.time() - t0
            out.update(ttfb_s=ttfb, probe_bytes=got,
                       mbps=round(got / 1e6 / max(dt - (ttfb or 0), .01), 1),
                       note="streaming ok")
    except Exception as e:
        out.update(http=0, ttfb_s=round(time.time() - t0, 1),
                   note=f"{type(e).__name__}: {e}"[:120])
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--worklist",
                    default=os.path.join(ROOT, "csv", "worklist_pilot.csv"))
    ap.add_argument("--n", type=int, default=12)
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--timeout", type=int, default=330,
                    help="read timeout; gateway 504s at ~300 s")
    ap.add_argument("--out", default=os.path.join(ROOT, "csv", "probe_results.csv"))
    args = ap.parse_args()

    wl = pd.read_csv(args.worklist)
    wl["report_created_at"] = pd.to_datetime(wl["report_created_at"], format="mixed",
                                             utc=True)
    # spread the sample across the full date range so we can see if age matters
    wl = wl.sort_values("report_created_at")
    idx = [round(i * (len(wl) - 1) / max(args.n - 1, 1)) for i in range(args.n)]
    sample = wl.iloc[sorted(set(idx))].to_dict("records")

    print(f"probing {len(sample)} studies, {args.workers} concurrent, "
          f"timeout {args.timeout}s\n")
    rows = []
    with cf.ThreadPoolExecutor(max_workers=args.workers) as ex:
        for row in ex.map(lambda r: probe(r, args.timeout), sample):
            rows.append(row)
            print(f"  {row['study_id']}  {str(row.get('report_created_at'))[:10]}  "
                  f"http={row.get('http')}  ttfb={row.get('ttfb_s')}s  "
                  f"{row.get('mbps','-')} MB/s  {row.get('note','')[:60]}",
                  flush=True)

    df = pd.DataFrame(rows)
    df.to_csv(args.out, index=False)
    ok = (df["http"] == 200).sum()
    print(f"\nOK {ok}/{len(df)} ({ok/len(df)*100:.0f}%)  -> {args.out}")
    print("\nby status:")
    print(df["http"].value_counts().to_string())
    if ok:
        good = df[df["http"] == 200]
        print(f"\nsuccessful: median ttfb {good.ttfb_s.median():.0f}s, "
              f"median {good.mbps.median():.1f} MB/s")
    bad = df[df["http"] != 200]
    if len(bad):
        print("\nfailed studies by month:")
        print(bad.groupby(bad["report_created_at"].astype(str).str[:7]).size().to_string())
        print("\nall studies by month:")
        print(df.groupby(df["report_created_at"].astype(str).str[:7]).size().to_string())


if __name__ == "__main__":
    main()
