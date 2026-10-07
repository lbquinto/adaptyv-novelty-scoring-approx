#!/usr/bin/env python
"""Compare a scorer output CSV with an expected/reference CSV (matched on `name`).

usage: python check_reproduction.py new.csv expected.csv [--tm-tol 0.02] [--plddt-tol 2.0]

Levels must match exactly. Sequence columns (MMseqs2) must agree to 1e-4 if the same reference databases are used. Structure columns depend on
ESMFold (GPU, half precision) and Foldseek, so struct_best_qtm is compared with a tolerance and mean_plddt with a looser one.
For rows whose level differs, the margin to the nearest level threshold is printed (a design right at a threshold can flip with tiny changes).
Exit code 0 if all levels match, 1 otherwise.
"""
import argparse, sys
import numpy as np
import pandas as pd

ap = argparse.ArgumentParser()
ap.add_argument("new"); ap.add_argument("expected")
ap.add_argument("--tm-tol", type=float, default=0.02); ap.add_argument("--plddt-tol", type=float, default=2.0)
a = ap.parse_args()
n = pd.read_csv(a.new).set_index("name"); e = pd.read_csv(a.expected).set_index("name")
common = n.index.intersection(e.index)
print(f"{len(common)} designs compared ({len(n) - len(common)} only in new, {len(e) - len(common)} only in expected)")
bad_levels = []
rep = []
for col, tol in (("seq_ident", 1e-4), ("seq_ident_x_qcov", 1e-4), ("seq_ident_qcov50", 1e-4), ("struct_best_qtm", a.tm_tol), ("mean_plddt", a.plddt_tol)):
    d = (n.loc[common, col] - e.loc[common, col]).abs()
    rep.append((col, tol, int((d <= tol).sum()), float(d.max()), float(d.median())))
for col in ("level", "level_strict"):
    ok = n.loc[common, col] == e.loc[common, col]
    rep.append((col, 0, int(ok.sum()), float((~ok).sum()), np.nan))
    if col == "level":
        bad_levels = list(common[~ok.values])
print(f"{'column':20s} {'tolerance':>9s} {'match':>9s} {'max |diff|':>11s} {'median |diff|':>14s}")
for col, tol, k, mx, md in rep:
    print(f"{col:20s} {tol:>9} {k:>4d}/{len(common):<4d} {mx:>11.4g} {md:>14.4g}")
same_t = (n.loc[common, "seq_best_target"].fillna("") == e.loc[common, "seq_best_target"].fillna("")).mean()
print(f"best sequence hit identical for {same_t:.0%} of designs; best structure hit identical for "
      f"{(n.loc[common, 'struct_best_target'].fillna('') == e.loc[common, 'struct_best_target'].fillna('')).mean():.0%} (ties between near-identical database entries can differ)")
if bad_levels:
    print("\nLevel differs for:")
    for nm in bad_levels:
        r, q = n.loc[nm], e.loc[nm]
        print(f"  {nm}: level new {int(r.level)} vs expected {int(q.level)}; s_new={r.seq_ident_x_qcov:.3f} s_exp={q.seq_ident_x_qcov:.3f}; "
              f"TM_new={r.struct_best_qtm:.3f} TM_exp={q.struct_best_qtm:.3f}  (thresholds: s 0.30/0.70, TM 0.5/0.8)")
sys.exit(1 if bad_levels else 0)
