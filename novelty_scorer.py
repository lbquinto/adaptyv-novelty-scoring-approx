#!/usr/bin/env python
"""Approximate re-implementation of Adaptyv's 1-4 protein novelty levels (protein sequences only).

NOT the competition's checker. Based on the public description at https://www.adaptyvbio.com/blog/novelty (their own pipeline is not public).
Several choices are our assumptions (marked ASSUMPTION below). Antibody/nanobody/scFv/Fab scoring is NOT included in this package.

Input CSV columns: name, sequence [, molecule_class]  (molecule_class, if present, must be "protein")
Output CSV (written to exactly the path you give): one row per design with the evidence behind the level
  seq_ident            best MMseqs2 identity of any hit (SwissProt, PDB seqres)
  seq_ident_x_qcov     best identity x query coverage over hits                  <- sequence similarity used for `level`
  seq_ident_qcov50     best identity among hits covering >= 50% of the query
  struct_best_qtm      best query-normalised TM-score (Foldseek, ESMFold model vs PDB and AFDB-SwissProt), hits covering >= cov_min of the query
  mean_plddt           mean pLDDT of the ESMFold model
  level                1 (known) .. 4 (fully de novo)
  level_strict         same rule, with sequence similarity = max(seq_ident_x_qcov, seq_ident_qcov50); level_strict <= level always

Level rule (s = sequence similarity, t = structure TM):  1: s>0.70 and t>=0.5 | 2: s>0.70 or t>=0.8 or (s>0.30 and t>=0.5) | 3: s>0.30 or t>=0.5 | 4: otherwise
Requires `mmseqs` and `foldseek` on PATH, a GPU for ESMFold (CPU works for a few sequences) and the four reference databases in --db-dir
(swissprot_seq, pdb_seq for MMseqs2; fs_pdb, fs_swissprot for Foldseek; see README.md).
"""
import argparse, json, os, shlex, shutil, subprocess, sys
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))


def sh(cmd):
    print("+", cmd, flush=True)
    subprocess.run(cmd, shell=True, check=True)


# ---------------------------------------------------------------- sequence
def write_fasta(path, names, seqs):
    with open(path, "w") as f:
        for n, s in zip(names, seqs):
            f.write(f">{n}\n{s}\n")


def mmseqs_search(q_fa, target, out, tmp, sens=7.5, threads=8):
    fmt = "query,target,fident,alnlen,qcov,tcov,evalue,bits,qlen"
    sh(f"mmseqs easy-search {shlex.quote(q_fa)} {shlex.quote(target)} {shlex.quote(out)} {shlex.quote(tmp)} -s {sens} -e 10 --max-seqs 2000 "
       f"--threads {threads} --format-output {fmt}")
    cols = fmt.split(",")
    if os.path.getsize(out) == 0:
        return pd.DataFrame(columns=cols)
    return pd.read_csv(out, sep="\t", names=cols)


def best_seq_sim(hits, name):
    """Return dict of best identity under several definitions for query `name`."""
    h = hits[hits["query"] == name]
    if h.empty:
        return dict(seq_ident=0.0, seq_ident_x_qcov=0.0, seq_ident_qcov50=0.0, seq_best_target="", seq_best_qcov=0.0)
    ixc = h.fident * h.qcov
    i = ixc.idxmax()
    q50 = h[h.qcov >= 0.5].fident.max() if (h.qcov >= 0.5).any() else 0.0
    return dict(seq_ident=float(h.fident.max()), seq_ident_x_qcov=float(ixc.max()),
                seq_ident_qcov50=float(q50), seq_best_target=h.loc[i, "target"],
                seq_best_qcov=float(h.loc[i, "qcov"]))


# ---------------------------------------------------------------- structure
def foldseek_search(pdb_dir, db, out, tmp, threads=8):
    fmt = "query,target,qcov,tcov,alntmscore,qtmscore,ttmscore,fident,evalue,alnlen"
    sh(f"foldseek easy-search {shlex.quote(pdb_dir)} {shlex.quote(db)} {shlex.quote(out)} {shlex.quote(tmp)} --alignment-type 1 -e 10 --max-seqs 2000 "
       f"--threads {threads} --format-output {fmt}")
    cols = fmt.split(",")
    if os.path.getsize(out) == 0:
        return pd.DataFrame(columns=cols)
    d = pd.read_csv(out, sep="\t", names=cols)
    d["query"] = d["query"].str.replace(r"\.pdb$", "", regex=True)
    return d


def struct_summary(hits, name, cov_min):
    h = hits[hits["query"] == name]
    # ASSUMPTION: no domain segmentation (Adaptyv uses consensus of 3 predictors). Whole chain = one domain.
    # TM normalised by query length (qtmscore) already folds coverage in; we additionally require qcov >= cov_min.
    h = h[h.qcov >= cov_min]
    if h.empty:
        return dict(struct_best_qtm=0.0, struct_best_target="", struct_best_qcov=0.0)
    i = h.qtmscore.idxmax()
    return dict(struct_best_qtm=float(h.qtmscore.max()), struct_best_target=h.loc[i, "target"],
                struct_best_qcov=float(h.loc[i, "qcov"]))


def protein_level(seq_sim, qtm, tm_high=0.8, tm_mod=0.5, seq_mid=0.30, seq_hi=0.70):
    """Level table from the Adaptyv blog. Returns 1..4."""
    s_hi, s_mid = seq_sim > seq_hi, seq_sim > seq_mid
    st_high, st_mod = qtm >= tm_high, qtm >= tm_mod  # high implies moderate
    if s_hi and st_mod:
        return 1
    if s_hi or st_high or (s_mid and st_mod):
        return 2
    if s_mid or st_mod:
        return 3
    return 4


def tool_version(exe):
    """Informational only. Some builds print a placeholder instead of a version, so the executable path is recorded as well."""
    try:
        v = subprocess.run([exe, "version"], capture_output=True, text=True, timeout=60).stdout.strip().splitlines()
        v = v[0] if v else ""
    except Exception as e:
        v = f"unknown ({e})"
    return f"{v} [{shutil.which(exe)}]" if (not v or "NOTFOUND" in v) else f"{v} [{shutil.which(exe)}]"


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("csv", help="input: name,sequence[,molecule_class]")
    ap.add_argument("out", help="output CSV path (written exactly as given)")
    ap.add_argument("--db-dir", default=os.environ.get("NOVELTY_DB", "db"),
                    help="directory with swissprot_seq, pdb_seq, fs_pdb, fs_swissprot (default: $NOVELTY_DB or ./db)")
    ap.add_argument("--workdir", default=None, help="scratch/intermediate files (default: <out without extension>_work)")
    ap.add_argument("--threads", type=int, default=8)
    ap.add_argument("--skip-fold", action="store_true", help="reuse PDBs already in workdir/pdb")
    ap.add_argument("--cov-min", type=float, default=0.7, help="min query coverage of a structural hit (Adaptyv blog: >70%%)")
    a = ap.parse_args()

    for exe in ("mmseqs", "foldseek"):
        if shutil.which(exe) is None:
            sys.exit(f"ERROR: `{exe}` not found on PATH (load/install it first; see README.md)")
    db = os.path.abspath(a.db_dir)
    for d in ("swissprot_seq", "pdb_seq", "fs_pdb", "fs_swissprot"):
        if not os.path.exists(os.path.join(db, d + ".dbtype")) and not os.path.exists(os.path.join(db, d)):
            sys.exit(f"ERROR: reference database `{d}` not found in {db} (see README.md, section 'Reference databases')")

    work = os.path.abspath(a.workdir or (os.path.splitext(a.out)[0] + "_work"))
    os.makedirs(f"{work}/pdb", exist_ok=True)
    tmp = f"{work}/tmp"
    shutil.rmtree(tmp, ignore_errors=True)
    os.makedirs(tmp)
    df = pd.read_csv(a.csv)
    if "molecule_class" not in df:
        df["molecule_class"] = "protein"
    df["molecule_class"] = df["molecule_class"].fillna("protein").str.lower()
    bad = df[df.molecule_class != "protein"]
    if len(bad):
        sys.exit(f"ERROR: only molecule_class 'protein' is supported in this package ({len(bad)} other rows, e.g. {bad.name.iloc[0]}); "
                 "antibody scoring is not included and has not been validated")
    df["sequence"] = df["sequence"].str.upper().str.replace(r"\s", "", regex=True)
    if not df.sequence.str.fullmatch(r"[A-Z]+").all():
        sys.exit("ERROR: sequences must contain only letters (single chain; no colon separators or other characters)")
    if df["name"].duplicated().any():
        sys.exit("ERROR: duplicate names in the input")
    prot = df
    res = {n: {"name": n} for n in df.name}

    fa = f"{work}/prot.fasta"
    write_fasta(fa, prot.name, prot.sequence.str.translate(str.maketrans("BZJUO", "XXXXX")))
    hits = [mmseqs_search(fa, f"{db}/swissprot_seq", f"{work}/mm_sp.m8", f"{tmp}/a", threads=a.threads),
            mmseqs_search(fa, f"{db}/pdb_seq", f"{work}/mm_pdb.m8", f"{tmp}/b", threads=a.threads)]
    for k, h in zip(("swissprot", "pdb"), hits):
        h["target"] = k + ":" + h["target"].astype(str)
    hits = pd.concat(hits, ignore_index=True)
    if not a.skip_fold:
        sh(f"{shlex.quote(sys.executable)} {shlex.quote(os.path.join(HERE, 'fold_esmfold.py'))} {shlex.quote(a.csv)} {shlex.quote(work + '/pdb')} --name-col name")
    sd = [foldseek_search(f"{work}/pdb", f"{db}/{d}", f"{work}/fs_{d}.m8", f"{tmp}/c{d}", threads=a.threads) for d in ("fs_pdb", "fs_swissprot")]
    for k, h in zip(("pdb", "afdb_sp"), sd):
        h["target"] = k + ":" + h["target"].astype(str)
    sd = pd.concat(sd, ignore_index=True)
    pl = pd.read_csv(f"{work}/pdb/plddt.csv").set_index("name").mean_plddt if os.path.exists(f"{work}/pdb/plddt.csv") else {}
    for n in prot.name:
        r = res[n]
        r.update(best_seq_sim(hits, n))
        r.update(struct_summary(sd, n, a.cov_min))
        r["mean_plddt"] = float(pl.get(n, np.nan)) if len(pl) else np.nan
        # ASSUMPTION: sequence similarity = identity x query coverage (our own convention);
        # the stricter alternative (identity of any hit covering >=50%) is reported as level_strict.
        r["level"] = protein_level(r["seq_ident_x_qcov"], r["struct_best_qtm"])
        r["level_strict"] = protein_level(max(r["seq_ident_x_qcov"], r["seq_ident_qcov50"]), r["struct_best_qtm"])

    out = df[["name", "molecule_class"]].merge(pd.DataFrame(res.values()), on="name")
    out.to_csv(a.out, index=False)
    meta = dict(argv=sys.argv, cov_min=a.cov_min, n_designs=len(out), db_dir=db, mmseqs=tool_version("mmseqs"),
                foldseek=tool_version("foldseek"), python=sys.version.split()[0])
    for m in ("torch", "transformers", "pandas", "numpy"):
        try:
            meta[m] = __import__(m).__version__
        except Exception:
            pass
    with open(os.path.splitext(a.out)[0] + ".meta.json", "w") as f:
        json.dump(meta, f, indent=2)
    print(out.to_string(max_colwidth=30))


if __name__ == "__main__":
    main()
