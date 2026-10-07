#!/usr/bin/env python
"""Fold sequences with ESMFold v1 (HuggingFace transformers) -> PDB files + pLDDT summary.

usage: fold_esmfold.py designs.csv outdir [--seq-col sequence --name-col name]
Writes outdir/<name>.pdb (B-factor = pLDDT 0-100) and outdir/plddt.csv. Existing PDBs are skipped.
Sequences with ':' (VH:VL) are folded as two separate chains joined by a glycine linker is NOT done;
each chain is folded independently as <name>_<k>.pdb (antibody path does not need structure).
"""
import argparse, os, sys
import pandas as pd
import torch
from transformers import AutoTokenizer, EsmForProteinFolding

# Model weights (facebook/esmfold_v1) are downloaded to the Hugging Face cache; set HF_HOME to choose where.


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("csv")
    ap.add_argument("outdir")
    ap.add_argument("--seq-col", default="sequence")
    ap.add_argument("--name-col", default="name")
    ap.add_argument("--chunk", type=int, default=64, help="trunk chunk size (lower = less memory)")
    a = ap.parse_args()
    os.makedirs(a.outdir, exist_ok=True)
    df = pd.read_csv(a.csv)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    tok = AutoTokenizer.from_pretrained("facebook/esmfold_v1")
    model = EsmForProteinFolding.from_pretrained("facebook/esmfold_v1", low_cpu_mem_usage=True).eval().to(dev)
    if dev == "cuda":
        model.esm = model.esm.half()
    model.trunk.set_chunk_size(a.chunk)
    rows = []
    for _, r in df.iterrows():
        chains = str(r[a.seq_col]).strip().upper().split(":")
        for k, seq in enumerate(chains):
            nm = str(r[a.name_col]) + (f"_{k}" if len(chains) > 1 else "")
            pdb = os.path.join(a.outdir, nm + ".pdb")
            if os.path.exists(pdb):
                continue
            seq = seq.translate(str.maketrans("BZJUOX", "AAAAAA"))
            inp = tok([seq], return_tensors="pt", add_special_tokens=False)["input_ids"].to(dev)
            with torch.no_grad():
                out = model(inp)
            plddt = out["plddt"][0].float().cpu().numpy()  # (L, 37) in 0-1 or 0-100
            ca = float(plddt[:, 1].mean())
            scale = 100.0 if ca <= 1.0 + 1e-6 else 1.0
            pdbstr = model.output_to_pdb(out)[0]
            with open(pdb, "w") as f:
                f.write(pdbstr)
            rows.append({"name": nm, "length": len(seq), "mean_plddt": ca * scale})
            print(nm, len(seq), round(ca * scale, 1), flush=True)
    if rows:
        p = os.path.join(a.outdir, "plddt.csv")
        new = pd.DataFrame(rows)
        if os.path.exists(p):
            new = pd.concat([pd.read_csv(p), new]).drop_duplicates("name", keep="last")
        new.to_csv(p, index=False)


if __name__ == "__main__":
    main()
