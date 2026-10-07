# novelty_scorer: an approximate novelty level (1-4) for protein sequences

**This is NOT the competition's novelty checker.** It is our own re-implementation of the scheme Adaptyv describes publicly (https://www.adaptyvbio.com/blog/novelty). Their own pipeline is not public, so several choices below are our assumptions, and we have **not** validated it against their scorer. Please treat the output as a screen, not as a prediction of the score you will get.

Scope: **single-chain protein sequences only.** Antibody, nanobody, scFv and Fab scoring is deliberately not included, as we have not validated any antibody sequences against the Adaptyv scorer.

**Performance**

I tested ~110 upload of miniprotein sequences that this scorer ranked 3/4 or 4/4:
| Our score | Adaptyv score |
|---|---|
| 3 | ~3% were 2s, ~15% were 4s, and ~82% were 3s |
| 4 | 1/1 was a 4 |


## What it computes
For each sequence:
1. **Sequence similarity.** MMseqs2 search against SwissProt and PDB sequences (`-s 7.5 -e 10 --max-seqs 2000`). For every hit it records the identity (`fident`) and the fraction of your sequence covered (`qcov`).
2. **Structure similarity.** The sequence is folded with ESMFold v1, and the model is searched with Foldseek (TM-align mode) against the PDB and AFDB-SwissProt. The structure score `struct_best_qtm` is the best query-normalised TM-score among hits that cover at least 70% of your sequence (`--cov-min`). The whole chain is treated as one domain (our assumption; Adaptyv uses a consensus of predictors and domain segmentation).
3. **Level.** With sequence similarity `s` and structure TM `t`:

| Level | Meaning | Condition |
|---|---|---|
| 1 | known | `s > 0.70` and `t >= 0.5` |
| 2 | | `s > 0.70`, or `t >= 0.8`, or (`s > 0.30` and `t >= 0.5`) |
| 3 | | `s > 0.30` or `t >= 0.5` |
| 4 | fully de novo | otherwise |

(Rules are applied top to bottom.)

**`level` versus `level_strict`.** Both use the same structure score and the same table. They differ only in `s`:
- `level`: `s` = best identity x query coverage over hits (`seq_ident_x_qcov`). A short local match counts for less than its identity.
- `level_strict`: `s` = the larger of that value and `seq_ident_qcov50`, the best identity among hits covering at least 50% of the sequence, taken at full identity.
Because `s_strict >= s`, `level_strict <= level` always. The identity x coverage convention is our assumption; the strict version is the more conservative reading.

## Output columns
`name, molecule_class, seq_ident, seq_ident_x_qcov, seq_ident_qcov50, seq_best_target, seq_best_qcov, struct_best_qtm, struct_best_target, struct_best_qcov, mean_plddt, level, level_strict`, plus `<out>.meta.json` (arguments, tool and package versions).
The output file is written exactly at the path you give (no `.csv` is added). `seq_best_target` is `swissprot:<id>` or `pdb:<id>`; `struct_best_target` is `pdb:<id>` or `afdb_sp:<AFDB id>`.

## Requirements
- Python 3.12 with the packages in `requirements.txt` (torch, transformers, pandas, numpy, ...). A GPU is recommended for ESMFold; CPU works for a few sequences.
- `mmseqs` (we used release 17-b804f) and `foldseek` (release 10-941cd33) on `PATH`.
- Four reference databases in one directory (`--db-dir`, or `export NOVELTY_DB=...`), with exactly these names: `swissprot_seq`, `pdb_seq` (MMseqs2 databases) and `fs_pdb`, `fs_swissprot` (Foldseek databases). We created ours with the tools' own downloaders:
  ```
  mmseqs   databases UniProtKB/Swiss-Prot swissprot_seq  tmp
  mmseqs   databases PDB                  pdb_seq        tmp
  foldseek databases PDB                  fs_pdb         tmp
  foldseek databases Alphafold/Swiss-Prot fs_swissprot   tmp
  ```
  Versions of the copies behind the numbers below: UniProt Swiss-Prot release 2026_03; MMseqs2 PDB database fetched 2026-10-02; Foldseek `pdb100` (PDB date 2025-01-01); AFDB-SwissProt v6 (2025_03). Results depend on the database versions, so scores from newer databases can differ.
- ESMFold weights (`facebook/esmfold_v1`, about 8 GB) download on first use into the Hugging Face cache (`HF_HOME`).

## Usage
```
python novelty_scorer.py designs.csv scores.csv --db-dir /path/to/db --threads 8
```
Input CSV: columns `name`, `sequence` (and optionally `molecule_class`, which must be `protein`). Names must be unique; sequences must be letters only. Intermediate files go to `scores_work/` (`--workdir` to change); `--skip-fold` reuses structures already in `<workdir>/pdb`. Run on one GPU node; a job-script example for Slurm:
```
module load MMseqs2 Foldseek            # or put the executables on PATH
export HF_HOME=/path/to/hf_cache NOVELTY_DB=/path/to/db
unset PYTHONPATH PYTHONHOME             # avoid package clashes with module-provided Python libraries
python novelty_scorer.py designs.csv scores.csv
```
Practical notes from our runs:
- A run has a fixed start-up cost of about 3-5 minutes (loading ESMFold, opening the databases); 55 designs took about 8 minutes in total on an RTX-class GPU once the weights were cached.
- **First load of the weights on a node can be very slow** if the cache is on a network filesystem (we saw 0.5 MB/s and a stalled job). Reading the weight files once sequentially first fixes it: `dd if=<file in HF_HOME/hub/models--facebook--esmfold_v1/blobs/> of=/dev/null bs=16M`.
- If `import transformers` fails with `regex>=... is required`, a `PYTHONPATH` from another environment is shadowing the package; unset it.
- mmseqs and foldseek need to be on `PATH`; the script exits with a clear message if they are not.

## Check that your installation reproduces our results
```
python novelty_scorer.py tests/test_designs_public.csv my_public.csv --db-dir /path/to/db
python tests/check_reproduction.py my_public.csv tests/expected_public.csv
```
`tests/test_designs_public.csv` has six public or synthetic sequences (EGF, GFP, ubiquitin, scrambled versions of GFP and ubiquitin, and an idealized helix repeat). `tests/expected_public.csv` is the output of our original scorer with our database copies. With **the same database versions** the sequence columns should match exactly and the levels should be identical; with newer databases the exact identities can differ, but the three known proteins should still be level 1, scrambled GFP and the idealized helix level 4, and scrambled ubiquitin is borderline (level 3, strict level 2 for us, because of one 52%-identity local match).

| Sequence | level | level_strict |
|---|---|---|
| EGF_known, GFP_known, ubiquitin_known | 1 | 1 |
| GFP_scrambled | 4 | 4 |
| ubiquitin_scrambled | 3 | 2 |
| idealized_helix_repeat | 4 | 4 |

## How well this package reproduces our earlier results
We ran this package against the original scripts and against scores we had already computed for 55 designs of our own (not distributed here), spanning all four levels (5 at level 1, 12 at level 2, 31 at level 3 and 7 at level 4), in a single GPU job on 2026-10-06 (RTX PRO 6000 Blackwell GPU, same database copies):
- **Public test set (6 sequences), package vs the original script:** identical on every column (zero difference), including levels.
- **55 designs vs existing scores** (computed earlier on other GPUs in separate runs): the three MMseqs2 sequence columns were identical for all 55. `level` matched for 54 of 55 and `level_strict` for 54 of 55. The single difference sat on a threshold: the structure score was 0.509 against 0.495 earlier (threshold 0.5), which moved the design from level 4 to 3. `struct_best_qtm` differed by less than 0.02 for 50 of 55 designs (median difference 0.0001, 54 of 55 within 0.05). One design had a large difference (0.41 against 0.00; level 4 in both runs), which we think is a hit sitting close to the 70% coverage requirement in one run and not in the other, but we did not investigate it. `mean_plddt` differed by at most 1.1.
So the code reproduces the previous implementation, and the remaining variability comes from the structure step (ESMFold numerics on different GPUs, and the coverage threshold of the Foldseek hits). A design whose TM score is near 0.5 or 0.8, or whose best hit covers close to 70% of the sequence, can change level between runs.

## Limitations and validation status
- No domain segmentation and a single structure predictor (ESMFold); no antibody path; reference databases are SwissProt, PDB and AFDB-SwissProt only, so hits in larger databases (UniRef, MGnify, AFDB-wide) are not seen and a sequence can look more novel here than it is.
- The identity x coverage convention and the 70% hit-coverage requirement are our interpretation of Adaptyv's description.

## Credits and licences (please verify before redistributing)
This project is licensed under the MIT License.
It does not include or redistribute MMseqs2, Foldseek, ESMFold or the reference databases, which are covered by their own
licences (see "Credits and licences").

Developed with the assistance of Claude Sonnet 5.5 (Anthropic).
