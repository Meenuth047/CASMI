"""Build work/db/pubchem.parquet by streaming-joining the three PubChem Extras dumps.

Stage A  CID-Mass.gz    -> (cid, mass) for rows passing the mass/formula filters
Stage B  CID-SMILES.gz  -> smiles for the cids of stage A (isotope/multi-component dropped)
Stage C  CID-InChI-Key.gz -> ik14 for the cids of stage A
then: intersect, dedupe on ik14 keeping the lowest cid, sort by mass, flag in_coconut, write.

All three dumps are sorted by CID, so every stage is a single forward scan and the
join is a np.searchsorted lookup into stage A's ascending cid array.
"""
import os, sys, time, gc, subprocess
os.environ["CUDA_VISIBLE_DEVICES"] = ""

import numpy as np
import pyarrow as pa
import pyarrow.csv as pacsv
import pyarrow.parquet as pq

DB = "/home/airbotix/Downloads/CASMI/work/db"
TMP = os.environ.get("PC_TMP", f"{DB}/tmp")
FILTERS = "/tmp/claude-1000/-home-airbotix-Downloads-CASMI/05c1e614-573f-40e8-ab30-eeab25201d4b/scratchpad/filters.sh"
OUT = os.environ.get("PC_OUT", f"{DB}/pubchem.parquet")
T0 = time.time()


DELETE_CONSUMED = os.environ.get("PC_DELETE") == "1"


def log(*a):
    print(f"[{time.time()-T0:7.1f}s]", *a, flush=True)


def drop_parts(prefix):
    """Free a consumed dump's byte-range parts (disk on this box is tight)."""
    if not DELETE_CONSUMED:
        return
    import glob
    freed = 0
    for p in glob.glob(prefix + ".part.*"):
        freed += os.path.getsize(p)
        os.remove(p)
    log(f"freed {freed/1e9:.2f} GB from {os.path.basename(prefix)}")


def stream(mode, prefix, names, types):
    cmd = ["nice", "-n", "10", "bash", FILTERS, mode, prefix]
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, bufsize=1 << 22)
    reader = pacsv.open_csv(
        proc.stdout,
        read_options=pacsv.ReadOptions(use_threads=True, block_size=1 << 25, column_names=names),
        parse_options=pacsv.ParseOptions(delimiter="\t", quote_char=False, escape_char=False,
                                         newlines_in_values=False),
        convert_options=pacsv.ConvertOptions(column_types=types),
    )
    try:
        for batch in reader:
            yield batch
    finally:
        try:
            proc.stdout.close()
        except Exception:
            pass
        proc.wait()
        if proc.returncode not in (0, None):
            raise RuntimeError(f"filter stage {mode} exited {proc.returncode}")


# ── Stage A: cid + mass ────────────────────────────────────────────────────
cid_parts, mass_parts, n = [], [], 0
for b in stream("mass", f"{TMP}/CID-Mass.gz", ["cid", "mass"],
                {"cid": pa.int64(), "mass": pa.float64()}):
    cid_parts.append(b.column(0).to_numpy(zero_copy_only=False).astype(np.int64))
    mass_parts.append(b.column(1).to_numpy(zero_copy_only=False))
    n += b.num_rows
    if len(cid_parts) % 200 == 0:
        log(f"  A {n:,}")
A_cid = np.concatenate(cid_parts); del cid_parts
A_mass = np.concatenate(mass_parts); del mass_parts
gc.collect()
assert np.all(np.diff(A_cid) > 0), "CID-Mass is not strictly ascending by cid"
log(f"A: {len(A_cid):,} cids pass mass+formula filters; cid max {A_cid[-1]:,}")
drop_parts(f"{TMP}/CID-Mass.gz")


def join_stage(mode, prefix, names, types, value_fn):
    """Scan a dump, keep only rows whose cid is in A_cid; returns (positions, values)."""
    pos_parts, val_parts, seen, kept = [], [], 0, 0
    last = -1
    for b in stream(mode, prefix, names, types):
        c = b.column(0).to_numpy(zero_copy_only=False).astype(np.int64)
        if len(c) and c[0] <= last:
            raise RuntimeError(f"{mode} not ascending by cid")
        last = c[-1] if len(c) else last
        idx = np.searchsorted(A_cid, c)
        np.clip(idx, 0, len(A_cid) - 1, out=idx)
        hit = A_cid[idx] == c
        seen += len(c)
        if hit.any():
            sel = np.nonzero(hit)[0]
            pos_parts.append(idx[sel])
            val_parts.append(value_fn(b.column(1), sel))
            kept += len(sel)
        if len(pos_parts) % 200 == 0:
            log(f"  {mode} scanned {seen:,} kept {kept:,}")
    log(f"{mode}: scanned {seen:,}, kept {kept:,}")
    return np.concatenate(pos_parts), val_parts


# ── Stage B: smiles ────────────────────────────────────────────────────────
def take_smiles(col, sel):
    return col.take(pa.array(sel)).cast(pa.large_string())


pos_s, smi_parts = join_stage("smiles", f"{TMP}/CID-SMILES.gz", ["cid", "smiles"],
                              {"cid": pa.int64(), "smiles": pa.large_string()}, take_smiles)
assert np.all(np.diff(pos_s) > 0), "smiles positions not strictly ascending"
smi = pa.concat_arrays([a.combine_chunks() if isinstance(a, pa.ChunkedArray) else a
                        for a in smi_parts])
del smi_parts
gc.collect()
log(f"B: smiles array {len(smi):,}, {smi.nbytes/1e9:.2f} GB")
drop_parts(f"{TMP}/CID-SMILES.gz")


# ── Stage C: ik14 ──────────────────────────────────────────────────────────
def take_ik(col, sel):
    return col.take(pa.array(sel)).to_numpy(zero_copy_only=False).astype("S14")


pos_k, ik_parts = join_stage("ik", f"{TMP}/CID-InChI-Key.gz", ["cid", "ik14"],
                             {"cid": pa.int64(), "ik14": pa.string()}, take_ik)
assert np.all(np.diff(pos_k) > 0), "ik positions not strictly ascending"
ik_all = np.concatenate(ik_parts); del ik_parts
gc.collect()
log(f"C: ik14 array {len(ik_all):,}")
drop_parts(f"{TMP}/CID-InChI-Key.gz")


# ── Assemble ───────────────────────────────────────────────────────────────
have_s = np.zeros(len(A_cid), bool); have_s[pos_s] = True
have_k = np.zeros(len(A_cid), bool); have_k[pos_k] = True
keep = have_s & have_k
kept_pos = np.nonzero(keep)[0]
del have_s, have_k, keep
log(f"rows with mass+smiles+ik14: {len(kept_pos):,}")

si = np.searchsorted(pos_s, kept_pos)
ki = np.searchsorted(pos_k, kept_pos)
del pos_s, pos_k
ik_keep = ik_all[ki]; del ik_all, ki
assert A_cid[-1] < 2 ** 31, "cid overflows int32"
cid_keep = A_cid[kept_pos].astype(np.int32)
mass_keep = A_mass[kept_pos]
del A_cid, A_mass, kept_pos
gc.collect()

# Rows are in ascending cid order. A *stable* sort on ik14 therefore leaves, within each
# group of equal keys, the lowest cid first -> take the first of every run.
srt = np.argsort(ik_keep, kind="stable")
ik_sorted = ik_keep[srt]
run_start = np.empty(len(srt), bool)
run_start[0] = True
np.not_equal(ik_sorted[1:], ik_sorted[:-1], out=run_start[1:])
first = np.sort(srt[run_start])   # back to ascending cid order -> deterministic tie order on mass
del srt, ik_sorted, run_start
assert np.all(np.diff(cid_keep.astype(np.int64)) > 0), "kept rows are not in ascending cid order"
assert len(np.unique(ik_keep[first])) == len(first), "ik14 not unique after dedupe"
log(f"unique ik14: {len(first):,}")

order = first[np.argsort(mass_keep[first], kind="stable")]
del first
f_cid = cid_keep[order]; f_mass = mass_keep[order]; f_ik = ik_keep[order]
f_smi = si[order]          # kept-row index -> position in the smiles stream
del cid_keep, mass_keep, ik_keep, si, order
gc.collect()

coco = pq.read_table(f"{DB}/coconut.parquet", columns=["ik14"]).column("ik14") \
        .to_numpy(zero_copy_only=False).astype("S14")
coco.sort()
f_in_coco = np.isin(f_ik, coco, assume_unique=False)
log(f"in_coconut: {int(f_in_coco.sum()):,}")

schema = pa.schema([("cid", pa.int32()), ("smiles", pa.string()),
                    ("mass", pa.float64()), ("ik14", pa.string()),
                    ("in_coconut", pa.bool_())])
RG = 100_000
BLK = 2_000_000
writer = pq.ParquetWriter(OUT, schema, compression="zstd", compression_level=6,
                          write_statistics=True, version="2.6")
N = len(f_cid)
for s in range(0, N, BLK):
    e = min(s + BLK, N)
    blk = pa.table({
        "cid": pa.array(f_cid[s:e]),
        "smiles": smi.take(pa.array(f_smi[s:e])).cast(pa.string()),
        "mass": pa.array(f_mass[s:e]),
        "ik14": pa.array([x.decode() for x in f_ik[s:e]], type=pa.string()),
        "in_coconut": pa.array(f_in_coco[s:e]),
    }, schema=schema)
    writer.write_table(blk, row_group_size=RG)
    log(f"  wrote {e:,}/{N:,}")
writer.close()
log(f"DONE {OUT}: {N:,} rows, {os.path.getsize(OUT)/1e9:.2f} GB, "
    f"mass {f_mass[0]:.4f}..{f_mass[-1]:.4f}")
