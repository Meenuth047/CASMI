"""Build work/db/pubchem.parquet from a full PubChem structure dump (SMILES only).

NCBI's FTP throttled this machine to <100 kB/s, which made the three official Extras
dumps (10.2 GB) unreachable, so the structures come from a mirrored full-PubChem
SMILES snapshot instead and every derived field (mass, ik14, element/charge filters)
is computed here with RDKit -- the same functions used for coconut.parquet.

Reads the dump from stdin (one SMILES per line, header tolerated).
"""
import os, sys, re, time, gc
os.environ["CUDA_VISIBLE_DEVICES"] = ""
sys.path.insert(0, "/home/airbotix/Downloads/CASMI")

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from multiprocessing import Pool

DB = "/home/airbotix/Downloads/CASMI/work/db"
OUT = os.environ.get("PC_OUT", f"{DB}/pubchem.parquet")
CHUNK = 20000
T0 = time.time()

ISO = re.compile(r"\[\d+[A-Za-z]")
ALLOWED_Z = frozenset((1, 6, 7, 8, 9, 15, 16, 17, 35, 53))   # H C N O F P S Cl Br I


def log(*a):
    print(f"[{time.time()-T0:7.1f}s]", *a, flush=True)


_c = None


def _init():
    global _c
    from casmi import common
    _c = common


def work(lines):
    """-> (smiles, mass, ik14) for the rows that pass every filter."""
    from rdkit import Chem
    o_s, o_m, o_k = [], [], []
    for s in lines:
        if "." in s or ISO.search(s):
            continue
        mol = _c.mol_from_smiles(s)
        if mol is None:
            continue
        has_c = False
        bad = False
        for a in mol.GetAtoms():
            z = a.GetAtomicNum()
            if z == 6:
                has_c = True
            elif z not in ALLOWED_Z:
                bad = True
                break
            if a.GetIsotope():
                bad = True
                break
        if bad or not has_c:
            continue
        if Chem.GetFormalCharge(mol) != 0:
            continue
        m = _c.exact_mass(mol)
        if not (100.0 <= m <= 1300.0):
            continue
        k = _c.plain_key(mol)
        if not k or len(k) != 14:
            continue
        o_s.append(s)
        o_m.append(m)
        o_k.append(k)
    return o_s, o_m, o_k


def chunks(stream):
    buf = []
    for line in stream:
        s = line.rstrip("\n").rstrip("\r")
        if not s or s.lower() in ("smiles", "canonical_smiles", "smile"):
            continue
        buf.append(s)
        if len(buf) >= CHUNK:
            yield buf
            buf = []
    if buf:
        yield buf


def main():
    smi_parts, mass_parts, ik_parts = [], [], []
    seen = kept = 0
    n_workers = int(os.environ.get("N_WORKERS", 16))
    log(f"using {n_workers} CPU worker processes")
    with Pool(n_workers, initializer=_init) as pool:
        for o_s, o_m, o_k in pool.imap(work, chunks(sys.stdin), chunksize=1):
            seen += CHUNK
            kept += len(o_s)
            smi_parts.append(pa.array(o_s, type=pa.large_string()))
            mass_parts.append(np.asarray(o_m, dtype=np.float64))
            ik_parts.append(np.asarray(o_k, dtype="S14"))
            if len(smi_parts) % 500 == 0:
                log(f"  ~{seen:,} read, {kept:,} kept  ({kept/max(time.time()-T0,1):,.0f} rows/s)")
    log(f"scanned ~{seen:,} lines, {kept:,} passed all filters")

    smi = pa.concat_arrays(smi_parts); del smi_parts
    mass = np.concatenate(mass_parts); del mass_parts
    ik = np.concatenate(ik_parts); del ik_parts
    gc.collect()
    log(f"arrays built: smiles {smi.nbytes/1e9:.2f} GB")

    # Rows are still in dump order (pool.imap preserves it), so a stable sort on ik14
    # puts the first-seen representative of each key first -> take the first of each run.
    order0 = np.argsort(ik, kind="stable")
    ik_sorted = ik[order0]
    run = np.empty(len(order0), bool)
    run[0] = True
    np.not_equal(ik_sorted[1:], ik_sorted[:-1], out=run[1:])
    first = np.sort(order0[run])
    del order0, ik_sorted, run
    log(f"unique ik14: {len(first):,}")

    order = first[np.argsort(mass[first], kind="stable")]
    del first
    f_mass = mass[order]
    f_ik = ik[order]
    del mass, ik
    gc.collect()

    coco = pq.read_table(f"{DB}/coconut.parquet", columns=["ik14"]).column("ik14") \
            .to_numpy(zero_copy_only=False).astype("S14")
    coco.sort()
    f_in = np.isin(f_ik, coco)
    log(f"in_coconut: {int(f_in.sum()):,}")

    schema = pa.schema([("cid", pa.int32()), ("smiles", pa.string()),
                        ("mass", pa.float64()), ("ik14", pa.string()),
                        ("in_coconut", pa.bool_())])
    N = len(order)
    BLK = 2_000_000
    writer = pq.ParquetWriter(OUT, schema, compression="zstd", compression_level=6,
                              write_statistics=True, version="2.6")
    for s in range(0, N, BLK):
        e = min(s + BLK, N)
        blk = pa.table({
            "cid": pa.array(np.full(e - s, -1, dtype=np.int32)),
            "smiles": smi.take(pa.array(order[s:e])).cast(pa.string()),
            "mass": pa.array(f_mass[s:e]),
            "ik14": pa.array([x.decode() for x in f_ik[s:e]], type=pa.string()),
            "in_coconut": pa.array(f_in[s:e]),
        }, schema=schema)
        writer.write_table(blk, row_group_size=100_000)
        log(f"  wrote {e:,}/{N:,}")
    writer.close()
    log(f"DONE {OUT}: {N:,} rows, {os.path.getsize(OUT)/1e9:.2f} GB, "
        f"mass {f_mass[0]:.4f}..{f_mass[-1]:.4f}")


if __name__ == "__main__":
    main()
