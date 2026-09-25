"""Fast offline candidate-structure lookup by neutral monoisotopic mass.

The candidate databases (``coconut.parquet``, ``pubchem.parquet``) are parquet files
sorted ascending by the ``mass`` column and written with small row groups, so the
per-row-group min/max statistics in the parquet footer act as a coarse index: a
narrow mass window touches one or two row groups out of hundreds.

``CandidateDB.__init__`` reads only the footer (no data pages), which makes opening a
multi-gigabyte file instant and keeps resident memory at a few hundred kilobytes.

Pure python + numpy + pyarrow + pandas, so the same file runs locally and inside the
Kaggle notebook with the parquet files in a read-only dataset directory.

    db = CandidateDB('/kaggle/input/casmi-db/pubchem.parquet', 'pubchem')
    df = db.query(302.04265 * (1 - 1e-5), 302.04265 * (1 + 1e-5))
    dfs = db.query_many([(lo, hi) for lo, hi in windows], columns=['smiles', 'mass'])
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

__all__ = ["CandidateDB", "MASS_COLUMN"]

MASS_COLUMN = "mass"


class CandidateDB:
    """Mass-window lookup over one candidate parquet file.

    Parameters
    ----------
    path : str
        Path to the parquet file (``coconut.parquet`` / ``pubchem.parquet``).
    name : str
        Value placed in the ``source`` column of every returned frame.
    """

    def __init__(self, path, name):
        self.path = str(path)
        self.name = str(name)
        try:
            self._pf = pq.ParquetFile(self.path, memory_map=True)
        except Exception:
            self._pf = pq.ParquetFile(self.path)

        md = self._pf.metadata
        self.num_rows = md.num_rows
        self.num_row_groups = md.num_row_groups
        self.columns = list(self._pf.schema_arrow.names)
        if MASS_COLUMN not in self.columns:
            raise ValueError(f"{self.path} has no '{MASS_COLUMN}' column")

        self._rg_rows = np.array(
            [md.row_group(i).num_rows for i in range(md.num_row_groups)], dtype=np.int64
        )
        self._rg_min, self._rg_max = self._mass_index(md)
        # Sorted-by-mass files let us binary-search the row-group index instead of
        # scanning it; verified once here rather than assumed.
        self._rg_sorted = bool(
            np.all(np.diff(self._rg_min) >= 0) and np.all(np.diff(self._rg_max) >= 0)
        )

    # ── row-group mass index ──────────────────────────────────────────────
    def _mass_index(self, md):
        """Per-row-group (min, max) of `mass`, from footer statistics when present."""
        n = md.num_row_groups
        lo = np.empty(n, dtype=np.float64)
        hi = np.empty(n, dtype=np.float64)
        col_idx = None
        for j in range(md.num_columns):
            if md.row_group(0).column(j).path_in_schema == MASS_COLUMN:
                col_idx = j
                break
        ok = col_idx is not None
        if ok:
            for i in range(n):
                st = md.row_group(i).column(col_idx).statistics
                if st is None or not st.has_min_max:
                    ok = False
                    break
                lo[i] = st.min
                hi[i] = st.max
        if ok:
            self.stats_source = "footer"
            return lo, hi
        # Fallback: one pass over just the mass column to build the index ourselves.
        self.stats_source = "scan"
        mass = self._pf.read(columns=[MASS_COLUMN]).column(MASS_COLUMN).to_numpy(
            zero_copy_only=False
        )
        start = 0
        for i, k in enumerate(self._rg_rows):
            chunk = mass[start:start + k]
            lo[i] = chunk.min() if k else np.inf
            hi[i] = chunk.max() if k else -np.inf
            start += k
        del mass
        return lo, hi

    def _row_groups_for(self, mass_lo, mass_hi):
        """Indices of the row groups whose [min, max] overlaps [mass_lo, mass_hi]."""
        if mass_hi < mass_lo:
            return np.empty(0, dtype=np.int64)
        if self._rg_sorted:
            first = int(np.searchsorted(self._rg_max, mass_lo, side="left"))
            last = int(np.searchsorted(self._rg_min, mass_hi, side="right"))
            if last <= first:
                return np.empty(0, dtype=np.int64)
            return np.arange(first, last, dtype=np.int64)
        return np.nonzero((self._rg_max >= mass_lo) & (self._rg_min <= mass_hi))[0]

    # ── reading ───────────────────────────────────────────────────────────
    def _resolve_columns(self, columns):
        """-> (columns to read from parquet, columns to return)."""
        if columns is None:
            want = list(self.columns)
        else:
            want = list(columns)
            missing = [c for c in want if c not in self.columns and c != "source"]
            if missing:
                raise KeyError(f"unknown column(s) {missing}; have {self.columns}")
            want = [c for c in want if c != "source"]
        read = want if MASS_COLUMN in want else want + [MASS_COLUMN]
        return read, want

    @staticmethod
    def _slice(tbl, mass_lo, mass_hi):
        """Rows of `tbl` with mass_lo <= mass <= mass_hi (table is mass-sorted inside a row group)."""
        mass = tbl.column(MASS_COLUMN).to_numpy(zero_copy_only=False)
        n = len(mass)
        if n == 0:
            return tbl.slice(0, 0)
        if mass[0] <= mass[-1] and (n < 3 or np.all(np.diff(mass) >= 0)):
            a = int(np.searchsorted(mass, mass_lo, side="left"))
            b = int(np.searchsorted(mass, mass_hi, side="right"))
            return tbl.slice(a, max(b - a, 0))
        sel = np.nonzero((mass >= mass_lo) & (mass <= mass_hi))[0]
        return tbl.take(pa.array(sel))

    def _finish(self, pieces, want):
        if pieces:
            tbl = pa.concat_tables(pieces) if len(pieces) > 1 else pieces[0]
        else:
            tbl = self._pf.schema_arrow.empty_table()
        df = tbl.select([c for c in want if c in tbl.schema.names]).to_pandas()
        df["source"] = self.name
        return df.reset_index(drop=True)

    # ── public API ────────────────────────────────────────────────────────
    def query(self, mass_lo, mass_hi, columns=None):
        """All rows with ``mass_lo <= mass <= mass_hi``, plus a ``source`` column."""
        read_cols, want = self._resolve_columns(columns)
        pieces = []
        for i in self._row_groups_for(mass_lo, mass_hi):
            part = self._slice(self._pf.read_row_group(int(i), columns=read_cols),
                               mass_lo, mass_hi)
            if part.num_rows:
                pieces.append(part)
        return self._finish(pieces, want)

    def query_many(self, windows, columns=None):
        """``query`` for many windows, reading every needed row group exactly once.

        Windows are grouped by row group, so 1500 narrow windows cost at most 1500
        row-group reads instead of one full pass per window.
        """
        read_cols, want = self._resolve_columns(columns)
        windows = list(windows)
        rg_to_windows: dict[int, list[int]] = {}
        per_window: list[list] = [[] for _ in windows]
        for w, (lo, hi) in enumerate(windows):
            for i in self._row_groups_for(lo, hi):
                rg_to_windows.setdefault(int(i), []).append(w)
        for i in sorted(rg_to_windows):
            tbl = self._pf.read_row_group(i, columns=read_cols)
            for w in rg_to_windows[i]:
                lo, hi = windows[w]
                part = self._slice(tbl, lo, hi)
                if part.num_rows:
                    per_window[w].append(part)
            del tbl
        return [self._finish(pieces, want) for pieces in per_window]

    # ── convenience ───────────────────────────────────────────────────────
    def query_ppm(self, mass, ppm=10.0, columns=None):
        d = mass * ppm * 1e-6
        return self.query(mass - d, mass + d, columns=columns)

    def counts_many(self, windows):
        """Number of rows in each window, without materialising the rows."""
        return [len(df) for df in self.query_many(windows, columns=[MASS_COLUMN])]

    def __repr__(self):
        return (f"CandidateDB(name={self.name!r}, rows={self.num_rows:,}, "
                f"row_groups={self.num_row_groups}, mass="
                f"{self._rg_min.min():.3f}..{self._rg_max.max():.3f}, "
                f"stats={self.stats_source})")
