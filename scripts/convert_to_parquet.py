# Convert NASA C-MAPSS raw .txt files to Parquet
# No preprocessing — raw values only.

import pandas as pd
import os

COL_NAMES = (
    ['unit_id', 'cycle']
    + ['setting_1', 'setting_2', 'setting_3']
    + [f's{i}' for i in range(1, 22)]   # s1 … s21
)

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC_DIR = os.path.join(ROOT, 'data', 'raw')
OUT_DIR = os.path.join(ROOT, 'data', 'processed')
os.makedirs(OUT_DIR, exist_ok=True)

# ── train / test trajectory files ──────────────────────────────────────────
for split in ('train', 'test'):
    for fd in range(1, 5):
        src = os.path.join(SRC_DIR, f'{split}_FD00{fd}.txt')
        if not os.path.exists(src):
            print(f'  SKIP (not found): {src}')
            continue
        df = pd.read_csv(src, sep=r'\s+', header=None, names=COL_NAMES)
        dst = os.path.join(OUT_DIR, f'{split}_FD00{fd}.parquet')
        df.to_parquet(dst, index=False)
        print(f'  {src}  →  {dst}  ({len(df):,} rows, {df["unit_id"].nunique()} units)')

# ── RUL ground-truth files (single column) ─────────────────────────────────
for fd in range(1, 5):
    src = os.path.join(SRC_DIR, f'RUL_FD00{fd}.txt')
    if not os.path.exists(src):
        print(f'  SKIP (not found): {src}')
        continue
    df = pd.read_csv(src, sep=r'\s+', header=None, names=['rul'])
    df.index.name = 'unit_id'          # row i = unit i+1 (0-indexed)
    dst = os.path.join(OUT_DIR, f'RUL_FD00{fd}.parquet')
    df.to_parquet(dst)
    print(f'  {src}  →  {dst}  ({len(df)} units)')

print('\nDone.')
