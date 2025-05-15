import glob
import re
import pandas as pd
from pathlib import Path
import pickle
from concurrent.futures import ProcessPoolExecutor
from typing import List

wpe_path = "win_probs.csv"  # adjust as needed
wpe_df = pd.read_csv(wpe_path)
wpe_df = wpe_df.sort_values(['MATCH_ID', 'TIMESTAMP'])

for side in ['BLUE', 'RED']:
    win_col    = f'{side}_WIN'
    reward_col = f'{side}_REWARD'
    wpe_df[reward_col] = (
        wpe_df
          .groupby('MATCH_ID')[win_col]
          .apply(lambda s: s.shift(-1) - s)
          .reset_index(level=0, drop=True)
    )

wpe_rewards = wpe_df[['MATCH_ID', 'TIMESTAMP', 'BLUE_REWARD', 'RED_REWARD']]

def _load_and_combine(idx: int, map_X: dict, map_Y: dict, wpe_rewards: pd.DataFrame) -> pd.DataFrame:
    print(f"loading index: {idx}")
    df_X = pd.read_csv(map_X[idx])
    # merge in rewards if missing
    if not {'BLUE_REWARD','RED_REWARD'}.issubset(df_X.columns):
        df_X = df_X.merge(wpe_rewards, on=['MATCH_ID','TIMESTAMP'], how='left')
    # load Y and drop its MATCH_ID
    df_Y = pd.read_csv(map_Y[idx]).drop(columns=['MATCH_ID'], errors='ignore')
    # concat side-by-side
    combined = pd.concat([df_X, df_Y], axis=1)
    return combined

def load_jpo_XY_list_parallel(directory: str, n_jobs: int = 4) -> List[pd.DataFrame]:
    dir_path = Path(directory)
    files_X = glob.glob(str(dir_path / "jpo_X_*.csv"))
    files_Y = glob.glob(str(dir_path / "jpo_Y_*.csv"))

    # build maps idx -> filepath
    def build_map(files, prefix):
        mp = {}
        pat = re.compile(rf"{prefix}_(\d+)\.csv$")
        for fn in files:
            m = pat.search(Path(fn).name)
            if m:
                mp[int(m.group(1))] = fn
        return mp

    map_X = build_map(files_X, "jpo_X")
    map_Y = build_map(files_Y, "jpo_Y")
    common_idxs = sorted(set(map_X) & set(map_Y))
    if not common_idxs:
        raise FileNotFoundError(f"No matching jpo_X_/jpo_Y_ pairs in {directory!r}")

    # parallel processing
    with ProcessPoolExecutor(max_workers=n_jobs) as exe:
        futures = [
            exe.submit(_load_and_combine, idx, map_X, map_Y, wpe_rewards)
            for idx in common_idxs
        ]
        combined_list = [f.result() for f in futures]

    return combined_list

# ─── Usage ──────────────────────────────────────────────────────────────────────

combined_list = load_jpo_XY_list_parallel("/scratch/network/ly4431/jpo", n_jobs=2)
print(f"Loaded {len(combined_list)} parts in parallel")

# persist if you like
with open("/scratch/network/ly4431/df.pkl", "wb") as f:
    pickle.dump(combined_list, f)
