"""
Split dataset for PMF: benchmarks of new models (after cutoff_date) are partially held out for testing.
"""
import pandas as pd
import numpy as np
import argparse
from pathlib import Path

DATA_DIR = Path(__file__).parent / "opencompass_cache"


def parse_time(t):
    if pd.isna(t):
        return None
    try:
        parts = str(t).split('/')
        return (int(parts[0]), int(parts[1])) if len(parts) >= 2 else None
    except Exception:
        return None


def split_data(data_path, cutoff_date="2025/01/01", test_ratio=0.4, random_seed=42):
    np.random.seed(random_seed)
    df = pd.read_csv(data_path)

    cutoff_year, cutoff_month = (int(x) for x in cutoff_date.split('/')[:2])

    old_idx, new_idx = [], []
    for idx, row in df.iterrows():
        t = parse_time(row['Time'])
        if t and (t[0] > cutoff_year or (t[0] == cutoff_year and t[1] >= cutoff_month)):
            new_idx.append(idx)
        else:
            old_idx.append(idx)

    meta_cols = {'Model', 'Parameters', 'Organization', 'OpenSource', 'Time'}
    bench_cols = [c for c in df.columns if c not in meta_cols]

    train_rows, test_rows = [], []

    # Old models go entirely into training set
    for i in old_idx:
        train_rows.append(df.iloc[i].copy())

    # New models: split benchmarks by ratio
    for i in new_idx:
        row = df.iloc[i]
        valid = [c for c in bench_cols if pd.notna(row[c])]
        if not valid:
            train_rows.append(row.copy())
            continue

        np.random.shuffle(valid)
        n_test = max(1, int(len(valid) * test_ratio))
        test_set, train_set = set(valid[:n_test]), set(valid[n_test:])

        row_train = row.copy()
        row_test = row.copy()
        for c in bench_cols:
            if c not in train_set:
                row_train[c] = np.nan
            if c not in test_set:
                row_test[c] = np.nan

        train_rows.append(row_train)
        test_rows.append(row_test)

    train_df = pd.DataFrame(train_rows)
    test_df = pd.DataFrame(test_rows)

    print(f"Old models: {len(old_idx)}, New models: {len(new_idx)}")
    print(f"Training set: {len(train_df)} models, {train_df[bench_cols].notna().sum().sum()} samples")
    print(f"Test set: {len(test_df)} models, {test_df[bench_cols].notna().sum().sum()} samples")
    return train_df, test_df


def main():
    parser = argparse.ArgumentParser(description="Split train/test sets")
    parser.add_argument("--data_path", type=str,
                        default=str(DATA_DIR / "opencompass_vlm_full.csv"))
    parser.add_argument("--cutoff_date", type=str, default="2025/01/01")
    parser.add_argument("--test_ratio", type=float, default=0.4)
    parser.add_argument("--random_seed", type=int, default=42)
    args = parser.parse_args()

    r = args.test_ratio
    train_path = DATA_DIR / f"train_data_wide_{r}.csv"
    test_path = DATA_DIR / f"test_data_wide_{r}.csv"

    train_df, test_df = split_data(
        args.data_path, args.cutoff_date, args.test_ratio, args.random_seed
    )

    train_df.to_csv(train_path, index=False)
    test_df.to_csv(test_path, index=False)
    print(f"Saved: {train_path}\nSaved: {test_path}")


if __name__ == "__main__":
    main()
