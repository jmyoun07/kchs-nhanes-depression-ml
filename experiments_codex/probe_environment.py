from pathlib import Path

import importlib.util

import pandas as pd


DATA_PATH = Path("data/processed/kchs_dev.parquet")
COLS = [
    "dep",
    "age",
    "sex",
    "edu",
    "income_q",
    "partnered",
    "living_alone",
    "smoking",
    "current_drinker",
    "active",
    "sleep_hrs",
    "bmi",
    "htn",
    "dm",
    "srh",
    "stress",
    "W",
    "strata",
    "psu",
    "year",
]


def has_module(name: str) -> bool:
    return importlib.util.find_spec(name) is not None


def main() -> None:
    print("package_availability")
    for name in ["pandas", "pyarrow", "sklearn", "lightgbm", "xgboost", "numpy"]:
        print(f"{name}: {has_module(name)}")

    df = pd.read_parquet(DATA_PATH, columns=COLS)
    print("\ndata")
    print(f"shape: {df.shape}")
    print(f"dep_rate_unweighted: {df['dep'].mean():.6f}")
    print(f"dep_rate_weighted: {(df['dep'] * df['W']).sum() / df['W'].sum():.6f}")
    print(f"weight_min_max: {df['W'].min():.6g}, {df['W'].max():.6g}")
    print("\ndtypes")
    print(df.dtypes.to_string())
    print("\nmissing")
    print(df.isna().sum().to_string())
    print("\nunique_counts")
    for col in COLS:
        if col not in {"W"}:
            print(f"{col}: {df[col].nunique(dropna=False)}")


if __name__ == "__main__":
    main()
