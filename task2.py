# eda.py
"""
Simple EDA script for the F1 DNF data.
Usage:
    python eda.py
Put f1_dnf.csv in the same folder as this script before running.
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime

# ------- Config -------
INPUT_CSV = "f1_dnf.csv"
CLEANED_CSV = "f1_dnf_cleaned.csv"
PLOTS_DIR = "plots"
README_OUT = "README.md"
# ----------------------

def ensure_dirs():
    os.makedirs(PLOTS_DIR, exist_ok=True)

def load_data(path):
    print(f"Loading {path} ...")
    df = pd.read_csv(path)
    print(f"Loaded: {df.shape[0]} rows, {df.shape[1]} columns")
    return df

def basic_info(df):
    print("\n--- DataFrame info() ---")
    df.info()
    print("\n--- head() ---")
    print(df.head(5))
    print("\n--- describe() (numeric) ---")
    print(df.describe().T)

def clean_basic(df):
    # strip column names & strings
    df.columns = [c.strip() for c in df.columns]
    for c in df.select_dtypes(include=['object']).columns:
        df[c] = df[c].astype(str).str.strip().replace({'nan': np.nan})
    # try to coerce numeric-like object columns to numeric (safe)
    for c in df.columns:
        if df[c].dtype == object:
            # if >50% of values look numeric after coercion, convert
            coerced = pd.to_numeric(df[c].str.replace(',', '').str.extract(r'([0-9\.\-]+)')[0], errors='coerce')
            if coerced.notna().sum() / len(df) > 0.5:
                df[c] = coerced
    # attempt to parse date-like columns (column name contains 'date' or 'dob')
    for c in df.columns:
        if 'date' in c.lower() or 'dob' in c.lower():
            parsed = pd.to_datetime(df[c], errors='coerce', dayfirst=True)
            if parsed.notna().sum() / len(df) > 0.2:
                df[c] = parsed
    return df

def missing_report(df):
    miss = df.isnull().sum().sort_values(ascending=False)
    miss_pct = (df.isnull().mean()*100).sort_values(ascending=False)
    miss_df = pd.concat([miss, miss_pct], axis=1)
    miss_df.columns = ['missing_count','missing_pct']
    print("\n--- Missing values (top 20) ---")
    print(miss_df.head(20))
    return miss_df

def plot_missing(miss_df):
    top = miss_df[miss_df['missing_pct']>0].head(20)
    plt.figure(figsize=(10,5))
    if top.empty:
        plt.text(0.5,0.5,"No missing values", ha='center', va='center', fontsize=14)
        plt.axis('off')
    else:
        top['missing_pct'].plot.bar()
        plt.title("Top columns by % missing")
        plt.ylabel("Percent missing")
        plt.tight_layout()
    p = os.path.join(PLOTS_DIR, "plot_missing.png")
    plt.savefig(p)
    plt.close()
    print(f"Saved {p}")

def plot_numeric_distribution(df, max_cols=3):
    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    for col in num_cols[:max_cols]:
        plt.figure(figsize=(8,4))
        # histogram
        sns.histplot(df[col].dropna(), kde=False)
        plt.title(f"Distribution of numeric column: {col}")
        plt.xlabel(col)
        plt.ylabel("count")
        plt.tight_layout()
        p = os.path.join(PLOTS_DIR, f"hist_{col}.png")
        plt.savefig(p)
        plt.close()
        print(f"Saved {p}")

def plot_by_year(df):
    # prefer 'year' column or extract from date-like columns
    if 'year' in df.columns:
        series = df['year']
    else:
        # try to find a date column
        date_cols = [c for c in df.columns if pd.api.types.is_datetime64_any_dtype(df[c])]
        series = None
        if date_cols:
            series = df[date_cols[0]].dt.year
    if series is not None:
        counts = series.value_counts().sort_index()
        plt.figure(figsize=(10,4))
        counts.plot(kind='bar')
        plt.title("Records by Year")
        plt.xlabel("Year")
        plt.ylabel("count")
        plt.tight_layout()
        p = os.path.join(PLOTS_DIR, "by_year.png")
        plt.savefig(p)
        plt.close()
        print(f"Saved {p}")
    else:
        print("No 'year' or date-like column found; skipping year plot.")

def plot_top_counts(df, column_candidates=['driver','team','constructor','name'], topn=15):
    for cand in column_candidates:
        if cand in df.columns:
            top = df[cand].fillna('Unknown').value_counts().head(topn)
            plt.figure(figsize=(10,5))
            top.plot(kind='bar')
            plt.title(f"Top {cand} (top {topn})")
            plt.ylabel("count")
            plt.tight_layout()
            p = os.path.join(PLOTS_DIR, f"top_{cand}.png")
            plt.savefig(p)
            plt.close()
            print(f"Saved {p}")

def plot_reasons(df, reason_candidates=['status','result','reason','retirement_reason']):
    for cand in reason_candidates:
        if cand in df.columns:
            top = df[cand].fillna('Unknown').value_counts().head(20)
            plt.figure(figsize=(10,6))
            top.plot(kind='bar')
            plt.title(f"Top categories in {cand}")
            plt.ylabel("count")
            plt.tight_layout()
            p = os.path.join(PLOTS_DIR, f"reasons_{cand}.png")
            plt.savefig(p)
            plt.close()
            print(f"Saved {p}")
            break

def save_readme(df, miss_df):
    n_rows, n_cols = df.shape
    top_missing = miss_df.head(10).to_string()
    content = f"""# EDA — F1 DNF (auto-generated)

Rows: {n_rows}
Columns: {n_cols}

Top missing columns:
{top_missing}

Files generated:
- Cleaned CSV: {CLEANED_CSV}
- Plots folder: {PLOTS_DIR}/

Notes:
- Inspect plots to decide imputation & encoding strategies.
- Consider encoding 'driver' and 'team' if present.
"""
    with open(README_OUT, "w", encoding="utf-8") as f:
        f.write(content)
    print(f"Saved {README_OUT}")

def main():
    ensure_dirs()
    if not os.path.exists(INPUT_CSV):
        print(f"ERROR: {INPUT_CSV} not found in folder. Put your CSV in this folder and re-run.")
        return
    df = load_data(INPUT_CSV)
    basic_info(df)
    df = clean_basic(df)
    # Save cleaned
    df.to_csv(CLEANED_CSV, index=False)
    print(f"Saved cleaned CSV -> {CLEANED_CSV}")
    # Missing report
    miss_df = missing_report(df)
    miss_df.to_csv(os.path.join(PLOTS_DIR, "missing_report.csv"))
    # Plots
    plot_missing(miss_df)
    plot_numeric_distribution(df, max_cols=3)
    plot_by_year(df)
    plot_top_counts(df)
    plot_reasons(df)
    # README
    save_readme(df, miss_df)
    print("EDA complete. Check the plots/ folder and README.md")

if __name__ == "__main__":
    main()