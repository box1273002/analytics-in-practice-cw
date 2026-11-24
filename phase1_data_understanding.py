# -*- coding: utf-8 -*-
"""
IB9BW0 Analytics in Practice | Phase 1: Data Understanding and Cleaning

Input:  campaign_data.csv (52,000 x 21), next_campaign.csv (12,000 x 20)
Output: audit printed to stdout only; no files are written; returns two cleaned DataFrames.

Path: set the AIP_DATA_DIR environment variable to the folder holding the two
      CSV files, or replace the placeholder below. Nothing else needs editing.
"""

import os

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

# ----------------------------------------------------------------
# 1. Configuration
# ----------------------------------------------------------------
# Folder that holds campaign_data.csv and next_campaign.csv.
# Override without touching the code:  export AIP_DATA_DIR="/your/folder"
DATA_DIR = os.environ.get("AIP_DATA_DIR", "path/to/dataset")

TRAIN_FILE = "campaign_data.csv"
DEPLOY_FILE = "next_campaign.csv"

COST_PER_EMAIL = 3.50                                   # cost per email sent
SPEND_PER_VISIT = 93.0                                  # average customer spend per visit
MARGIN = 0.30                                           # gross profit margin
VALUE_PER_VISIT = SPEND_PER_VISIT * MARGIN              # value per store visit = 28
BREAKEVEN_PP = COST_PER_EMAIL / VALUE_PER_VISIT * 100   # break-even increment = 12.5pp

NUM_COLS = ["recency", "purchase", "age", "satisfaction_score", "spend"]
BIN_COLS = ["mens", "womens", "new_customer", "dependent", "account",
            "employed", "phone", "payment_card"]
CAT_COLS = ["zip_area", "channel", "delivery", "marriage"]

# purchase_segment bands, used to rebuild the ordinal from the purchase amount
SEG_BINS = [0, 100, 200, 350, 500, 750, 1000, np.inf]


# ----------------------------------------------------------------
# 2. Load and inspect
# ----------------------------------------------------------------
def check_data_dir():
    """Fail early with an actionable message instead of a bare FileNotFoundError."""
    missing = [f for f in (TRAIN_FILE, DEPLOY_FILE)
               if not os.path.isfile(os.path.join(DATA_DIR, f))]
    if missing:
        raise FileNotFoundError(
            f"Could not find {', '.join(missing)} in DATA_DIR = {DATA_DIR!r}.\n"
            f"Set the environment variable AIP_DATA_DIR to the folder that holds "
            f"campaign_data.csv and next_campaign.csv, e.g.\n"
            f"    export AIP_DATA_DIR=\"path/to/dataset\""
        )


def load(name):
    """Read a CSV with an explicit definition of missing.

    keep_default_na=False stops pandas from turning 'None'/'NA' into NaN;
    na_values=[""] declares that only blank cells count as missing.
    The dataset contains no pseudo-missing codes, but this makes the check
    explicit rather than assumed.
    """
    return pd.read_csv(os.path.join(DATA_DIR, name),
                       keep_default_na=False, na_values=[""], dtype=str)


def overview(df, name):
    """Raw-text overview: shape, sample rows, per-column cardinality and missingness.

    Built on the text-level frame, because df.info() would report all 21
    columns as str and say nothing about how values are actually encoded.
    """
    print(f"\n=== {name} ===")
    print(f"shape: {df.shape[0]:,} rows x {df.shape[1]} columns")
    print(df.head(3).to_string())

    rows = []
    for c in df.columns:
        s = df[c]
        m = s.isna()
        top = s[~m].value_counts().head(3)
        rows.append({
            "column": c,
            "dtype": str(s.dtype),
            "n_unique": int(s.nunique(dropna=True)),   # 1 = zero variance, n = identifier
            "n_missing": int(m.sum()),
            "pct_missing": round(m.mean() * 100, 1),
            "top_3_values": " | ".join(f"{k}({v:,})" for k, v in top.items()),
        })
    info = pd.DataFrame(rows).set_index("column")
    print("\n--- Column overview ---")
    print(info.to_string(max_colwidth=40))
    return info


def numeric_summary(df, cols):
    """Numeric summary with p01/p99, 3*IQR fences and skew.

    Quantiles rather than min/max characterise the tails; skew decides whether
    a log transform or a tree-based model is needed later.
    """
    num = df[cols].apply(pd.to_numeric, errors="coerce")
    q1, q3 = num.quantile(0.25), num.quantile(0.75)
    iqr = q3 - q1
    out = pd.DataFrame({
        "n": num.notna().sum(),
        "mean": num.mean().round(2),
        "sd": num.std().round(2),
        "min": num.min(),
        "p01": num.quantile(0.01).round(2),
        "median": num.median(),
        "p99": num.quantile(0.99).round(2),
        "max": num.max(),
        "skew": num.skew().round(2),
        "n_far_3iqr": ((num < q1 - 3 * iqr) | (num > q3 + 3 * iqr)).sum(),
    })
    print("\n--- Numeric summary ---")
    print(out.to_string())
    return out


# ----------------------------------------------------------------
# 3. Quality diagnostics
# ----------------------------------------------------------------
def parse_segment(df):
    """purchase_segment -> ordinal 1..7; falls back to the band implied by the amount.

    Labels in the file look like '3) 200 - 350', so the leading index is parsed
    rather than matching the range text ('7) 1,000 +' also carries a thousands
    separator and a space).
    """
    ord_seg = df["purchase_segment"].str.extract(r"^\s*(\d)\)", expand=False).astype("float")
    rebuilt = pd.cut(
        pd.to_numeric(df["purchase"], errors="coerce"),
        bins=SEG_BINS, labels=False, right=False
    ) + 1
    return ord_seg.fillna(rebuilt).astype("Int64")


def segment_consistency(df):
    """Cross-check the segment label against the band implied by the amount."""
    lbl = df["purchase_segment"].str.extract(r"^\s*(\d)\)", expand=False).astype("float")
    reb = pd.cut(
        pd.to_numeric(df["purchase"], errors="coerce"),
        bins=SEG_BINS, labels=False, right=False
    ) + 1
    both = lbl.notna() & reb.notna()
    return {
        "checked": int(both.sum()),
        "mismatch": int((lbl[both] != reb[both]).sum()),
        "label_missing": int(lbl.isna().sum()),
    }


def smd(x, flag):
    """Standardised mean difference: (mean | missing - mean | observed) / overall SD.

    |SMD| < 0.1 is conventionally read as missing-at-random; above 0.1 the
    missingness is related to customer attributes rather than to chance.
    """
    x = pd.to_numeric(x, errors="coerce")
    sd = x.std()
    if not sd or np.isnan(sd):
        return np.nan
    return (x[flag].mean() - x[~flag].mean()) / sd


def missing_profile(df):
    """Missingness profile: the evidence behind impute / keep-NaN decisions.

    Each row carries its own observed-group reference, so the table is
    self-contained and no reference value is hard-coded.
    Must run before clean(), which drops spend.
    """
    y = df["visit"].astype(int)
    seg = parse_segment(df).astype(float)
    purchase = pd.to_numeric(df["purchase"], errors="coerce")
    spend = pd.to_numeric(df["spend"], errors="coerce")
    new = df["new_customer"].astype(float)
    rows = []
    for c in ["satisfaction_score", "recency", "age", "zip_area", "channel"]:
        m = df[c].isna()
        rows.append({
            "column": c,
            "n_missing": int(m.sum()),
            "pct": round(m.mean() * 100, 1),
            "visit_missing": round(y[m].mean() * 100, 1),
            "visit_observed": round(y[~m].mean() * 100, 1),
            "gap_pp": round((y[m].mean() - y[~m].mean()) * 100, 1),
            "flag_auc": round(roc_auc_score(y, m.astype(int)), 3),
            "new_share": round(new[m].mean() * 100, 1),
            "new_share_obs": round(new[~m].mean() * 100, 1),
            "high_band": round(seg[m].ge(5).mean() * 100, 1),
            "high_band_obs": round(seg[~m].ge(5).mean() * 100, 1),
            "smd_purchase": round(smd(purchase, m), 2),
            "spend_if_visit": round(spend[(y == 1) & m].mean(), 1),
            "spend_if_visit_obs": round(spend[(y == 1) & ~m].mean(), 1),
        })
    prof = pd.DataFrame(rows).set_index("column")
    print("\n--- Missingness profile (missing vs observed) ---")
    print(prof.to_string())
    return prof


# ----------------------------------------------------------------
# 4. Cleaning
# ----------------------------------------------------------------
def clean(df, is_training):
    """Apply the cleaning decisions.

    No value is invented for a missing cell, except an explicit 'Unknown'
    level for the two nominal variables where the blank is a real signal.
    """
    log = []
    df = df.copy()

    # 1. Identifier: not a quantity, moved out of the feature set (re-attach on submission)
    ids = df.pop("Customer_ID")

    # 2. Zero variance: account is 1 for every customer
    df["account"] = df["account"].astype("Int64")
    if df["account"].nunique(dropna=False) == 1:
        df = df.drop(columns=["account"])
        log.append(("delete", "account", "zero variance: every customer has an account"))

    # 3. Target leakage: spend is recorded after the campaign; spend>0 implies visit
    if "spend" in df.columns:
        df = df.drop(columns=["spend"])
        log.append(("delete", "spend", "post-outcome variable: spend>0 implies visit"))

    # 4. Redundancy: purchase_segment is a coarse copy of purchase, and its label is
    #    missing for some rows while the amount is always present.
    #    Keep the continuous purchase; rebuild an ordinal for reporting only
    df["purchase"] = pd.to_numeric(df["purchase"], errors="coerce")
    df["purchase_segment_ord"] = parse_segment(df)
    df = df.drop(columns=["purchase_segment"])
    log.append(("delete", "purchase_segment",
                "coarse copy of purchase; ordinal rebuilt for reporting only"))

    # 5. recency: missing group is 25.7% high band (observed 11.7%), visit 30.6% vs 15.3%
    #    Zero-filling would read as 'bought this month' (the opposite direction);
    #    median-filling would read as an average customer. NaN kept.
    df["recency"] = pd.to_numeric(df["recency"], errors="coerce")
    df["recency_missing"] = df["recency"].isna().astype(int)
    log.append(("keep-nan", "recency", "missing concentrated in high-value customers; imputation forbidden"))

    # 6. age: 86.1% of missing rows are new customers (overall 48.8%) - incomplete profiles
    df["age"] = pd.to_numeric(df["age"], errors="coerce")
    df["age_missing"] = df["age"].isna().astype(int)
    log.append(("keep-nan", "age", "structural missingness: new customers with incomplete profiles"))

    # 7. satisfaction_score: 59.8% missing = survey non-response; imputation invents an attitude
    #    No -1 sentinel (it would break the 2..10 ordinal scale). NaN kept.
    df["satisfaction_score"] = pd.to_numeric(df["satisfaction_score"], errors="coerce")
    df["satisfaction_missing"] = df["satisfaction_score"].isna().astype(int)
    log.append(("keep-nan", "satisfaction_score", "survey non-response; imputation invents an attitude"))

    # 8. zip_area / channel: 1.5% missing, but the missing group visits 26% vs 16%
    #    An explicit unknown level is kept instead of the mode
    for c in ["zip_area", "channel"]:
        df[c] = df[c].fillna("Unknown")
        log.append(("unknown-level", c, "missing group visits 26% vs 16%; blank is a valid level"))

    # 9. delivery / marriage: numeric codes that are categories; averaging 1/2/3 is meaningless
    for c in ["delivery", "marriage"]:
        s = pd.to_numeric(df[c], errors="coerce").astype("Int64").astype(str)
        df[c] = s.replace({"<NA>": "Unknown", "nan": "Unknown"}).astype("category")
        log.append(("measurement-level", c, "numeric codes are categories; averaging 1/2/3 is meaningless"))

    # 10. Binary flags to Int64 (nullable integer, avoids float casting)
    for c in BIN_COLS:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce").astype("Int64")

    # 11. Nominal categories declared explicitly
    for c in CAT_COLS + ["email_segment"]:
        df[c] = df[c].astype("category")

    # 12. The label exists in the training file only
    if is_training:
        df["visit"] = pd.to_numeric(df["visit"]).astype(int)

    return df, ids, log


# ----------------------------------------------------------------
# 5. Pipeline
# ----------------------------------------------------------------
def main():
    check_data_dir()
    raw_train = load(TRAIN_FILE)
    raw_deploy = load(DEPLOY_FILE)

    # Text-level first impression: must run before the numeric cast,
    # otherwise '3) 200 - 350' and '1,000 +' are converted away
    overview(raw_train, "campaign_data")
    overview(raw_deploy, "next_campaign")

    for df in (raw_train, raw_deploy):
        for c in NUM_COLS:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    raw_train["visit"] = raw_train["visit"].astype(int)

    numeric_summary(raw_train, NUM_COLS + ["visit"])

    print(f"\ntrain {raw_train.shape}  deploy {raw_deploy.shape}")
    print("columns present in train only:", sorted(set(raw_train) - set(raw_deploy)))

    print("\n--- purchase_segment consistency ---")
    for nm, d in (("campaign_data", raw_train), ("next_campaign", raw_deploy)):
        cons = segment_consistency(d)
        print(f"  {nm:14s} checked {cons['checked']:,} | mismatch {cons['mismatch']} "
              f"| label missing {cons['label_missing']}")

    print("\n--- Category levels ---")
    print(raw_train[["zip_area", "channel", "email_segment", "delivery", "marriage"]]
          .apply(lambda s: sorted(s.dropna().unique())).to_string())

    print("\n--- Duplicates and ID integrity ---")
    print("duplicate Customer_ID:", int(raw_train["Customer_ID"].duplicated().sum()))
    print("ID overlap between files:", len(set(raw_train["Customer_ID"]) & set(raw_deploy["Customer_ID"])))

    missing_profile(raw_train)

    print("\n--- Treatment arms and class imbalance ---")
    print(raw_train["email_segment"].value_counts().to_string())
    print(f"positive rate {raw_train['visit'].mean() * 100:.2f}%  "
          f"imbalance 1:{1 / raw_train['visit'].mean() - 1:.1f}  "
          f"break-even lift {BREAKEVEN_PP:.1f}pp")

    # Missing rates should be equal across arms; inequality means missingness
    # is related to the email received (confounding between arm and data quality)
    print("\nMissing rate by arm (should be equal):")
    for c in ["zip_area", "channel", "recency", "age", "satisfaction_score"]:
        rates = [round(float(raw_train.loc[raw_train["email_segment"] == a, c].isna().mean() * 100), 2)
                 for a in ["No E-Mail", "Mens E-Mail", "Womens E-Mail"]]
        print(f"  {c:20} {rates}")

    train_clean, _, log1 = clean(raw_train, True)
    deploy_clean, _, log2 = clean(raw_deploy, False)

    print("\n--- Cleaning log ---")
    for step, col, why in log1:
        print(f"  {step:18} {col:20} {why}")

    print(f"\ncleaned: {train_clean.shape} and {deploy_clean.shape}")
    print("missing kept by design: recency", int(train_clean["recency"].isna().sum()),
          "| age", int(train_clean["age"].isna().sum()),
          "| satisfaction", int(train_clean["satisfaction_score"].isna().sum()))

    # Feature set for modelling: no ID, no purchase_segment_ord (redundant),
    # no email_segment (that is the treatment, not a covariate)
    FEATURES = ["recency", "recency_missing", "purchase", "mens", "womens", "zip_area",
                "new_customer", "channel", "age", "age_missing", "dependent", "employed",
                "phone", "delivery", "marriage", "payment_card",
                "satisfaction_score", "satisfaction_missing"]
    print(f"\nmodel features ({len(FEATURES)}):", FEATURES)
    print("held out: Customer_ID (submission), purchase_segment_ord (report), email_segment (treatment)")

    return train_clean, deploy_clean


if __name__ == "__main__":
    train_clean, deploy_clean = main()
