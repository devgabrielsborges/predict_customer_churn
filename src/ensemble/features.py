"""Feature engineering pipeline (7 steps + digits + n-gram composites).

Every function mutates DataFrames *in-place* and returns the list of new
column names it created.  The top-level ``engineer_features`` orchestrator
honours the boolean toggles from :class:`config.CFG`.
"""
from __future__ import annotations

from itertools import combinations
from typing import Sequence

import numpy as np
import pandas as pd

from ensemble.config import CATS, NUMS, SERVICE_COLS, TOP_CATS_FOR_NGRAM, CFG


# ── helpers ──────────────────────────────────────────────────────────────

def _pctrank_against(values: np.ndarray, reference: np.ndarray) -> np.ndarray:
    ref_sorted = np.sort(reference)
    return (np.searchsorted(ref_sorted, values) / len(ref_sorted)).astype("float32")


def _zscore_against(values: np.ndarray, reference: np.ndarray) -> np.ndarray:
    mu, sigma = np.mean(reference), np.std(reference)
    if sigma == 0:
        return np.zeros(len(values), dtype="float32")
    return ((values - mu) / sigma).astype("float32")


# ── Step 1: Frequency encoding ──────────────────────────────────────────

def frequency_encoding(
    train: pd.DataFrame,
    test: pd.DataFrame,
    orig: pd.DataFrame,
) -> list[str]:
    created: list[str] = []
    for col in NUMS:
        freq = pd.concat([train[col], orig[col], test[col]]).value_counts(normalize=True)
        name = f"FREQ_{col}"
        for df in (train, test, orig):
            df[name] = df[col].map(freq).fillna(0).astype("float32")
        created.append(name)
    return created


# ── Step 2: Arithmetic interactions ─────────────────────────────────────

def arithmetic_interactions(
    train: pd.DataFrame,
    test: pd.DataFrame,
    orig: pd.DataFrame,
) -> list[str]:
    for df in (train, test, orig):
        df["charges_deviation"] = (
            df["TotalCharges"] - df["tenure"] * df["MonthlyCharges"]
        ).astype("float32")
        df["monthly_to_total_ratio"] = (
            df["MonthlyCharges"] / (df["TotalCharges"] + 1)
        ).astype("float32")
        df["avg_monthly_charges"] = (
            df["TotalCharges"] / (df["tenure"] + 1)
        ).astype("float32")
    return ["charges_deviation", "monthly_to_total_ratio", "avg_monthly_charges"]


# ── Step 3: Service counts ──────────────────────────────────────────────

def service_counts(
    train: pd.DataFrame,
    test: pd.DataFrame,
    orig: pd.DataFrame,
) -> list[str]:
    for df in (train, test, orig):
        df["service_count"] = (df[SERVICE_COLS] == "Yes").sum(axis=1).astype("float32")
        df["has_internet"] = (df["InternetService"] != "No").astype("float32")
        df["has_phone"] = (df["PhoneService"] == "Yes").astype("float32")
    return ["service_count", "has_internet", "has_phone"]


# ── Step 4: ORIG_proba mapping ──────────────────────────────────────────

def orig_proba_mapping(
    train: pd.DataFrame,
    test: pd.DataFrame,
    orig: pd.DataFrame,
    target: str,
) -> list[str]:
    created: list[str] = []
    for col in CATS + NUMS:
        tmp = orig.groupby(col)[target].mean()
        name = f"ORIG_proba_{col}"
        train[name] = train[col].map(tmp).fillna(0.5).astype("float32")
        test[name] = test[col].map(tmp).fillna(0.5).astype("float32")
        created.append(name)
    return created


# ── Step 5: Distribution features ───────────────────────────────────────

def distribution_features(
    train: pd.DataFrame,
    test: pd.DataFrame,
    orig: pd.DataFrame,
    target: str,
) -> list[str]:
    churner_tc = orig.loc[orig[target] == 1, "TotalCharges"].values
    nonchurner_tc = orig.loc[orig[target] == 0, "TotalCharges"].values
    all_tc = orig["TotalCharges"].values
    is_mc_mean = orig.groupby("InternetService")["MonthlyCharges"].mean()

    for df in (train, test):
        tc = df["TotalCharges"].values
        df["pctrank_nonchurner_TC"] = _pctrank_against(tc, nonchurner_tc)
        df["pctrank_churner_TC"] = _pctrank_against(tc, churner_tc)
        df["pctrank_orig_TC"] = _pctrank_against(tc, all_tc)
        df["zscore_churn_gap_TC"] = (
            np.abs(_zscore_against(tc, churner_tc))
            - np.abs(_zscore_against(tc, nonchurner_tc))
        ).astype("float32")
        df["zscore_nonchurner_TC"] = _zscore_against(tc, nonchurner_tc)
        df["pctrank_churn_gap_TC"] = (
            _pctrank_against(tc, churner_tc)
            - _pctrank_against(tc, nonchurner_tc)
        ).astype("float32")
        df["resid_IS_MC"] = (
            df["MonthlyCharges"]
            - df["InternetService"].map(is_mc_mean).fillna(0)
        ).astype("float32")

        # Conditional percentile ranks
        for group_col, suffix in [("InternetService", "IS"), ("Contract", "C")]:
            vals = np.zeros(len(df), dtype="float32")
            for cat_val in orig[group_col].unique():
                mask = df[group_col] == cat_val
                ref = orig.loc[orig[group_col] == cat_val, "TotalCharges"].values
                if len(ref) > 0 and mask.sum() > 0:
                    vals[mask] = _pctrank_against(
                        df.loc[mask, "TotalCharges"].values, ref
                    )
            df[f"cond_pctrank_{suffix}_TC"] = vals

    return [
        "pctrank_nonchurner_TC", "pctrank_churner_TC", "pctrank_orig_TC",
        "zscore_churn_gap_TC", "zscore_nonchurner_TC", "pctrank_churn_gap_TC",
        "resid_IS_MC", "cond_pctrank_IS_TC", "cond_pctrank_C_TC",
    ]


# ── Step 6: Quantile distance features ──────────────────────────────────

def quantile_distance_features(
    train: pd.DataFrame,
    test: pd.DataFrame,
    orig: pd.DataFrame,
    target: str,
) -> list[str]:
    churner_tc = orig.loc[orig[target] == 1, "TotalCharges"].values
    nonchurner_tc = orig.loc[orig[target] == 0, "TotalCharges"].values

    created: list[str] = []
    for q_label, q_val in [("q25", 0.25), ("q50", 0.50), ("q75", 0.75)]:
        ch_q = np.quantile(churner_tc, q_val)
        nc_q = np.quantile(nonchurner_tc, q_val)
        for df in (train, test):
            df[f"dist_To_ch_{q_label}"] = np.abs(df["TotalCharges"] - ch_q).astype("float32")
            df[f"dist_To_nc_{q_label}"] = np.abs(df["TotalCharges"] - nc_q).astype("float32")
            df[f"qdist_gap_To_{q_label}"] = (
                df[f"dist_To_nc_{q_label}"] - df[f"dist_To_ch_{q_label}"]
            ).astype("float32")
        created += [
            f"dist_To_ch_{q_label}", f"dist_To_nc_{q_label}", f"qdist_gap_To_{q_label}",
        ]
    return created


# ── Step 7: Numericals as category ──────────────────────────────────────

def numerical_as_category(
    train: pd.DataFrame,
    test: pd.DataFrame,
) -> list[str]:
    created: list[str] = []
    for col in NUMS:
        name = f"CAT_{col}"
        for df in (train, test):
            df[name] = df[col].astype(str).astype("category")
        created.append(name)
    return created


# ── Digit features ──────────────────────────────────────────────────────

def digit_features(
    train: pd.DataFrame,
    test: pd.DataFrame,
) -> list[str]:
    created: list[str] = []

    for df in (train, test):
        # Tenure
        t_str = df["tenure"].astype(str)
        df["tenure_first_digit"] = t_str.str[0].astype(int)
        df["tenure_last_digit"] = t_str.str[-1].astype(int)
        df["tenure_second_digit"] = t_str.apply(lambda x: int(x[1]) if len(x) > 1 else 0)
        df["tenure_mod10"] = df["tenure"] % 10
        df["tenure_mod12"] = df["tenure"] % 12
        df["tenure_num_digits"] = t_str.str.len()
        df["tenure_is_multiple_10"] = (df["tenure"] % 10 == 0).astype("float32")
        df["tenure_rounded_10"] = np.round(df["tenure"] / 10) * 10
        df["tenure_dev_from_round10"] = np.abs(df["tenure"] - df["tenure_rounded_10"])

        # MonthlyCharges
        mc_str = df["MonthlyCharges"].astype(str).str.replace(".", "", regex=False)
        df["mc_first_digit"] = mc_str.str[0].astype(int)
        df["mc_last_digit"] = mc_str.str[-1].astype(int)
        df["mc_second_digit"] = mc_str.apply(lambda x: int(x[1]) if len(x) > 1 else 0)
        df["mc_mod10"] = np.floor(df["MonthlyCharges"]) % 10
        df["mc_mod100"] = np.floor(df["MonthlyCharges"]) % 100
        df["mc_num_digits"] = np.floor(df["MonthlyCharges"]).astype(int).astype(str).str.len()
        df["mc_is_multiple_10"] = (np.floor(df["MonthlyCharges"]) % 10 == 0).astype("float32")
        df["mc_is_multiple_50"] = (np.floor(df["MonthlyCharges"]) % 50 == 0).astype("float32")
        df["mc_rounded_10"] = np.round(df["MonthlyCharges"] / 10) * 10
        df["mc_fractional"] = df["MonthlyCharges"] - np.floor(df["MonthlyCharges"])
        df["mc_dev_from_round10"] = np.abs(df["MonthlyCharges"] - df["mc_rounded_10"])

        # TotalCharges
        tc_str = df["TotalCharges"].astype(str).str.replace(".", "", regex=False)
        df["tc_first_digit"] = tc_str.str[0].astype(int)
        df["tc_last_digit"] = tc_str.str[-1].astype(int)
        df["tc_second_digit"] = tc_str.apply(lambda x: int(x[1]) if len(x) > 1 else 0)
        df["tc_mod10"] = np.floor(df["TotalCharges"]) % 10
        df["tc_mod100"] = np.floor(df["TotalCharges"]) % 100
        df["tc_num_digits"] = np.floor(df["TotalCharges"]).astype(int).astype(str).str.len()
        df["tc_is_multiple_10"] = (np.floor(df["TotalCharges"]) % 10 == 0).astype("float32")
        df["tc_is_multiple_100"] = (np.floor(df["TotalCharges"]) % 100 == 0).astype("float32")
        df["tc_rounded_100"] = np.round(df["TotalCharges"] / 100) * 100
        df["tc_fractional"] = df["TotalCharges"] - np.floor(df["TotalCharges"])
        df["tc_dev_from_round100"] = np.abs(df["TotalCharges"] - df["tc_rounded_100"])

        # Derived
        df["tenure_years"] = df["tenure"] // 12
        df["tenure_months_in_year"] = df["tenure"] % 12
        df["mc_per_digit"] = df["MonthlyCharges"] / (df["mc_num_digits"].astype(float) + 0.001)
        df["tc_per_digit"] = df["TotalCharges"] / (df["tc_num_digits"].astype(float) + 0.001)

    created = [
        "tenure_first_digit", "tenure_last_digit", "tenure_second_digit",
        "tenure_mod10", "tenure_mod12", "tenure_num_digits",
        "tenure_is_multiple_10", "tenure_rounded_10", "tenure_dev_from_round10",
        "mc_first_digit", "mc_last_digit", "mc_second_digit",
        "mc_mod10", "mc_mod100", "mc_num_digits",
        "mc_is_multiple_10", "mc_is_multiple_50",
        "mc_rounded_10", "mc_fractional", "mc_dev_from_round10",
        "tc_first_digit", "tc_last_digit", "tc_second_digit",
        "tc_mod10", "tc_mod100", "tc_num_digits",
        "tc_is_multiple_10", "tc_is_multiple_100",
        "tc_rounded_100", "tc_fractional", "tc_dev_from_round100",
        "tenure_years", "tenure_months_in_year",
        "mc_per_digit", "tc_per_digit",
    ]
    return created


# ── N-gram composites ───────────────────────────────────────────────────

def create_ngram_composites(
    train: pd.DataFrame,
    test: pd.DataFrame,
    top_k: int = 6,
) -> tuple[list[str], list[str]]:
    """Create bi-gram and tri-gram composite categorical features.

    Returns (bigram_cols, trigram_cols).
    """
    cats = TOP_CATS_FOR_NGRAM[:top_k]

    bigram_cols: list[str] = []
    for c1, c2 in combinations(cats, 2):
        name = f"BG_{c1}_{c2}"
        for df in (train, test):
            df[name] = (df[c1].astype(str) + "_" + df[c2].astype(str)).astype("category")
        bigram_cols.append(name)

    top4 = cats[:4]
    trigram_cols: list[str] = []
    for c1, c2, c3 in combinations(top4, 3):
        name = f"TG_{c1}_{c2}_{c3}"
        for df in (train, test):
            df[name] = (
                df[c1].astype(str) + "_" + df[c2].astype(str) + "_" + df[c3].astype(str)
            ).astype("category")
        trigram_cols.append(name)

    return bigram_cols, trigram_cols


# ── Orchestrator ────────────────────────────────────────────────────────

class FeatureGroups:
    """Container returned by ``engineer_features`` that tracks column names."""

    def __init__(self):
        self.nums: list[str] = list(NUMS)
        self.cats: list[str] = list(CATS)
        self.new_nums: list[str] = []
        self.num_as_cat: list[str] = []
        self.bigram_cols: list[str] = []
        self.trigram_cols: list[str] = []

    @property
    def ngram_cols(self) -> list[str]:
        return self.bigram_cols + self.trigram_cols

    @property
    def all_features(self) -> list[str]:
        return self.nums + self.cats + self.new_nums + self.num_as_cat + self.ngram_cols

    @property
    def te_columns(self) -> list[str]:
        return self.num_as_cat + self.cats

    @property
    def te_ngram_columns(self) -> list[str]:
        return self.ngram_cols

    @property
    def to_remove(self) -> list[str]:
        return self.num_as_cat + self.cats + self.ngram_cols


def engineer_features(
    train: pd.DataFrame,
    test: pd.DataFrame,
    orig: pd.DataFrame,
    cfg: CFG,
) -> FeatureGroups:
    """Run the full feature engineering pipeline, respecting CFG toggles."""
    fg = FeatureGroups()

    # Always-on steps
    fg.new_nums += frequency_encoding(train, test, orig)
    fg.new_nums += arithmetic_interactions(train, test, orig)
    fg.new_nums += service_counts(train, test, orig)

    if cfg.use_orig_proba:
        fg.new_nums += orig_proba_mapping(train, test, orig, cfg.TARGET)

    if cfg.use_distribution_features:
        fg.new_nums += distribution_features(train, test, orig, cfg.TARGET)

    if cfg.use_quantile_features:
        fg.new_nums += quantile_distance_features(train, test, orig, cfg.TARGET)

    if cfg.use_digit_features:
        fg.new_nums += digit_features(train, test)

    fg.num_as_cat = numerical_as_category(train, test)

    if cfg.use_ngram_features:
        fg.bigram_cols, fg.trigram_cols = create_ngram_composites(
            train, test, top_k=cfg.ngram_top_k,
        )

    return fg
