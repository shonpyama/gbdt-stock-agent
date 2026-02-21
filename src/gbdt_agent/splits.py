from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import date
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd


logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class WalkForwardSplit:
    train_dates: List[date]
    val_dates: List[date]
    test_dates: List[date]

    def windows(self) -> Dict[str, Dict[str, Optional[str]]]:
        def win(ds: List[date]) -> Dict[str, Optional[str]]:
            if not ds:
                return {"start": None, "end": None}
            return {"start": ds[0].isoformat(), "end": ds[-1].isoformat()}

        return {"train": win(self.train_dates), "val": win(self.val_dates), "test": win(self.test_dates)}


def make_walkforward_splits(
    dates: Sequence[date],
    *,
    train: int,
    val: int,
    test: int,
    step: int,
) -> List[WalkForwardSplit]:
    if train <= 0 or val <= 0 or test <= 0 or step <= 0:
        raise ValueError("train/val/test/step must be positive")
    ds = list(dates)
    splits: List[WalkForwardSplit] = []
    start = 0
    n = len(ds)
    while True:
        train_end = start + train
        val_end = train_end + val
        test_end = val_end + test
        if test_end > n:
            break
        splits.append(WalkForwardSplit(
            train_dates=ds[start:train_end],
            val_dates=ds[train_end:val_end],
            test_dates=ds[val_end:test_end],
        ))
        start += step
    return splits


def apply_purge_embargo(
    split: WalkForwardSplit,
    *,
    purge: int,
    embargo: int,
) -> WalkForwardSplit:
    purge = max(0, int(purge))
    embargo = max(0, int(embargo))

    train = split.train_dates[: max(0, len(split.train_dates) - purge)]
    val = split.val_dates[embargo:]
    val = val[: max(0, len(val) - purge)]
    test = split.test_dates[embargo:]
    return WalkForwardSplit(train_dates=train, val_dates=val, test_dates=test)


def split_no_overlap(split: WalkForwardSplit) -> bool:
    if not split.train_dates or not split.val_dates or not split.test_dates:
        return False
    return max(split.train_dates) < min(split.val_dates) and max(split.val_dates) < min(split.test_dates)


def label_audit(labels: pd.DataFrame) -> Tuple[bool, str]:
    if labels.empty:
        return False, "labels_empty"
    bad = labels[labels["label_date"] <= labels["decision_date"]]
    if not bad.empty:
        return False, f"label_date_not_after_decision_date rows={len(bad)}"
    return True, ""


def feature_audit(features: pd.DataFrame) -> Tuple[bool, str]:
    if features.empty:
        return False, "features_empty"
    if "feature_available_date" not in features.columns:
        return False, "missing_feature_available_date"
    bad = features[features["feature_available_date"].isna() | (features["feature_available_date"] >= features["decision_date"])]
    if not bad.empty:
        return False, f"feature_available_date_not_before_decision_date rows={len(bad)}"
    return True, ""


def suspicious_feature_corr(
    merged: pd.DataFrame,
    feature_cols: Sequence[str],
    target_col: str,
    *,
    corr_threshold: float = 0.995,
) -> List[str]:
    suspicious: List[str] = []
    y = pd.to_numeric(merged[target_col], errors="coerce")
    for c in feature_cols:
        s = pd.to_numeric(merged[c], errors="coerce")
        m = s.notna() & y.notna()
        if int(m.sum()) < 50:
            continue
        x = s[m].to_numpy(dtype=float)
        yy = y[m].to_numpy(dtype=float)
        if np.std(x) == 0 or np.std(yy) == 0:
            continue
        corr = float(np.corrcoef(x, yy)[0, 1])
        if np.isnan(corr):
            continue
        if abs(corr) >= corr_threshold:
            suspicious.append(f"{c}:corr={corr:.4f}")
    return suspicious


def run_leakage_gate(
    *,
    features: pd.DataFrame,
    labels: pd.DataFrame,
    split: WalkForwardSplit,
    target_col: str,
    corr_threshold: float,
) -> Dict[str, Any]:
    ok_split = split_no_overlap(split)
    overlap_count = len(set(split.train_dates).intersection(set(split.test_dates)))
    ok_no_overlap_dates = overlap_count == 0
    ok_feat, feat_msg = feature_audit(features)
    ok_lab, lab_msg = label_audit(labels)

    # Correlation check on train set
    feature_cols = [c for c in features.columns if c not in {"decision_date", "symbol", "feature_available_date", "feature_version"}]
    merged = features.merge(labels, on=["decision_date", "symbol"], how="inner")
    train_mask = merged["decision_date"].isin(split.train_dates)
    suspicious = suspicious_feature_corr(
        merged[train_mask], feature_cols, target_col=target_col, corr_threshold=corr_threshold
    )

    passed = bool(ok_split and ok_no_overlap_dates and ok_feat and ok_lab and (len(suspicious) == 0))
    critical = {
        "split_no_overlap": ok_split,
        "train_test_date_overlap_zero": ok_no_overlap_dates,
        "feature_audit": ok_feat,
        "label_audit": ok_lab,
        "corr_threshold_pass": len(suspicious) == 0,
    }
    messages = []
    if not ok_no_overlap_dates:
        messages.append(f"train_test_date_overlap={overlap_count}")
    if feat_msg:
        messages.append(feat_msg)
    if lab_msg:
        messages.append(lab_msg)
    if suspicious:
        messages.append(f"suspicious_feature_corr: {suspicious[:10]}" + (" ..." if len(suspicious) > 10 else ""))

    return {
        "passed": passed,
        "critical": critical,
        "messages": messages,
        "suspicious_feature_corr": suspicious,
        "split_windows": split.windows(),
        "corr_threshold": corr_threshold,
        "target_col": target_col,
    }
