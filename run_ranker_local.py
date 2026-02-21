#!/usr/bin/env python3
"""
FMP Factor ML Ranker - ローカル実行版

Colabを使わずにローカルPCで直接実行し、結果を保存します。
保存先: ./outputs/

使用方法:
    python run_ranker_local.py                  # デフォルト銘柄
    python run_ranker_local.py AAPL,MSFT,NVDA   # 銘柄指定
    python run_ranker_local.py sp100            # S&P 100

必要なライブラリ:
    pip install pandas numpy requests xgboost scikit-learn tqdm
"""

import hashlib
import json
import os
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
import requests
from tqdm import tqdm

# ============================================================
# moomooコスト計算
# ============================================================


@dataclass(frozen=True)
class MoomooFeeParams:
    rate_ex_tax: float = 0.0012
    rate_in_tax: float = 0.00132
    cap_ex_tax: float = 20.0
    cap_in_tax: float = 22.0
    min_fee: float = 0.01


def calc_moomoo_fee_usd(
    notional_usd: float, tax_mode: str = "in", params: MoomooFeeParams = None
) -> float:
    """moomoo手数料（1注文あたり・片道）"""
    if params is None:
        params = MoomooFeeParams()
    if notional_usd <= 0:
        return 0.0

    rate = params.rate_in_tax if tax_mode == "in" else params.rate_ex_tax
    cap = params.cap_in_tax if tax_mode == "in" else params.cap_ex_tax

    fee = notional_usd * rate
    fee = min(fee, cap)
    fee = max(fee, params.min_fee)
    return round(fee, 2)


def calc_trade_cost_usd(
    entry_notional: float,
    exit_notional: float,
    n_entry_orders: int = 1,
    n_exit_orders: int = 1,
    tax_mode: str = "in",
    equity_slippage_bps: float = 0.0,
    fx_slippage_bps: float = 0.0,
) -> Dict:
    """取引総コスト"""
    n_entry_orders = max(1, n_entry_orders)
    n_exit_orders = max(1, n_exit_orders)

    entry_fee_per = calc_moomoo_fee_usd(entry_notional / n_entry_orders, tax_mode)
    exit_fee_per = calc_moomoo_fee_usd(exit_notional / n_exit_orders, tax_mode)

    entry_fee = entry_fee_per * n_entry_orders
    exit_fee = exit_fee_per * n_exit_orders
    total_fee = entry_fee + exit_fee

    total_notional = entry_notional + exit_notional
    equity_slippage = total_notional * equity_slippage_bps / 10000
    fx_slippage = total_notional * fx_slippage_bps / 10000
    total_cost = total_fee + equity_slippage + fx_slippage

    return {
        "total_fee_usd": round(total_fee, 2),
        "total_cost_usd": round(total_cost, 2),
        "n_orders": n_entry_orders + n_exit_orders,
    }


# ============================================================
# 設定
# ============================================================


@dataclass
class Config:
    api_key: str = ""
    symbols: List[str] = field(default_factory=list)
    tax_mode: str = "in"
    default_shares: int = 100
    equity_slippage_bps: float = 5.0
    fx_slippage_bps: float = 2.0
    forward_days: int = 20
    n_splits: int = 5
    top_n: int = 20
    # ローカル保存パス
    output_dir: str = "./outputs"
    cache_dir: str = "./cache"


def parse_cli_args(argv: list[str]) -> tuple[str, str]:
    """Return (symbols_input, asof_date_str)."""
    sym_input = "sp100"
    asof_date = os.environ.get("RANKER_ASOF_DATE", "").strip()
    for arg in argv:
        if arg.startswith("--asof-date="):
            asof_date = arg.split("=", 1)[1].strip()
        elif arg.startswith("--"):
            continue
        elif sym_input == "sp100":
            sym_input = arg
    return sym_input, asof_date


def load_api_key() -> str:
    key = os.environ.get("FMP_API_KEY", "").strip()
    if key and "=" not in key:
        return key
    if key.upper().startswith("FMP_API_KEY="):
        parsed = key.split("=", 1)[1].strip().strip("'").strip('"')
        if parsed:
            return parsed

    key_file = Path("/content/.env_fmp")
    if key_file.exists():
        for raw in key_file.read_text(encoding="utf-8").splitlines():
            line = raw.strip()
            if not line or line.startswith("#"):
                continue
            if line.upper().startswith("FMP_API_KEY="):
                value = line.split("=", 1)[1].strip().strip("'").strip('"')
                if value:
                    return value
            elif "=" not in line:
                return line
    return ""


# ============================================================
# FMPクライアント
# ============================================================


class FMPClient:
    BASE_URL = "https://financialmodelingprep.com"

    def __init__(self, api_key: str, cache_dir: str = "./cache"):
        self._api_key = api_key
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.session = requests.Session()
        self._last_request = 0

    def _request(self, endpoint: str, params: dict = None) -> pd.DataFrame:
        params = params.copy() if params else {}

        # キャッシュ
        cache_key = hashlib.md5(
            f"{endpoint}:{json.dumps(params, sort_keys=True)}".encode()
        ).hexdigest()
        cache_path = self.cache_dir / f"{cache_key}.parquet"

        if cache_path.exists():
            mtime = datetime.fromtimestamp(cache_path.stat().st_mtime)
            if datetime.now() - mtime < timedelta(hours=24):
                try:
                    return pd.read_parquet(cache_path)
                except:
                    pass

        # レート制限
        elapsed = time.time() - self._last_request
        if elapsed < 0.2:
            time.sleep(0.2 - elapsed)

        params["apikey"] = self._api_key
        url = f"{self.BASE_URL}{endpoint}"

        for attempt in range(3):
            try:
                self._last_request = time.time()
                r = self.session.get(url, params=params, timeout=30)

                if r.status_code == 429:
                    time.sleep(30)
                    continue
                if r.status_code >= 400:
                    return pd.DataFrame()

                r.raise_for_status()
                data = r.json()

                if isinstance(data, list):
                    df = pd.DataFrame(data)
                elif isinstance(data, dict) and "historical" in data:
                    df = pd.DataFrame(data["historical"])
                else:
                    df = pd.DataFrame()

                if not df.empty:
                    try:
                        df.to_parquet(cache_path)
                    except:
                        pass

                return df
            except:
                if attempt < 2:
                    time.sleep(2)

        return pd.DataFrame()

    def prices_eod(self, symbol: str) -> pd.DataFrame:
        df = self._request("/stable/historical-price-eod/full", {"symbol": symbol})
        if not df.empty and "date" in df.columns:
            df["date"] = pd.to_datetime(df["date"])
            df["symbol"] = symbol
        return df


# ============================================================
# 特徴量生成
# ============================================================


def calc_rsi(close: pd.Series, period: int = 14) -> pd.Series:
    delta = close.diff()
    gain = delta.where(delta > 0, 0).rolling(period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(period).mean()
    return 100 - (100 / (1 + gain / (loss + 1e-10)))


def calc_atr(
    high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14
) -> pd.Series:
    tr = pd.concat(
        [high - low, abs(high - close.shift(1)), abs(low - close.shift(1))], axis=1
    ).max(axis=1)
    return tr.rolling(period).mean()


def generate_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    for p in [5, 20, 60]:
        df[f"return_{p}d"] = df["close"].pct_change(p)

    df["high_52w"] = df["close"].rolling(252).max()
    df["high_52w_dev"] = (df["close"] - df["high_52w"]) / (df["high_52w"] + 1e-10)

    for p in [20, 50, 200]:
        sma = df["close"].rolling(p).mean()
        df[f"ma_{p}_dev"] = (df["close"] - sma) / (sma + 1e-10)

    df["vol_20d"] = df["close"].pct_change().rolling(20).std() * np.sqrt(252)
    df["vol_ratio"] = df["volume"] / (df["volume"].rolling(20).mean() + 1)
    df["rsi_14"] = calc_rsi(df["close"], 14)
    df["atr_14"] = calc_atr(df["high"], df["low"], df["close"], 14)
    df["atr_ratio"] = df["atr_14"] / (df["close"] + 1e-10)

    return df


# ============================================================
# メイン
# ============================================================


def main():
    print("=" * 60)
    print("📊 FMP Factor ML Ranker - ローカル実行版")
    print("=" * 60)

    # APIキー
    api_key = load_api_key()
    if not api_key:
        print("❌ API Key必須: FMP_API_KEY か /content/.env_fmp を設定してください")
        sys.exit(1)
    print("✅ APIキー設定済み")

    sym_input, asof_date_input = parse_cli_args(sys.argv[1:])
    live_mode = bool(asof_date_input)

    if sym_input.lower() == "sp100":
        symbols = [
            "AAPL",
            "ABBV",
            "ABT",
            "ACN",
            "ADBE",
            "AMGN",
            "AMT",
            "AMZN",
            "AVGO",
            "AXP",
            "BA",
            "BAC",
            "BLK",
            "BMY",
            "C",
            "CAT",
            "CL",
            "CMCSA",
            "COP",
            "COST",
            "CRM",
            "CSCO",
            "CVS",
            "CVX",
            "DHR",
            "DIS",
            "DOW",
            "GD",
            "GE",
            "GILD",
            "GM",
            "GOOG",
            "GOOGL",
            "GS",
            "HD",
            "HON",
            "IBM",
            "INTC",
            "JNJ",
            "JPM",
            "KO",
            "LIN",
            "LLY",
            "LMT",
            "LOW",
            "MA",
            "MCD",
            "MDLZ",
            "MDT",
            "MET",
            "META",
            "MMM",
            "MO",
            "MRK",
            "MS",
            "MSFT",
            "NEE",
            "NFLX",
            "NKE",
            "NVDA",
            "ORCL",
            "PEP",
            "PFE",
            "PG",
            "PM",
            "QCOM",
            "RTX",
            "SBUX",
            "SO",
            "T",
            "TGT",
            "TMO",
            "TMUS",
            "TXN",
            "UNH",
            "UNP",
            "UPS",
            "V",
            "VZ",
            "WFC",
            "WMT",
            "XOM",
        ][
            :30
        ]  # 上位30銘柄
    else:
        symbols = [s.strip().upper() for s in sym_input.split(",")]

    print(f"📈 銘柄数: {len(symbols)}")

    config = Config(api_key=api_key, symbols=symbols)

    # 出力ディレクトリ
    output_dir = Path(config.output_dir)
    (output_dir / "csv").mkdir(parents=True, exist_ok=True)
    (output_dir / "reports").mkdir(parents=True, exist_ok=True)

    # データ取得
    print("\n📊 データ取得中...")
    client = FMPClient(config.api_key, config.cache_dir)

    all_data = []
    for sym in tqdm(symbols, desc="Fetching"):
        df = client.prices_eod(sym)
        if df.empty or len(df) < 60:
            continue
        df = df.sort_values("date").reset_index(drop=True)
        df = generate_features(df)
        df["symbol"] = sym

        # Forward Return
        df["fwd_ret_gross"] = np.log(
            df["close"].shift(-config.forward_days) / df["close"]
        )

        # コスト計算
        df["cost_bps"] = df["close"].apply(
            lambda p: (
                (
                    calc_trade_cost_usd(
                        config.default_shares * p,
                        config.default_shares * p,
                        tax_mode=config.tax_mode,
                        equity_slippage_bps=config.equity_slippage_bps,
                        fx_slippage_bps=config.fx_slippage_bps,
                    )["total_cost_usd"]
                    / (config.default_shares * p)
                    * 10000
                )
                if p > 0
                else 0
            )
        )

        df["fwd_ret_net"] = df["fwd_ret_gross"] - df["cost_bps"] / 10000
        if live_mode:
            df = df.dropna(subset=["rsi_14", "return_20d"])
        else:
            df = df.dropna(subset=["fwd_ret_gross", "rsi_14"])

        if len(df) >= 60:
            all_data.append(df)

    if not all_data:
        print("❌ データ取得失敗")
        return

    full_df = pd.concat(all_data, ignore_index=True)
    print(f"✅ データ取得完了: {len(full_df)}行")

    # ランキング日
    latest_date = full_df["date"].max()
    if asof_date_input:
        asof_date = pd.to_datetime(asof_date_input, errors="coerce")
        if pd.notna(asof_date):
            candidate_dates = full_df.loc[full_df["date"] <= asof_date, "date"]
            if not candidate_dates.empty:
                latest_date = candidate_dates.max()
        print(f"📅 ランキング対象日: {latest_date.date()} (requested={asof_date_input})")
    latest = full_df[full_df["date"] == latest_date].copy()

    # EVスコア
    if live_mode:
        latest["ev_score"] = latest["return_20d"] - latest["cost_bps"] / 10000
        latest["fwd_ret_net"] = latest["ev_score"]
    else:
        latest["ev_score"] = latest["fwd_ret_net"]
    latest = latest.sort_values("ev_score", ascending=False).head(config.top_n)

    # ============================================================
    # 保存
    # ============================================================
    date_str = datetime.now().strftime("%Y%m%d_%H%M%S")

    # CSV
    csv_path = output_dir / "csv" / f"candidates_{date_str}.csv"
    latest.to_csv(csv_path, index=False)

    # JSON (Claude Code用)
    summary = {
        "meta": {
            "generated_at": datetime.now().isoformat(),
            "n_symbols": len(symbols),
            "tax_mode": config.tax_mode,
            "fee_rate": "0.132%" if config.tax_mode == "in" else "0.12%",
            "slippage_equity_bps": config.equity_slippage_bps,
            "slippage_fx_bps": config.fx_slippage_bps,
            "ranking_date": latest_date.strftime("%Y-%m-%d"),
            "requested_asof_date": asof_date_input or "",
            "ranking_mode": "live_no_forward" if live_mode else "forward_return",
        },
        "top_candidates": [],
    }

    for i, (_, row) in enumerate(latest.iterrows(), 1):
        summary["top_candidates"].append(
            {
                "rank": i,
                "symbol": row["symbol"],
                "date": row["date"].strftime("%Y-%m-%d"),
                "ev_score": round(float(row.get("ev_score", 0)), 4),
                "fwd_ret_net_pct": round(float(row.get("fwd_ret_net", 0)) * 100, 2),
                "cost_bps": round(float(row.get("cost_bps", 0)), 1),
                "rsi_14": round(float(row.get("rsi_14", 50)), 1),
            }
        )

    json_path = output_dir / "reports" / f"claude_summary_{date_str}.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    # ============================================================
    # ターミナル表示
    # ============================================================
    print("\n" + "=" * 70)
    print("📊 ランキング結果")
    print("=" * 70)

    print(f"\n🔧 設定:")
    print(f"   銘柄数: {len(symbols)}")
    print(
        f"   税モード: {config.tax_mode} ({'0.132%' if config.tax_mode == 'in' else '0.12%'})"
    )

    print(f"\n🏆 候補ランキング Top 10:")
    print(f"   ⚠️ これは「検証上の候補」であり、投資推奨ではありません。")
    print(
        f"\n   {'順位':<6} {'銘柄':<8} {'EVスコア':<12} {'Net Return':<12} {'コスト':<10} {'RSI'}"
    )
    print(f"   {'-' * 60}")

    for i, (_, row) in enumerate(latest.head(10).iterrows(), 1):
        ev = row.get("ev_score", 0) * 100
        net_ret = row.get("fwd_ret_net", 0) * 100
        cost = row.get("cost_bps", 0)
        rsi = row.get("rsi_14", 50)
        print(
            f"   {i:<6} {row['symbol']:<8} {ev:+.2f}%       {net_ret:+.2f}%       {cost:.1f}bp     {rsi:.1f}"
        )

    print("\n" + "=" * 70)
    print("📁 保存されたファイル（ローカルPC）:")
    print(f"   CSV:  {csv_path.absolute()}")
    print(f"   JSON: {json_path.absolute()}")
    print("=" * 70)

    print("\n🤖 Claude Codeで分析:")
    print(f"   python display_report.py {json_path}")
    print("\n✅ 完了!")


if __name__ == "__main__":
    main()
