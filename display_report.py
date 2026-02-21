#!/usr/bin/env python3
"""
FMP Factor ML Ranker レポート表示スクリプト

Colabで生成したJSON/Markdownレポートをターミナルに表示します。
Claude Code for VS Codeで使用することを想定しています。

使用方法:
    python display_report.py                    # 最新のレポートを表示
    python display_report.py path/to/report.json  # 指定したJSONを表示
    python display_report.py --md path/to/report.md  # Markdownを表示
"""

import json
import os
import sys
from datetime import datetime
from pathlib import Path


def find_latest_report(base_dir: str = ".") -> tuple:
    """最新のレポートファイルを検索"""
    base = Path(base_dir)

    # JSONを探す
    json_files = list(base.glob("**/claude_summary_*.json"))
    if not json_files:
        json_files = list(base.glob("**/outputs/reports/*.json"))

    # MDを探す
    md_files = list(base.glob("**/claude_report_*.md"))
    if not md_files:
        md_files = list(base.glob("**/outputs/reports/*.md"))

    latest_json = max(json_files, key=lambda x: x.stat().st_mtime, default=None)
    latest_md = max(md_files, key=lambda x: x.stat().st_mtime, default=None)

    return latest_json, latest_md


def display_json_report(json_path: str):
    """JSONレポートをターミナルに表示"""
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    print("=" * 70)
    print("📊 FMP Factor ML Ranker バックテストレポート")
    print("=" * 70)

    # === メタ情報 ===
    meta = data.get("meta", {})
    print(f"\n🔧 設定:")
    print(f"   生成日時: {meta.get('generated_at', 'N/A')}")
    print(f"   銘柄数: {meta.get('n_symbols', 'N/A')}")
    print(f"   税モード: {meta.get('tax_mode', 'N/A')} ({meta.get('fee_rate', 'N/A')})")
    print(f"   上限: {meta.get('fee_cap', 'N/A')}")
    print(
        f"   スリッページ: equity={meta.get('slippage_equity_bps', 0)}bp, fx={meta.get('slippage_fx_bps', 0)}bp"
    )
    print(f"   デバッグモード: {meta.get('debug_mode', False)}")

    # === Walk-Forward結果 ===
    wf = data.get("walk_forward", {})
    folds = wf.get("folds", [])

    print(f"\n📈 Walk-Forward検証結果:")
    print(f"   {'Fold':<6} {'Accuracy':<10} {'AUC':<10} {'Test件数'}")
    print(f"   {'-' * 45}")

    for m in folds:
        print(
            f"   {m.get('fold', 0)+1:<6} {m.get('accuracy', 0):.3f}      {m.get('auc', 0):.3f}      {m.get('n_test', 0)}"
        )

    print(f"   {'-' * 45}")
    print(
        f"   {'平均':<6} {wf.get('avg_accuracy', 0):.3f}      {wf.get('avg_auc', 0):.3f}"
    )

    # === 売りルール比較 ===
    exit_rules = data.get("exit_rules_comparison", {})

    print(f"\n💰 売りルール比較 (Net評価):")
    print(
        f"   {'ルール':<15} {'取引数':<8} {'注文数':<8} {'Gross':<10} {'Net':<10} {'Sharpe':<8} {'勝率'}"
    )
    print(f"   {'-' * 75}")

    best_sharpe = -float("inf")
    best_rule = None

    for rule, r in exit_rules.items():
        gross = r.get("mean_gross_pct", 0)
        net = r.get("mean_net_pct", 0)
        sharpe = r.get("sharpe_net", 0)
        win = r.get("win_rate_pct", 0)
        n_trades = r.get("n_trades", 0)
        n_orders = r.get("total_orders", 0)
        eligible = bool(r.get("eligible", True))
        tag = "" if eligible else " (ineligible)"

        print(
            f"   {rule:<15} {n_trades:<8} {n_orders:<8} {gross:+.2f}%     {net:+.2f}%     {sharpe:.2f}     {win:.1f}%{tag}"
        )

        if eligible and sharpe > best_sharpe:
            best_sharpe = sharpe
            best_rule = rule

    if best_rule:
        print(f"\n   ✅ Net Sharpe最適: {best_rule} (Sharpe={best_sharpe:.2f})")
    elif exit_rules:
        print("\n   ⚠️ 有効ルールなし（最低取引数条件を満たさず）")

    # === 特徴量重要度 ===
    importance = data.get("feature_importance", [])

    print(f"\n🔍 特徴量重要度 Top 10:")
    for item in importance[:10]:
        bar = "█" * int(item.get("importance", 0) * 30)
        print(
            f"   {item.get('rank', 0):2d}. {item.get('feature', ''):<20} {bar} {item.get('importance', 0):.4f}"
        )

    # === 候補銘柄 ===
    candidates = data.get("top_candidates", [])

    print(f"\n🏆 本日の候補ランキング Top 10:")
    print(f"   ⚠️ これは「検証上の候補」であり、投資推奨ではありません。")
    score_def = meta.get("score_definition", {})
    basis = candidates[0].get("ev_score_basis", "") if candidates else ""
    third_col_name = "Net Return"
    if basis == "pred_proba_minus_0_5":
        third_col_name = "Action"
        sel_rule = score_def.get("selected_rule")
        sel_thr = score_def.get("decision_threshold")
        if sel_rule is not None and sel_thr is not None:
            print(f"   ルール: {sel_rule} (threshold={float(sel_thr):.2f})")
    print(
        f"\n   {'順位':<6} {'銘柄':<8} {'確率':<8} {'EVスコア':<12} {third_col_name:<12} {'コスト'}"
    )
    print(f"   {'-' * 65}")

    for c in candidates[:10]:
        rank = c.get("rank", 0)
        symbol = c.get("symbol", "")
        prob = c.get("pred_proba", 0) * 100
        ev = c.get("ev_score", 0) * 100
        if basis == "pred_proba_minus_0_5":
            third_val = c.get("action", "LONG" if prob >= 50.0 else "FLAT")
        else:
            third_val = f"{c.get('fwd_ret_20d_net_pct', 0):+.2f}%"
        cost = c.get("cost_bps", 0)

        if basis == "pred_proba_minus_0_5":
            print(
                f"   {rank:<6} {symbol:<8} {prob:.1f}%    {ev:+.2f}%       {third_val:<12} {cost:.1f}bp"
            )
        else:
            print(
                f"   {rank:<6} {symbol:<8} {prob:.1f}%    {ev:+.2f}%       {third_val:<12} {cost:.1f}bp"
            )

    print("\n" + "=" * 70)
    print("📁 ソースファイル:", json_path)
    print("=" * 70)


def display_markdown_report(md_path: str):
    """Markdownレポートをターミナルに表示"""
    with open(md_path, "r", encoding="utf-8") as f:
        content = f.read()

    print("=" * 70)
    print("📝 Markdownレポート表示")
    print("=" * 70)
    print()
    print(content)
    print()
    print("=" * 70)
    print("📁 ソースファイル:", md_path)
    print("=" * 70)


def main():
    args = sys.argv[1:]

    # 引数なし: 最新のレポートを探す
    if not args:
        json_path, md_path = find_latest_report()

        if json_path:
            print(f"📂 最新のJSONレポートを検出: {json_path}")
            display_json_report(str(json_path))
        elif md_path:
            print(f"📂 最新のMarkdownレポートを検出: {md_path}")
            display_markdown_report(str(md_path))
        else:
            print("❌ レポートファイルが見つかりません。")
            print("   Colabで生成したレポートをこのディレクトリにコピーしてください。")
            print("   または引数でファイルパスを指定してください:")
            print("     python display_report.py path/to/claude_summary_*.json")
        return

    # --md オプション: Markdownを表示
    if args[0] == "--md":
        if len(args) < 2:
            print("❌ Markdownファイルのパスを指定してください")
            return
        display_markdown_report(args[1])
        return

    # ファイルパス指定
    file_path = args[0]

    if not os.path.exists(file_path):
        print(f"❌ ファイルが見つかりません: {file_path}")
        return

    if file_path.endswith(".json"):
        display_json_report(file_path)
    elif file_path.endswith(".md"):
        display_markdown_report(file_path)
    else:
        print(f"❌ サポートされていないファイル形式: {file_path}")
        print("   .json または .md ファイルを指定してください")


if __name__ == "__main__":
    main()
