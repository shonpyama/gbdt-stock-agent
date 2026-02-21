# FMPクオンツパイプライン

Google Colab（GPU）で動作するFMPデータ取得〜特徴量作成〜時系列ウォークフォワード検証の共通基盤。

## 📁 ディレクトリ構成

```
銘柄　colab分析/
├── src/
│   ├── moomoo_costs.py    # moomoo証券コスト計算（注文回数対応）
│   ├── fmp_client.py      # FMP APIクライアント（リトライ・キャッシュ）
│   ├── features.py        # 特徴量エンジン
│   ├── validation.py      # 時系列検証・リーク検知
│   ├── backtest.py        # バックテストロジック
│   ├── metrics.py         # パフォーマンス指標
│   ├── graph_build.py     # [NEW] グラフ構築（GNN用）
│   ├── train_gnn.py       # [NEW] GNN学習（PyTorch Geometric）
│   └── __init__.py        # パッケージ初期化
├── colab/
│   └── fmp_profile_gnn.ipynb  # [NEW] GNNノートブック
├── colab_notebooks/
│   └── 00_setup_and_smoke_test.ipynb
├── outputs/
│   ├── logs/
│   ├── reports/
│   └── csv/
├── generate_fmp_gnn_notebook.py  # [NEW] GNNノートブック生成
└── README.md
```

## 🚀 クイックスタート

### 方法1: ローカルPC実行（推奨・結果が直接保存される）

```bash
# APIキー設定（どちらか）
export FMP_API_KEY="your_api_key"
# または /content/.env_fmp に以下を保存
# FMP_API_KEY=your_api_key
# （生キー1行のみでも可）

# 依存関係インストール + スモーク実行
./setup_local.sh

# GBDT専用導線（GNNを使わない）
./run_gbdt_only.sh AAPL,MSFT,NVDA quick
./run_gbdt_only.sh AAPL,MSFT,NVDA,AMZN,META improve

# 実行（結果は ./outputs/ に保存）
./run_all_local.sh AAPL,MSFT,NVDA       # パイプライン+ランカー+比較
# レビューゲートを失敗時exit=1にする場合
STRICT_REVIEW_GATE=1 ./run_all_local.sh AAPL,MSFT,NVDA
python run_ranker_local.py              # S&P 100（30銘柄）
python run_ranker_local.py AAPL,MSFT    # 銘柄指定

# 結果をターミナルに表示
python display_report.py
# ゲート判定JSONを再生成
python evaluate_review_gate.py
# 最新実行マニフェスト(成果物・ログ・git状態)を確認
cat outputs/reports/latest_run_manifest.json
# 最新KPIを1コマンドで表示
python print_latest_kpi.py
# ゲートFAIL時にパラメータを自動で切り替えて再試行
./auto_improve_loop.sh AAPL,MSFT,NVDA,AMZN,META
# 自動改善の試行上限やゲート閾値を上書き
MAX_ATTEMPTS=4 GATE_MIN_AVG_AUC=0.55 ./auto_improve_loop.sh AAPL,MSFT,NVDA,AMZN,META
# 現在状態(差分・ログ・最新成果物)をDrive上にチェックポイント保存
./save_workspace_checkpoint.sh manual
# dirtyワークツリーでもコミット1件を安全にpush
./push_commit_safely.sh origin main HEAD
# 既にorigin/mainに含まれるコミットを指定した場合は何もせず終了
./push_commit_safely.sh origin main 7469eb7
```

**保存先**: `./outputs/csv/`, `./outputs/reports/`

### 方法2: Google Colab実行

1. `colab_notebooks/01_factor_ml_ranker_moomoo.ipynb` をColabにアップロード
2. GPUランタイムを選択
3. Colab Secretsで `FMP_API_KEY` を設定
4. セルを順次実行
5. 結果をダウンロードまたはGoogle Driveに保存

## 💰 moomoo証券 コスト仕様

### 米国株/ETF 取引手数料

| パラメータ | 税抜 | 税込 | 説明 |
|-----------|------|------|------|
| レート | 0.12% | 0.132% | 約定代金に対する料率 |
| 上限 | $20 | $22 | 1注文あたり上限 |
| 下限 | $0.01 | $0.01 | 最小手数料 |

**重要**: 
- 手数料は「1注文あたり（片道）」
- 分割注文・部分利確は**注文回数分だけ手数料が加算**
- バックテストでは必ず注文回数をカウント

### 使用例

```python
from src.moomoo_costs import calc_moomoo_fee_usd, calc_trade_cost_usd

# 片道手数料（税抜）
fee = calc_moomoo_fee_usd(10000.0, tax_mode="ex")
# → $12.0 (10000 × 0.12%)

# 分割注文の往復コスト
costs = calc_trade_cost_usd(
    entry_notional=100000.0,
    exit_notional=100000.0,
    n_entry_orders=5,  # 5分割エントリー
    n_exit_orders=3,   # 3分割利確
    tax_mode="ex"
)
# → 手数料が注文回数分(8回)加算される
```

## 🔒 セキュリティ方針

- **APIキー**: 環境変数 `FMP_API_KEY` から取得。ログ出力禁止。
- **破壊的コマンド**: 禁止 (`rm -rf`, `sudo` 等)
- **投資推奨**: 禁止。「検証上の候補」「ランキング上位候補」と表現。

## 📊 モジュール一覧

| モジュール | 機能 |
|-----------|------|
| `moomoo_costs.py` | 手数料計算（注文回数対応） |
| `fmp_client.py` | FMP API（リトライ・キャッシュ・レート制御） |
| `features.py` | 特徴量（RSI, MACD, ATR, ADX等） |
| `validation.py` | Purged Walk-Forward、リーク検知 |
| `backtest.py` | バックテスト（注文回数カウント） |
| `metrics.py` | Sharpe, MaxDD, 勝率等 |
| `graph_build.py` | グラフ構築（GNN用） |
| `train_gnn.py` | GCN学習（PyTorch Geometric） |

---

## 🕸️ GNNプロジェクト（FMP Profile GNN）

FMP APIのCompany ProfileとStock Peersを使って銘柄間のpeer関係グラフを構築し、GCNでセクター分類を学習します。

### 実行手順（Google Colab）

1. **ノートブックをアップロード**
   ```
   colab/fmp_profile_gnn.ipynb
   ```

2. **GPUランタイムを選択**（推奨）
   - Runtime → Change runtime type → GPU

3. **セルを順次実行**
   - Cell A: 環境セットアップ（PyTorch Geometric自動インストール）
   - Cell B: FMP APIキー入力
   - Cell C: S&P 500銘柄取得（デフォルト200銘柄）
   - Cell D: Profile/Peers取得（キャッシュ有効）
   - Cell E: グラフ構築
   - Cell F-G: GCN学習
   - Cell H: 評価・可視化
   - Cell I: Artifacts保存

### 出力ファイル

| ファイル | 内容 |
|---------|------|
| `nodes.csv` | ticker, sector, 特徴量 |
| `edges.csv` | src, dst（無向エッジ） |
| `label_map.json` | sector/industryマッピング |
| `training_curve.png` | 学習曲線 |

### ノートブック再生成

```bash
python generate_fmp_gnn_notebook.py
```

---

## 📝 ライセンス

Private - 社内利用限定

---

## 🤖 Claude Code for VS Code 連携

### レポート表示スクリプト

Colabで生成したレポートをターミナルに表示:

```bash
# 最新のレポートを自動検出して表示
python display_report.py

# JSONファイルを指定
python display_report.py outputs/reports/claude_summary_*.json

# Markdownファイルを指定
python display_report.py --md outputs/reports/claude_report_*.md
```

### 出力例

```
======================================================================
📊 FMP Factor ML Ranker バックテストレポート
======================================================================

🔧 設定:
   銘柄数: 20
   税モード: in (0.132%)
   スリッページ: equity=5.0bp, fx=2.0bp

📈 Walk-Forward検証結果:
   Fold   Accuracy   AUC        Test件数
   ---------------------------------------------
   1      0.550      0.520      150

💰 売りルール比較 (Net評価):
   ルール             取引数      注文数      Gross      Net
   -----------------------------------------------------------
   atr_trailing    50       100      +1.50%     +1.20%

   ✅ Net Sharpe最適: atr_trailing

🏆 本日の候補ランキング Top 10:
   ⚠️ これは「検証上の候補」であり、投資推奨ではありません。

   1      NVDA     72.0%    +1.25%
======================================================================
```

### Claude Codeへの依頼例

```
このバックテスト結果（outputs/reports/claude_summary_*.json）を分析して、
1. 最もパフォーマンスが良い売りルールとその理由
2. 特徴量重要度から見える市場の特徴
3. 上位候補銘柄の共通点
を説明してください。
```
