# UrbanSound8K-分類
2026年ディープラーニング前期課題用のリポジトリです。UrbanSound8kデータセットの分類モデルを構築・評価します。

## 環境構築
```bash
python -m venv .venv
source .venv/bin/activate   # Windows: .\.venv\Scripts\activate
pip install --upgrade pip
pip install -r requirements.txt

# UrbanSound8K を自動ダウンロード（保存先は configs/paths.yaml の paths.raw_dir）
python scripts/prepare_data.py --paths configs/paths.yaml
````

## 前処理（特徴量生成）

```bash
python src/featurize.py --config configs/base.yaml --paths configs/paths.yaml
```

## 学習

```bash
python src/train.py --config configs/train_small.yaml --fold 1
```

## 評価

```bash
python src/evaluate.py --fold 1
```

## データ

UrbanSound8K データセット（配布元に従って取得してください）。
`data/raw/UrbanSound8K/` 以下に配置してください。

