# scripts/prepare_data.py
"""
UrbanSound8K を soundata で取得するスクリプト。
保存先は configs/paths.yaml の paths.raw_dir を使います。
既に存在する場合は download() が差分のみ処理します。
"""

import argparse
from pathlib import Path
import yaml
import soundata


def load_raw_dir(paths_yaml: str) -> Path:
    paths = yaml.safe_load(open(paths_yaml, "r", encoding="utf-8"))["paths"]
    raw = Path(paths["raw_dir"]).expanduser().resolve()
    raw.mkdir(parents=True, exist_ok=True)
    return raw


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--paths", default="configs/paths.yaml")
    args = parser.parse_args()

    raw_dir = load_raw_dir(args.paths)
    print(f"[INFO] data_home = {raw_dir}")

    # soundata の data_home を paths.raw_dir に合わせる
    ds = soundata.initialize("urbansound8k", data_home=str(raw_dir))

    # ダウンロード（初回は時間がかかります）
    print("[INFO] downloading UrbanSound8K (if needed)...")
    ds.download()  # 既存ファイルは再DLされません

    # 整合性チェック
    print("[INFO] validating dataset...")
    # validate() は dict を返します（missing / invalid 等の要約）
    report = ds.validate()
    print("[INFO] validation summary:", report["summary"])

    # 期待されるディレクトリ構造の案内
    audio_dir = Path(raw_dir, "audio")
    meta_csv = Path(raw_dir, "metadata", "UrbanSound8K.csv")
    print(f"[INFO] audio dir:   {audio_dir}")
    print(f"[INFO] metadata:    {meta_csv}")
    if not audio_dir.exists() or not meta_csv.exists():
        print("[WARN] 期待するパスが見つかりません。validateの結果を確認してください。")

    print("[DONE] UrbanSound8K is ready.")


if __name__ == "__main__":
    main()
