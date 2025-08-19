import librosa, numpy as np
from pathlib import Path
import yaml, argparse

def extract_logmel(wav_path, cfg):
    y, sr = librosa.load(wav_path, sr=cfg["sr"], mono=True)
    mel = librosa.feature.melspectrogram(
        y=y, sr=sr,
        n_fft=cfg["n_fft"],
        hop_length=cfg["hop_length"],
        n_mels=cfg["n_mels"],
        fmin=cfg["fmin"], fmax=cfg["fmax"]
    )
    return librosa.power_to_db(mel, ref=np.max).astype(np.float32)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/base.yaml")
    parser.add_argument("--paths", default="configs/paths.yaml")
    args = parser.parse_args()

    cfg = yaml.safe_load(open(args.config))["audio"]
    paths = yaml.safe_load(open(args.paths))["paths"]

    raw_dir = Path(paths["raw_dir"])
    feat_dir = Path(paths["feat_dir"]); feat_dir.mkdir(parents=True, exist_ok=True)

    wavs = list(raw_dir.rglob("*.wav"))
    for wav in wavs:
        out = feat_dir / (wav.stem + ".npy")
        if out.exists(): continue
        logmel = extract_logmel(wav, cfg)
        np.save(out, logmel)
