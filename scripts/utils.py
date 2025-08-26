"""
このファイルには、データセット準備やモデル訓練・検証のためのユーティリティ関数とクラスが含まれています。
"""


import librosa
import numpy as np
import os
import torch
from torch.utils.data import Dataset
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from sklearn.preprocessing import LabelEncoder
from scipy.ndimage import zoom
from tqdm import tqdm
import matplotlib.pyplot as plt
import japanize_matplotlib


def extract_spectrogram(file_path, target_shape=(128, 128), duration=4.0):
    """
    librosaを用いて音声ファイルからスペクトログラムのnumpy配列を抽出する関数
    
    Args:
        file_path: 音声ファイルのパス
        target_shape: 出力するスペクトログラムの形状(height, width)
        duration: 音声の長さ(秒)

    Returns:
        スペクトログラム(dB変換済み)のnumpy配列
    """
    # 音声ファイル読込
    y, sr = librosa.load(file_path, duration=duration)

    # パディング(指定秒数に正規化)
    target_length = int(sr * duration)
    if len(y) < target_length:
        y = y[:target_length]
    else:
        y = np.pad(y, (0, target_length - len(y)), "constant")

    # STFT計算
    stft = librosa.stft(y, n_fft=1024, hop_length=512)
    magnitude = np.abs(stft)

    # dB変換
    db_spec = librosa.amplitude_to_db(magnitude, ref=np.max)

    # リサイズ
    if db_spec.shape != target_shape:
        zoom_factors = (target_shape[0] / db_spec.shape[0], target_shape[1] / db_spec.shape[1])
        db_spec = zoom(db_spec, zoom_factors)

    # 正規化 (-1 to 1)
    db_spec = (db_spec - db_spec.min()) / (db_spec.max() - db_spec.min()) * 2 - 1

    return db_spec.astype(np.float32)





class UrbanSound8KDataset(Dataset):
    """
    UrbanSound8K用のPyTorchデータセットクラス
    事前にpandasでCSVを読み込み、DataFrameを渡すことを想定しています。
    
    """
    def __init__(self, df, data_path, target_shape=(128, 128), duration=4.0):
        self.df = df.reset_index(drop=True)
        self.data_path = data_path
        self.target_shape = target_shape
        self.duration = duration
        
        # ラベルエンコーダー
        self.label_encoder = LabelEncoder()
        self.labels = self.label_encoder.fit_transform(df['class'])
        self.num_classes = len(self.label_encoder.classes_)
        
        print(f"クラス数: {self.num_classes}")
        print(f"クラス: {self.label_encoder.classes_}")
    
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        file_path = os.path.join(self.data_path, f"fold{row['fold']}", row['slice_file_name'])
        
        # スペクトログラム抽出
        spectrogram = extract_spectrogram(file_path, self.target_shape, self.duration)
        
        # チャンネル次元を追加 (1, H, W)
        spectrogram = np.expand_dims(spectrogram, axis=0)
        
        # PyTorchテンソルに変換
        spectrogram = torch.tensor(spectrogram, dtype=torch.float32)
        label = torch.tensor(self.labels[idx], dtype=torch.long)
        
        return spectrogram, label
    


def train_model(model, device, train_loader, val_loader, epochs, learning_rate, model_name):
    """
    モデルを訓練する関数

    Args:
        model: 訓練するモデル
        device: 使用するデバイス (cpu or cuda)
        train_loader: 訓練データのDataLoader
        val_loader: 検証データのDataLoader
        epochs: エポック数
        learning_rate: 学習率
        model_name: モデル名

    Returns:
        model: 訓練後のモデル
        train_losses: 訓練データの損失履歴
        val_losses: 検証データの損失履歴
        train_accuracies: 訓練データの精度履歴
        val_accuracies: 検証データの精度履歴
    """
    model = model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    train_losses = []
    val_losses = []
    train_accuracies = []
    val_accuracies = []
    
    for epoch in range(epochs):
        # 訓練フェーズ
        model.train()
        train_loss = 0
        train_correct = 0
        train_total = 0
        
        for batch_idx, (data, target) in enumerate(tqdm(train_loader, desc=f'Epoch {epoch+1}/{epochs} [Train]')):
            data, target = data.to(device), target.to(device)
            
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            _, predicted = output.max(1)
            train_total += target.size(0)
            train_correct += predicted.eq(target).sum().item()
        
        # 検証フェーズ
        model.eval()
        val_loss = 0
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            for data, target in val_loader:
                data, target = data.to(device), target.to(device)
                output = model(data)
                val_loss += criterion(output, target).item()
                
                _, predicted = output.max(1)
                val_total += target.size(0)
                val_correct += predicted.eq(target).sum().item()
        
        # 統計計算
        train_loss /= len(train_loader)
        val_loss /= len(val_loader)
        train_acc = 100. * train_correct / train_total
        val_acc = 100. * val_correct / val_total
        
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        train_accuracies.append(train_acc)
        val_accuracies.append(val_acc)
        
        print(f'Epoch {epoch+1}/{epochs}:')
        print(f'  Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%')
        print(f'  Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%')
        print('-' * 50)
    
    # 結果の可視化
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    
    # 損失の推移
    ax1.plot(range(1, epochs+1), train_losses, label='Train Loss')
    ax1.plot(range(1, epochs+1), val_losses, label='Val Loss')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.set_title(f'{model_name} - Loss推移')
    ax1.legend()
    ax1.grid(True)
    
    # 精度の推移
    ax2.plot(range(1, epochs+1), train_accuracies, label='Train Accuracy')
    ax2.plot(range(1, epochs+1), val_accuracies, label='Val Accuracy')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy (%)')
    ax2.set_title(f'{model_name} - Accuracy推移')
    ax2.legend()
    ax2.grid(True)
    
    plt.tight_layout()
    plt.show()
    
    return model, train_losses, val_losses, train_accuracies, val_accuracies