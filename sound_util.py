from pathlib import Path
from typing import Tuple

import librosa
import librosa.display
from librosa.filters import mel as librosa_mel_fn
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def load_wav(wav_path: str, sr: int = None) -> Tuple[np.ndarray, int]:
    """
    wavファイルを読み込む関数.

    Args:
        wav_path (str): wavファイルへのパス
        sr (int): サンプリング周波数. Defaults to None(original sampling rate)

    Returns:
        Tuple[np.ndarray, int]: numpy配列形式のデータとサンプリング周波数
    """
    # 音声データの読み込み
    data, sr = librosa.load(wav_path, sr=sr)
    data = data.astype(np.float)  # change to ndarray
    # print(f'sampling rate: {sr}')
    # print(f'time: {len(data) // sr} s')

    return data, sr


def show_log_melsp(log_melsp: np.ndarray, sr: int) -> None:
    """
    メルスペクトログラムを描画する関数.

    Args:
        melsp (np.ndarray): 対数メルスペクトログラム
        sr (int, optional): サンプリング周波数.
    """
    librosa.display.specshow(log_melsp.squeeze().numpy(), sr=sr, x_axis='time', y_axis='mel', hop_length=256)
    plt.colorbar(format='%+2.0f dB')
    plt.show()


class Audio2Mel(nn.Module):
    """対数メルスペクトログラムを作成する関数."""
    def __init__(
        self,
        n_fft=1024,
        hop_length=256,
        win_length=1024,
        sampling_rate=24000,
        n_mel_channels=80,
        mel_fmin=0.0,
        mel_fmax=None,
    ):
        super().__init__()
        ##############################################
        # FFT Parameters                             #
        ##############################################
        window = torch.hann_window(win_length).float()
        mel_basis = librosa_mel_fn(
            sampling_rate, n_fft, n_mel_channels, mel_fmin, mel_fmax
        )
        mel_basis = torch.from_numpy(mel_basis).float()
        self.register_buffer("mel_basis", mel_basis)
        self.register_buffer("window", window)
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.win_length = win_length
        self.sampling_rate = sampling_rate
        self.n_mel_channels = n_mel_channels

    def forward(self, audio):
        p = (self.n_fft - self.hop_length) // 2
        audio = F.pad(audio, (p, p), "reflect").squeeze(1)
        fft = torch.stft(
            audio,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            win_length=self.win_length,
            window=self.window,
            center=False,
        )
        real_part, imag_part = fft.unbind(-1)
        magnitude = torch.sqrt(real_part ** 2 + imag_part ** 2)
        mel_output = torch.matmul(self.mel_basis, magnitude)
        log_mel_spec = torch.log10(torch.clamp(mel_output, min=1e-5))  # db単位に変換
        return log_mel_spec


if __name__ == '__main__':
    fft = Audio2Mel()
