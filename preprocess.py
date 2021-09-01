from pathlib import Path

import torch

from sound_util import Audio2Mel, load_wav


if __name__ == '__main__':
    fft = Audio2Mel()

    root_dir = Path.cwd().joinpath('data/audio')
    data_dirs = list(root_dir.iterdir())
    for data_dir in data_dirs:
        speaker = data_dir.stem
        print(speaker)
        wav_paths = list(data_dir.iterdir())
        for i, wav_path in enumerate(wav_paths):
            wav_array, sr = load_wav(wav_path)
            # tensorに変換，nn.Moduleで扱えるようにバッチの次元を追加
            wav_tensor = torch.from_numpy(wav_array).float().unsqueeze(0)
            log_melsp = fft(wav_tensor.unsqueeze(0))  # 内部でpaddingを行うため，2Dの次元を追加
            print(log_melsp.shape)
        break
