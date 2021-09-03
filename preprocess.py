import pathlib
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

from sound_util import load_wav, show_log_melsp, calc_log_melsp


def save_log_melsp(root_dir: pathlib.PosixPath, save_dir: pathlib.PosixPath) -> None:
    """
    対数メルスペクトログラムを保存する関数.

    Args:
        root_dir (pathlib.PosixPath): 音声データディレクトリのパスオブジェクト
        save_dir (pathlib.PosixPath): 保存先ディレクトリのパスオブジェクト
    """
    FRAMES = 128

    data_dirs = list(root_dir.iterdir())
    for data_dir in tqdm(data_dirs):
        speaker = data_dir.stem
        # print(speaker)
        wav_paths = list(data_dir.iterdir())
        for wav_index, wav_path in enumerate(wav_paths):
            wav_array, sr = load_wav(wav_path)
            log_melsp = calc_log_melsp(wav_array)
            # print(log_melsp.shape)
            # show_log_melsp(log_melsp, sr)
            # break

            # 128フレームずつに分けて保存
            file_name = f'{speaker}_{wav_index}'
            for start_idx in range(0, log_melsp.shape[1] - FRAMES + 1, FRAMES):
                one_audio_seg = log_melsp[:, start_idx: start_idx + FRAMES]

                if one_audio_seg.shape[1] == FRAMES:
                    temp_name = f'{file_name}_{start_idx}'
                    file_path = save_dir.joinpath(temp_name)
                    np.save(file_path, one_audio_seg)
                    # print(f'[SAVE]: {file_path}.npy')
        # break


if __name__ == '__main__':
    root_dir = Path.cwd().joinpath('data/audio')
    save_dir = Path.cwd().joinpath('data/log_melsp')
    if not save_dir.exists():
        save_dir.mkdir()

    save_log_melsp(root_dir, save_dir)
