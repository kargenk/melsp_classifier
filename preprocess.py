from itertools import product
import pathlib
from pathlib import Path
from typing import Tuple, Dict

from librosa.effects import time_stretch, pitch_shift
import numpy as np
import torch
from tqdm import tqdm

from sound_util import load_wav, show_log_melsp, calc_log_melsp


def add_noise(data: np.ndarray, rate: float = 0.02) -> np.ndarray:
    """
    音源にホワイトノイズを付加する関数.

    Args:
        data (np.ndarray): wavデータ
        rate (float, optional): ホワイトノイズを付与する際の係数. Defaults to 0.02.

    Returns:
        np.ndarray: ホワイトノイズが付与されたデータ
    """
    w_noise = np.random.randn(len(data))
    data_noised = data + rate * w_noise
    return data_noised


def time_domain_stretch(data: np.ndarray, rate: float = 0.5) -> Tuple[np.ndarray]:
    """
    時間軸方向に伸縮する関数.ピッチはそのままで和速のみ変化させる.

    Args:
        data (np.ndarray): wavデータ
        rate (float): どれくらい元の速さから増減させるかの度合い，正の数で指定すること. Defaults to 0.5.

    Returns:
        Tuple[np.ndarray, np.ndarray]: 時間方向に伸縮したデータ
    """
    # positive, if >1 then speed up, if <1 then slow down
    return time_stretch(data, rate)


def freq_domain_shift(data: np.ndarray, sr: int, shift_steps: int = 1) -> Tuple[np.ndarray]:
    """
    基本周波数(f0)をシフトさせる関数.

    Args:
        data (np.ndarray): wavデータ
        sr (int): サンプリング周波数
        shift_step (float): 何度シフトさせるか，正の数で指定すること. Defaults to 1.

    Returns:
        Tuple[np.ndarray, np.ndarray]: ピッチを上下させたデータ
    """
    return pitch_shift(data, sr, n_steps=shift_steps)


def augmentation(data: np.ndarray, sr: int) -> Dict[str, np.ndarray]:
    """
    組み合わせたデータ拡張を行う関数.

    Args:
        data (np.ndarray): wavデータ
        sr (int): サンプリング周波数

    Returns:
        Dict[str, np.ndarray]: データ拡張の組み合わせがキー，各データ拡張を行なったデータがバリューの辞書
    """
    # データ拡張の組み合わせ
    noise_flag = [True, False]        # add or not
    stretch_params = [1.5, 0.5, 1.0]  # speed up, slow down, neutral
    shift_params = [1, -1, 0]         # shift up, shift down, neutral

    # ファイル保存用のキー
    key_noise = ['noise', 'not-noise']
    key_stretch = ['spped-up', 'slow-down', 'not-stretch']
    key_shift = ['shift-up', 'shift-down', 'not-shift']

    # 組み合わせを作成
    augs = list(product(noise_flag, stretch_params, shift_params))
    keys = list(product(key_noise, key_stretch, key_shift))
    augmented_data = {}
    for (is_noise, stretch_rate, shift_step), key in zip(augs, keys):
        if is_noise:
            data = add_noise(data)
        processed = time_domain_stretch(data, rate=stretch_rate)
        processed = freq_domain_shift(processed, sr, shift_steps=shift_step)
        key = '_'.join(key)
        augmented_data[key] = processed

    return augmented_data


def save_log_melsp(root_dir: pathlib.PosixPath, save_dir: pathlib.PosixPath) -> None:
    """
    64フレームずつ移動しながら，128フレームのサイズの対数メルスペクトログラムを保存する関数.

    Args:
        root_dir (pathlib.PosixPath): 音声データディレクトリのパスオブジェクト
        save_dir (pathlib.PosixPath): 保存先ディレクトリのパスオブジェクト
    """
    FRAMES = 128     # フレーム数
    hop_length = 64  # 移動幅

    data_dirs = list(root_dir.iterdir())
    for data_dir in tqdm(data_dirs):
        speaker = data_dir.stem
        # print(speaker)
        wav_paths = list(data_dir.iterdir())

        for wav_index, wav_path in enumerate(wav_paths):
            wav_array, sr = load_wav(wav_path)
            wav_aug_dict = augmentation(wav_array, sr)

            for key, wav_array in wav_aug_dict.items():
                # 対数メルスペクトログラムを計算
                log_melsp = calc_log_melsp(wav_array)

                # 64フレームずつ移動しながら128フレームのサイズに分けて保存
                file_name = f'{speaker}_{wav_index}_' + key
                for start_idx in range(0, log_melsp.shape[1] - FRAMES + 1, hop_length):
                    one_audio_seg = log_melsp[:, start_idx: start_idx + FRAMES]

                    if one_audio_seg.shape[1] == FRAMES:
                        temp_name = f'{file_name}_{start_idx}'
                        file_path = save_dir.joinpath(temp_name)
                        np.save(file_path, one_audio_seg)
                        # print(f'[SAVE]: {file_path}.npy')
            # break


if __name__ == '__main__':
    root_dir = Path.cwd().joinpath('data/audio')
    save_dir = Path.cwd().joinpath('data/log_melsp/aug')
    if not save_dir.exists():
        save_dir.mkdir()

    save_log_melsp(root_dir, save_dir)
