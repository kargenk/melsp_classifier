from pathlib import Path

import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from dataloader import MelspDataset
from model import MelspClassifier


if __name__ == '__main__':
    device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')

    # 訓練後のモデルの読み込み
    model = MelspClassifier()
    weights = torch.load('models/ep94_val-acc0991.pt')
    model.load_state_dict(weights)
    model.eval()
    model.to(device)

    root = Path.cwd().joinpath('data/log_melsp/_exp')
    dataset = MelspDataset(root)
    melsp, label, speaker = dataset[0]
    # print(melsp.shape, label, speaker)

    dataloader = DataLoader(dataset, batch_size=8, shuffle=True)

    num_epoch = 1
    epoch_corrects = 0.0
    for i in range(1, num_epoch + 1):
        for melsps, labels, speaker in tqdm(dataloader):
            melsps = melsps.to(device)
            labels = labels.to(device)
            with torch.set_grad_enabled(False):
                outputs = model(melsps)
                _, preds = torch.max(outputs, 1)  # 最も値が大きいインデックスが予測したクラス
                epoch_corrects += torch.sum(preds == labels.data)
            print(f'True label: {labels}')
            print(f'speaker: {speaker}')
            print(f'preds     : {preds}')
            print(outputs)
            print(f'model outputs: {F.softmax(outputs, dim=1)}')
            break
        # num_data = len(dataloader.dataset)
        # epoch_acc = epoch_corrects.double() / num_data
        # print(epoch_acc)
