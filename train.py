from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm

from dataloader import MelspDataset
from logger import Logger
from model import MelspClassifier


def train_model(model, dataloaders_dict, criterion, optimizer, logger, num_epochs):
    device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    best_val_acc = 0.0
    for epoch in range(1, num_epochs + 1):
        print(f'Epoch {epoch}/{num_epochs}')
        print('-' * 20)

        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()
            else:
                model.eval()

            epoch_loss = 0.0
            epoch_corrects = 0

            for melsps, labels, speakers in tqdm(dataloaders_dict[phase]):
                melsps = melsps.to(device)
                labels = labels.to(device)
                optimizer.zero_grad()
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(melsps)
                    loss = criterion(outputs, labels)
                    _, preds = torch.max(outputs, 1)  # 最も値が大きいインデックスが予測したクラス

                    # 訓練時のみ逆伝播して更新
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                    epoch_loss += loss.item() * melsps.size(0)
                    epoch_corrects += torch.sum(preds == labels.data)
                    # break

            num_data = len(dataloaders_dict[phase].dataset)
            epoch_loss = epoch_loss / num_data
            epoch_acc = epoch_corrects.double() / num_data
            print(f'{phase} Loss: {epoch_loss:.4f} Accuracy: {epoch_acc:.4f}')

            # TensorBoardのログに表示
            logger.scalar_summary(f'{phase}/loss', epoch_loss, epoch)
            logger.scalar_summary(f'{phase}/accuracy', epoch_acc, epoch)

            # validationデータでのaccuracyが上がればそのモデルを保存
            if phase == 'val' and epoch_acc > best_val_acc:
                best_val_acc = epoch_acc
                _best_val_acc = str(best_val_acc.item())[:5].replace('.', '')
                save_path = f'models/ep{epoch}_val-acc{_best_val_acc}.pt'
                torch.save(model.state_dict(), save_path)
                print(f'[SAVE] epoch {epoch} model to {save_path}')


if __name__ == '__main__':
    torch.manual_seed(42)

    # データセットの作成
    data_dir = Path.cwd().joinpath('data/log_melsp/train')
    dataset = MelspDataset(data_dir)

    # データセットを訓練用とバリデーション用に分割
    n_samples = len(dataset)
    train_size = int(n_samples * 0.8)
    val_size = n_samples - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    # データローダの作成
    train_dataloader = DataLoader(train_dataset, batch_size=128, shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=128, shuffle=True)
    dataloaders_dict = {'train': train_dataloader, 'val': val_dataloader}

    model = MelspClassifier()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters())
    logger = Logger('logs')
    logger.model_summary(model, torch.empty((1, 1, 80, 128)))

    train_model(model, dataloaders_dict, criterion, optimizer, logger, 100)
