from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm

from dataloader import MelspDataset
from model import MelspClassifier


def train_model(model, dataloaders_dict, criterion, optimizer, num_epochs):
    for epoch in range(1, num_epochs + 1):
        print(f'Epoch {epoch + 1}/{num_epochs}')
        print('-' * 20)

        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()
            else:
                model.eval()

            epoch_loss = 0.0
            epoch_corrects = 0

            for melsps, labels, speakers in tqdm(dataloaders_dict[phase]):
                optimizer.zero_grad()
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(melsps)
                    loss = criterion(outputs, labels)
                    _, preds = torch.max(outputs, 1)

                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                    epoch_loss += loss.item() * melsps.size(0)
                    epoch_corrects += torch.sum(preds == labels.data)

            num_data = len(dataloaders_dict[phase].dataset)
            epoch_loss = epoch_loss / num_data
            epoch_acc = epoch_corrects.double() / num_data
            print(f'{phase} Loss: {epoch_loss:.4f} Accuracy: {epoch_acc:.4f}')


if __name__ == '__main__':
    torch.manual_seed(42)

    # データセットの作成
    data_dir = Path.cwd().joinpath('data/log_melsp/aug')
    dataset = MelspDataset(data_dir)

    # データセットを訓練用とバリデーション用に分割
    n_samples = len(dataset)
    train_size = int(n_samples * 0.8)
    val_size = n_samples - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    # データローダの作成
    train_dataloader = DataLoader(dataset, batch_size=8, shuffle=True)
    val_dataloader = DataLoader(dataset, batch_size=8, shuffle=True)
    dataloaders_dict = {'train': train_dataloader, 'val': val_dataloader}

    model = MelspClassifier()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters())

    train_model(model, dataloaders_dict, criterion, optimizer, 1)
