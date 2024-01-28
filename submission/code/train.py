import os
import random
import pickle

import numpy as np
from PIL import Image

from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset

import torchvision.transforms.functional as TF

import aosnet

class AOSDataset(Dataset):
    def __init__(self, X, y, load_on_demand=True, random_aug=False):
        self.X = X
        self.y = y
        self.load_on_demand = load_on_demand
        self.random_aug = random_aug

    def __len__(self):
        return len(self.y)

    # Load and return data on-demand (channels first)
    def _load_sample(self, path, target=False):
        # NOTE: Might have to be updated according to how the data can be accessed from the drive!
        if path.endswith(".npy"):
            return np.load(path)
        elif path.endswith(".png") or path.endswith(".jpg"):
            if target:
                return np.array(Image.open(path))
            else:
                return np.concatenate([np.expand_dims(np.array(Image.open(path.split(".")[0].split("_")[0] + f"_{i}.png")), axis=0) for i in [10, 40, 150]], axis=0)
        else:
            raise NotImplementedError(f"Please implement the _load_sample function for this unknown datatype {path.split('.')[-1]}")

    def preprocess_channel(self, channel):
        # Convert channel to PIL Image
        channel = Image.fromarray(channel)

        # Histogram Equalization
        channel_equalized = np.array(TF.equalize(channel))

        return channel_equalized

    def __getitem__(self, idx):

        if self.load_on_demand:
            data = self._load_sample(self.X[idx])
            target = self._load_sample(self.y[idx], target=True)
        else:
            data = self.X[idx]
            target = self.y[idx]

        # Apply preprocessing steps to each channel separately
        preprocessed_channels = []

        if self.random_aug:
            channel_ids = list(range(data.shape[0]))
            random.shuffle(channel_ids) # Randomize the order of channel_ids
        else:
            channel_ids = list(range(data.shape[0]))

        for channel_id in channel_ids:
            channel_data = data[channel_id]

            prep_eq = self.preprocess_channel(channel_data)

            preprocessed_channels.extend([channel_data, prep_eq])

        data = np.array(preprocessed_channels)

        del preprocessed_channels

        # Normalize
        data = torch.Tensor(data) / 255.0
        target = torch.Tensor(target).unsqueeze(0) / 255.0 # 1 x ...

        # Clip values to the range [0, 1]
        data = torch.clamp(data, 0, 1)
        target = torch.clamp(target, 0, 1)

        # Apply random augmentation
        if self.random_aug:
            angle = random.choice([0, 90, 180, 270])
            data = TF.rotate(data, angle)
            target = TF.rotate(target, angle)

            if random.random() > 0.5:
                data = TF.hflip(data)
                target = TF.hflip(target)

            if random.random() > 0.5:
                data = TF.vflip(data)
                target = TF.vflip(target)

        return data, target


def divide(data):
    train_ratio = 0.8
    test_ratio = 0.1
    val_ratio = 0.1

    train_split = int(train_ratio * len(data))
    test_split = train_split + int(test_ratio * len(data))

    train_data = data[:train_split]
    test_data = data[train_split:test_split]
    val_data = data[test_split:]

    return train_data, test_data, val_data

def load_data(X, y, batch_size=6):
    # Data can either be a numpy array or a list of filenames
    X_train, X_test, X_val = divide(X)
    y_train, y_test, y_val = divide(y)

    # Create AOSDataset instances
    train_dataset = AOSDataset(X_train, y_train, random_aug=True)
    test_dataset = AOSDataset(X_test, y_test)
    val_dataset = AOSDataset(X_val, y_val)

    # Create DataLoader instances
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)

    return train_loader, test_loader, val_loader


if __name__ == '__main__':
    SAVE_PATH = "C:/tmp/" # Directory to save the model (and the losses as ".pkl")
    X_PATH = "C:/tmp/X"
    Y_PATH = "C:/tmp/y"
    
    X = [os.path.join(X_PATH, file) for file in os.listdir(X_PATH) if file.split(".")[0].endswith("_10")]
    y = [os.path.join(Y_PATH, file) for file in os.listdir(Y_PATH)]

    # Alternatively load X and y from a ".npy" files:
    # X = np.load("X_PATH")
    # y = np.load("Y_PATH")

    train_loader, test_loader, val_loader = load_data(X, y, batch_size=6)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = aosnet.AOSNet(6, 1)
    # model.load_state_dict(torch.load("path_to_a_trained_model")) # Load model checkpoint (i.e., weights)
    model.to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)
    start_epoch = 0

    # Track losses
    train_losses = []
    test_losses = []
    val_losses = []

    # Track PSNRs
    train_psnrs = []
    test_psnrs = []
    val_psnrs = []

    # Training loop
    num_epochs = 50
    for epoch in range(start_epoch, num_epochs):
        train_loss = 0.0
        train_psnr = 0.0
        model.train()
        for batch_idx, (batch_data, batch_targets) in enumerate(
                tqdm(train_loader, desc=f"[TRAIN] Epoch {epoch + 1}/{num_epochs}", unit="Batches")):
            batch_data, batch_targets = batch_data.to(device), batch_targets.to(device)
            print(batch_data.shape)
            print(batch_targets.shape)
            # optimizer.zero_grad()
            outputs = model(batch_data)
            loss = criterion(outputs, batch_targets)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()

            # Calculate PSNR for each batch
            mse = F.mse_loss(outputs, batch_targets)
            psnr = 20 * torch.log10(1.0 / torch.sqrt(mse))
            train_psnr += psnr.item()

            del loss, batch_data, batch_targets, outputs, mse, psnr
            # torch.cuda.empty_cache()

        # Calculate average training loss for the epoch
        avg_train_loss = train_loss / len(train_loader)
        train_losses.append(avg_train_loss)

        # Calculate average training PSNR for the epoch
        avg_train_psnr = train_psnr / len(train_loader)
        train_psnrs.append(avg_train_psnr)

        # Update learning rate
        scheduler.step()

        # Validation
        val_loss = 0.0
        val_psnr = 0.0
        model.eval()
        with torch.no_grad():
            for batch_idx, (batch_data, batch_targets) in enumerate(
                    tqdm(val_loader, desc=f"[VALIDATION] Epoch {epoch + 1}/{num_epochs}", unit="Batches")):
                batch_data, batch_targets = batch_data.to(device), batch_targets.to(device)
                outputs = model(batch_data)
                loss = criterion(outputs, batch_targets)

                val_loss += loss.item()

                # Calculate PSNR for each batch
                mse = F.mse_loss(outputs, batch_targets)
                psnr = 20 * torch.log10(1.0 / torch.sqrt(mse))
                val_psnr += psnr.item()

                del loss, batch_data, batch_targets, outputs, mse, psnr
                # torch.cuda.empty_cache()

        # Calculate average validation loss for the epoch
        avg_val_loss = val_loss / len(val_loader)
        val_losses.append(avg_val_loss)

        # Calculate average validation PSNR for the epoch
        avg_val_psnr = val_psnr / len(val_loader)
        val_psnrs.append(avg_val_psnr)

        loss_dict = {"train_losses": train_losses, "val_losses": val_losses}
        with open(f"{SAVE_PATH}losses_{epoch + 1}.pkl", "wb") as file:
            pickle.dump(loss_dict, file)

        # Save PSNRs
        loss_dict = {"train_psnrs": train_losses, "val_psnrs": val_losses}
        with open(f"{SAVE_PATH}psnrs_{epoch + 1}.pkl", "wb") as file:
            pickle.dump(loss_dict, file)

    model_save_path = f"{SAVE_PATH}final_model.pth"
    torch.save(model.state_dict(), model_save_path)
