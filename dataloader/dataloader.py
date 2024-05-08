import os
import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
import random
from utils import scan_directory, read_header, load_recording, train_valid_split, normalize_ecg
import pywt
from scipy import signal


def dataloader_train(opt):
    print('Load the dataset...')
    if not os.path.exists(f'./dataset/train_{opt.mode}.csv'):
        print('Create a csv file dataset from the WFDB database.')
        ecg_header, _ = scan_directory(opt.dirs_for_train)
        ecg_header_info = read_header(ecg_header, opt.target_all, opt.unlabels, opt.mapping_info, opt.classes_all)
        ecg_header_info = pd.DataFrame(ecg_header_info)

        train, valid = train_valid_split(opt, ecg_header_info, getattr(opt, f"target_{opt.mode}"), valid_size=opt.valid_ratio,
                                         classes=getattr(opt, f"classes_{opt.mode}"))

        # save the training settings...
        train.to_csv(f'./dataset/train_{opt.mode}.csv')
        valid.to_csv(f'./dataset/valid_{opt.mode}.csv')

        train_header = pd.read_csv(f'./dataset/train_{opt.mode}.csv')
        valid_header = pd.read_csv(f'./dataset/valid_{opt.mode}.csv')
        train = pd.DataFrame(train_header)  # [:500]
        valid = pd.DataFrame(valid_header)  # [:500]
        print(len(train))

    else:
        train_header = pd.read_csv(f'./dataset/train_{opt.mode}.csv')
        valid_header = pd.read_csv(f'./dataset/valid_{opt.mode}.csv')

        train = pd.DataFrame(train_header)  # [:500]
        valid = pd.DataFrame(valid_header)  # [:100]

        print(len(train))

    train_loader = DataLoader(
        dataset=ECGDatasetMultiLabel(opt, train, mode='train'),
        batch_size=opt.batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=True,
        drop_last=True,
        sampler=None
    )

    valid_loader = DataLoader(
        dataset=ECGDatasetMultiLabel(opt, valid, mode='valid'),
        batch_size=opt.batch_size, shuffle=False, num_workers=0
    )

    return train_loader, valid_loader

def dataloader_valid(opt):
    """Create test data loader."""

    print('Load the dataset...')
    if not os.path.exists(f'./dataset/valid_{opt.mode}.csv'):
        exit()  # Exit if test dataset CSV doesn't exist
    else:
        test = pd.read_csv(f'./dataset/valid_{opt.mode}.csv')
        test = pd.DataFrame(test)  # [:500]
        print(len(test))

    # Create test data loader
    test_loader = DataLoader(
        dataset=ECGDatasetMultiLabel(opt, test, mode='test'),
        batch_size=1,
        shuffle=False,
        num_workers=0
    )

    return test_loader

def dataloader_inference(opt):
    """Create test data loader."""

    print('Load the dataset...')
    if not os.path.exists('./dataset/valid.csv'):
        exit()  # Exit if test dataset CSV doesn't exist
    else:
        test = pd.read_csv('./dataset/valid.csv')
        test = pd.DataFrame(test)  # [:500]
        print(len(test))

    # Create test data loader
    test_loader = DataLoader(
        dataset=ECGDatasetMultiLabel(opt, test, mode='test'),
        batch_size=1,
        shuffle=False,
        num_workers=0
    )

    return test_loader


def wavelet_baseline_cancellation_ecg(ecg_data, wavelet='db4', level=9):
    """
    This function is for removing baseline wander

    Parameters:
    - ecg_data: [12, 5000] numpy array
    - wavelet: the name of wavelet ex) 'db4'
    - level

    Returns:
    - baseline_removed_ecg: 12-lead ECG signal with baseline removed
    """
    baseline_removed_ecg = np.zeros_like(ecg_data)
    for i in range(ecg_data.shape[0]):
        coeffs = pywt.wavedec(ecg_data[i], wavelet, level=level)
        coeffs[0] = np.zeros_like(coeffs[0])
        baseline_removed = pywt.waverec(coeffs, wavelet)
        baseline_removed_ecg[i] = baseline_removed[:len(ecg_data[i])]

    return baseline_removed_ecg


class ECGDatasetMultiLabel(Dataset):
    """ECG Dataset class that implements the required methods for PyTorch's Dataset class."""

    def __init__(self, opt, dataset, mode):
        # Initialize variables and load data
        self.fs = opt.fs
        self.samples = opt.samples
        self.data_length = opt.data_length
        self.mode = mode
        self.dataset = dataset

    def __len__(self):
        """Return the number of items in the dataset."""
        return len(self.dataset)

    def __getitem__(self, idx):
        """Return an item from the dataset at the specified index."""
        fs = self.dataset.iloc[idx]['fs']  # Fetch the sampling frequency
        targets = self.dataset.iloc[idx]['target']  # Fetch the targets

        # Load ECG recording
        inputs = load_recording(self.dataset.iloc[idx]['record'])

        # [Step 1] Normalization: Use z-score normalization
        # inputs = normalize_ecg(inputs[:, :self.data_length * self.fs])

        # [Step 2] Adjust input length to a fixed size (self.samples)
        start_index = random.randint(0, inputs.shape[1] - self.samples - 1)
        inputs = inputs[:, start_index:start_index + self.samples]

        # [Step 3] Filtering
        # inputs = wavelet_baseline_cancellation_ecg(inputs)

        inputs = np.nan_to_num(inputs)  # Convert NaN values to zero
        inputs = torch.from_numpy(inputs)  # Convert to PyTorch tensor

        # Process and convert targets to tensor
        targets = list(map(float, targets.replace('[', '').replace(']', '').split()))
        targets = torch.Tensor(targets)

        return inputs, targets
