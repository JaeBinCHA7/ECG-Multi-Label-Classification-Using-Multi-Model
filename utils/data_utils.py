import os
import time
import torch
import torch.nn as nn
from sklearn.metrics import f1_score
import numpy as np
from tqdm import tqdm
from skmultilearn.model_selection import iterative_train_test_split
import pandas as pd


######################################################################################################################
#                                                   for dataset                                                      #
######################################################################################################################
def scan_directory(dir_name):
    if os.path.isdir(dir_name) is False:
        print("[Error] There is no directory '%s'." % dir_name)
        exit()

    header_addrs = []
    file_addrs = []

    for dirpath, dirnames, filenames in tqdm(os.walk(dir_name)):
        for filename in filenames:
            if filename.endswith(".hea"):
                filepath = dirpath + '/' + filename
                header_addrs.append(filepath)
            if filename.endswith(".mat"):
                filepath = dirpath + '/' + filename
                file_addrs.append(filepath)

    return header_addrs, file_addrs


def read_header(header_addrs, target_labels, unlabels, mapping_info, classes):
    targets_list = np.load(target_labels, allow_pickle=True)
    unlabels_list = np.load(unlabels, allow_pickle=True)
    mapping_list = np.load(mapping_info, allow_pickle=True)
    target_label_dict = {label: i for i, label in enumerate(targets_list)}

    mapping_label_dict = {}
    for item in mapping_list:
        for label in item[0].split():
            mapping_label_dict[label] = item[1]

    header_info = []
    for h in tqdm(header_addrs):
        hdr = load_header(h)
        dx_all = np.array(get_labels(hdr))
        dx_labels = [dx for dx in get_labels(hdr) if dx not in [str(label) for label in unlabels_list]]

        mapped_dx_labels = [mapping_label_dict.get(dx, dx) for dx in dx_labels]

        target_array = np.zeros(classes)
        for dx in mapped_dx_labels:
            if dx in target_label_dict:
                target_array[target_label_dict[dx]] = 1

        tmp = {
            'header': h,
            'record': h.replace('.hea', '.mat'),
            'nsamp': get_nsamp(hdr) if hdr else 5000,  # 500 Hz X 10 sec
            'leads': get_leads(hdr),
            'age': get_age(hdr),
            'sex': get_sex(hdr),
            'raw_dx': dx_all,
            'dx': mapped_dx_labels,
            'fs': get_frequency(hdr),
            'target': target_array
        }
        header_info.append(tmp)

    return header_info


def load_header(header_file):
    with open(header_file, 'r') as f:
        header = f.read()
    return header

def get_nsamp(header):
    return header.split('\n')[0].split(' ')[3]

# Get labels from header.
def get_labels(header):
    labels = list()
    for l in header.split('\n'):
        if l.startswith('#Dx'):
            try:
                entries = l.split(': ')[1].split(',')
                for entry in entries:
                    labels.append(entry.strip())
            except:
                pass
    return labels

# Get frequency from header.
def get_frequency(header):
    frequency = None
    for i, l in enumerate(header.split('\n')):
        if i == 0:
            try:
                frequency = float(l.split(' ')[2])
            except:
                pass
        else:
            break
    return frequency


# Get leads from header.
def get_leads(header):
    leads = list()
    for i, l in enumerate(header.split('\n')):
        entries = l.split(' ')
        if i == 0:
            num_leads = int(entries[1])
        elif i <= num_leads:
            leads.append(entries[-1])
        else:
            break
    return tuple(leads)


# Get age from header.
def get_age(header):
    age = None
    for l in header.split('\n'):
        if l.startswith('#Age'):
            try:
                age = float(l.split(': ')[1].strip())
            except:
                age = float('nan')
    return age


# Get sex from header.
def get_sex(header):
    sex = None
    for l in header.split('\n'):
        if l.startswith('#Sex'):
            try:
                sex = l.split(': ')[1].strip()
            except:
                pass
    return sex


# Load recording file as an array.
def load_recording(recording_file, header=None, leads=None, key='val'):
    from scipy.io import loadmat
    recording = loadmat(recording_file)[key]
    if header and leads:
        recording = choose_leads(recording, header, leads)
    return recording


# Select specific leads from the recording.
def choose_leads(recording, header, leads):
    num_leads = len(leads)
    num_samples = np.shape(recording)[1]
    chosen_recording = np.zeros((num_leads, num_samples), recording.dtype)
    available_leads = get_leads(header)
    for i, lead in enumerate(leads):
        if lead in available_leads:
            j = available_leads.index(lead)
            chosen_recording[i, :] = recording[j, :]
    return chosen_recording


# Extract lead names from the header.
def get_leads(header):
    leads = list()
    for i, l in enumerate(header.split('\n')):
        entries = l.split(' ')
        if i == 0:
            num_leads = int(entries[1])
        elif i <= num_leads:
            leads.append(entries[-1])
        else:
            break
    return tuple(leads)


# Create a new directory.
def mkdir(path):
    if not os.path.exists(path):
        os.makedirs(path)


# Normalize the ECG data using Z-score normalization.
def normalize_ecg(ecg_data):
    mean = np.mean(ecg_data, axis=1, keepdims=True)
    std = np.std(ecg_data, axis=1, keepdims=True)
    return (ecg_data - mean) / (std + 1e-8)  # Prevent division by zero


def train_valid_split(opt, header_info, target_label, valid_size, classes):
    files = header_info['header'].to_numpy().reshape(-1, 1)
    targets = np.stack(header_info['target'].to_list(), axis=0)
    x_train, y_train, x_valid, y_valid = iterative_train_test_split(files, targets, test_size=valid_size)

    train = read_header(x_train[:, 0].tolist(), target_label, opt.unlabels, opt.mapping_info, classes=classes)
    valid = read_header(x_valid[:, 0].tolist(), target_label, opt.unlabels, opt.mapping_info, classes=classes)

    train = pd.DataFrame(train)
    valid = pd.DataFrame(valid)

    return train, valid


######################################################################################################################
#                                                   for training                                                     #
######################################################################################################################
def optimizer_to(optim, device):
    for param in optim.state.values():
        # Not sure there are any global tensors in the state dict
        if isinstance(param, torch.Tensor):
            param.data = param.data.to(device)
            if param._grad is not None:
                param._grad.data = param._grad.data.to(device)
        elif isinstance(param, dict):
            for subparam in param.values():
                if isinstance(subparam, torch.Tensor):
                    subparam.data = subparam.data.to(device)
                    if subparam._grad is not None:
                        subparam._grad.data = subparam._grad.data.to(device)


# Calculate total number of parameters in a model.
def cal_total_params(our_model):
    total_parameters = 0
    for variable in our_model.parameters():
        shape = variable.size()
        variable_parameters = 1
        for dim in shape:
            variable_parameters *= dim
        total_parameters += variable_parameters

    return total_parameters


# Display a progress bar during training/validation.
class Bar(object):
    def __init__(self, dataloader):
        if not hasattr(dataloader, 'dataset'):
            raise ValueError('Attribute `dataset` not exists in dataloder.')
        if not hasattr(dataloader, 'batch_size'):
            raise ValueError('Attribute `batch_size` not exists in dataloder.')

        self.dataloader = dataloader
        self.iterator = iter(dataloader)
        self.dataset = dataloader.dataset
        self.batch_size = dataloader.batch_size
        self._idx = 0
        self._batch_idx = 0
        self._time = []
        self._DISPLAY_LENGTH = 50

    def __len__(self):
        return len(self.dataloader)

    def __iter__(self):
        return self

    def __next__(self):
        if len(self._time) < 2:
            self._time.append(time.time())

        self._batch_idx += self.batch_size
        if self._batch_idx > len(self.dataset):
            self._batch_idx = len(self.dataset)

        try:
            batch = next(self.iterator)
            self._display()
        except StopIteration:
            raise StopIteration()

        self._idx += 1
        if self._idx >= len(self.dataloader):
            self._reset()

        return batch

    def _display(self):
        if len(self._time) > 1:
            t = (self._time[-1] - self._time[-2])
            eta = t * (len(self.dataloader) - self._idx)
        else:
            eta = 0

        rate = self._idx / len(self.dataloader)
        len_bar = int(rate * self._DISPLAY_LENGTH)
        bar = ('=' * len_bar + '>').ljust(self._DISPLAY_LENGTH, '.')
        idx = str(self._batch_idx).rjust(len(str(len(self.dataset))), ' ')

        tmpl = '\r{}/{}: [{}] - ETA {:.1f}s'.format(
            idx,
            len(self.dataset),
            bar,
            eta
        )
        print(tmpl, end='')
        if self._batch_idx == len(self.dataset):
            print()

    def _reset(self):
        self._idx = 0
        self._batch_idx = 0
        self._time = []
