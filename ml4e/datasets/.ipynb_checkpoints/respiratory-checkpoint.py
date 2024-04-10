import os
import glob
import librosa
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from ml4e.datasets.data_loaders import kaggleDataset

def getKaggleRespiratoryDataset():
    path = kaggleDataset('vbookshelf/respiratory-sound-database')
    return path

class KaggleRespiratoryDataset(Dataset):
    def __init__(self, featurizer = 'stft'):
        self.featurizer = featurizer
        self.data = []

        DATASET_PATH = getKaggleRespiratoryDataset() + '/Respiratory_Sound_Database/Respiratory_Sound_Database/'

        # Get paths of .wav files
        wav_files = glob.glob(DATASET_PATH + 'audio_and_txt_files/*.wav')

        for i, file_path in enumerate(wav_files):
            y, sr = librosa.load(file_path)
            D = librosa.stft(y)

            cycles = pd.read_csv(file_path[:-3] + 'txt', delimiter='\t', header=None).values

            pID = file_path.split('/')[-1][:3]

            diag_file = pd.read_csv(DATASET_PATH + 'patient_diagnosis.csv', header=None)
            diag = diag_file[diag_file[0] == int(pID)].iloc[0, 1]

            self.data.append([int(pID), torch.tensor(np.array(D, dtype=np.float32)), cycles, diag])

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = self.data[idx]
        return sample
