import cv2 as cv
import os
import torch
import numpy as np

from torch.utils.data.dataset import Dataset
from torch.utils.data.dataloader import DataLoader


class TinyImageNet(Dataset):
    def __init__(self, root_dir, filelist) -> None:
        super().__init__()
        self.root = root_dir  # data directory
        self.filelist = self.load_files(filelist)  # read filelist

    def __getitem__(self, index):
        pass
    
    def load_files(self, filelist):
        # load file list
        pass

    def transform(self, data):
        # processing data
        pass

    def __len__(self):
        return len(self.filelist)
