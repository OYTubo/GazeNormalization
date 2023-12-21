import numpy as np
import h5py
import torch
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
import os
import json
import random
from typing import List
import csv
import warp_norm
import cv2
import pandas as pd


trans_train = transforms.Compose([
        transforms.ToPILImage(),
        transforms.ToTensor(),  # this also convert pixel value from [0,255] to [0,1]
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])

trans = transforms.Compose([
        transforms.ToPILImage(),
        transforms.ToTensor(),  # this also convert pixel value from [0,255] to [0,1]
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])



def get_test_loader(data_dir,
                           batch_size,
                           num_workers=0):
    # load dataset
    print('load the test file list from: ', data_dir)
    test_set = TestDataset(data_dir)
    test_loader = DataLoader(test_set, batch_size=batch_size, num_workers=num_workers, shuffle=False)
    return test_loader


class TestDataset(Dataset):
    def __init__(self, csv_file_path, transform=None):
        self.data = pd.read_csv(csv_file_path)
        self.transform = trans

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        image_path = self.data.loc[idx, 'image_path']
        label = self.data.loc[idx, 'label']

        # 读取图像
        image = cv2.imread(image_path)
        image = image[:, :, [2, 1, 0]]  # from BGR to RGB

        if self.transform:
            image = self.transform(image)

        return image_path, image, label
