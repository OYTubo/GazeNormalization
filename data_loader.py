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
import pickle

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



def get_test_loader(datalist, dataset, datapath,
                    batch_size,
                    num_workers=0):
    # load dataset
    print('load the test file list from: ', datalist)
    test_set = TestDataset(datalist,dataset,datapath)
    test_loader = DataLoader(test_set, batch_size=batch_size, num_workers=num_workers, shuffle=False)
    return test_loader


class TestDataset(Dataset):
    def __init__(self, datalist, dataset, datapath, transform=None):
        self.data = pd.read_csv(datalist)
        with open(dataset, 'rb') as fo:
            self.data2 = pickle.load(fo, encoding='bytes')
        self.data_path = datapath
        self.transform = trans

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        image_data = self.data.iloc[idx].tolist()
        image_path = image_data[2]
        image_path = os.path.join(self.data_path, image_path)
        label = (int(image_data[3]), int(image_data[4]))
        # 读取图像
        image = cv2.imread(image_path)
        image = image[:, :, [2, 1, 0]]  # from BGR to RGB

        if self.transform:
            image = self.transform(image)

        return image_path, image, label
