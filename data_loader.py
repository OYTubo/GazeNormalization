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

# from eve dataset
default_camera_matrix = np.array([[1.7806042e+03, 0.0000000e+00, 9.5932886e+02], 
                                  [0.0000000e+00, 1.7798547e+03, 5.7931006e+02], 
                                  [0.0000000e+00, 0.0000000e+00, 1.0000000e+00]])

default_camera_distortion = np.array([-0.16321888, 0.66783406, -0.00121854, -0.00303158, -1.02159927])

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
                           num_workers=4):
    # load dataset
    print('load the test file list from: ', data_dir)
    sub_folder_use = 'test'
    test_set = GazeDataset(data_dir)
    test_loader = DataLoader(test_set, batch_size=batch_size, num_workers=num_workers)

    return test_loader


class GazeDataset(Dataset):
    def __init__(self, path, camera_matrix = default_camera_matrix, camera_distortion = default_camera_distortion, transform = True):
        image_path = os.path.join(path, 'Photo')
        self.images = [os.path.join(image_path, file) for file in os.listdir(image_path)]
        labels = []
        with open(os.path.join(path, 'coordinate_test.txt'), 'r') as f:
            reader = csv.reader(f, delimiter=',')
            for row in reader:
                labels.append(row)
        self.gaze_centers =[[int(i[-2]), int(i[-1])] for i in labels[1:]]
        self.transform = transform
        ##
        #camera_matrix = []
        
        self.camera_matrix = camera_matrix
        self.camera_distortion = camera_distortion
        

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = self.images[idx]
        image = cv2.imread(image)
        gaze_center = np.array(self.gaze_centers[idx])
        if self.transform:
            image,gaze_center = warp_norm.GazeNormalization(image,default_camera_matrix, default_camera_distortion, gaze_center)
            return image, gaze_center


