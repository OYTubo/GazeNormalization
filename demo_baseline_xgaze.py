import os
import cv2
import dlib
from imutils import face_utils
import numpy as np
import torch
from torchvision import transforms
from model import gaze_network
from tqdm import tqdm
import data_loader
import trainer
import argparse
from config import get_config

trans = transforms.Compose([
        transforms.ToPILImage(),
        transforms.ToTensor(),  # this also convert pixel value from [0,255] to [0,1]
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])


if __name__ == '__main__':
    config, unparsed = get_config()
    config.is_train = False
    config.batch_size = 2
    config.use_gpu = True
    config.ckpt_dir = './ckpt'
    config.pre_trained_model_path = './ckpt/epoch_24_ckpt.pth.tar'
    test_data = data_loader.get_test_loader('/home/hgh/hghData/Datasets',batch_size=config.batch_size,num_workers=0)
    xgaze = trainer.Trainer(config, test_data)
    xgaze.test()