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

if __name__ == '__main__':
    config, unparsed = get_config()
    config.is_train = False
    config.batch_size = 128
    config.use_gpu = True
    config.ckpt_dir = './ckpt'
    config.pre_trained_model_path = './ckpt/epoch_24_ckpt.pth.tar'
    datalist = '/home/hgh/hghData/all_3_5.pkl'
    datapath = '/home/hgh/hghData/pre_3_5'
    test_data = data_loader.get_test_loader(datalist, datapath, batch_size=config.batch_size,num_workers=0)
    xgaze = trainer.Trainer(config, test_data)
    xgaze.test()