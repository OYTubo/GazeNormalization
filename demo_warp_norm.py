import os
import cv2
import dlib
from imutils import face_utils
import numpy as np
import torch
from torchvision import transforms
import warp_norm
import time

img_file_name = './test/test00.JPG'
image = cv2.imread(img_file_name)
cam_file_name = './test/cam00.xml'  # this is camera calibration information file obtained with OpenCV
if not os.path.isfile(cam_file_name):
    print('no camera calibration file is found.')
    exit(0)
fs = cv2.FileStorage(cam_file_name, cv2.FILE_STORAGE_READ)
camera_matrix = fs.getNode('Camera_Matrix').mat() # camera calibration information is used for data normalization
camera_distortion = fs.getNode('Distortion_Coefficients').mat()
face_patch_gaze,gzn = warp_norm.GazeNormalization(image, camera_matrix, camera_distortion, gc = np.array([100,100]), method = 'xgaze68')
cv2.imwrite('./test/result00.JPG', face_patch_gaze)