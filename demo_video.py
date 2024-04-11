import h5py
import cv2
import warp_norm
import matplotlib
import sys
sys.path.append("./FaceAlignment")
import face_alignment
from skimage import io
import cv2
import numpy as np
import torch
import matplotlib.pyplot as plt
from model import gaze_network
from torchvision import transforms
import pickle
import gaze_normalize
import os

fileroot = '/home/hgh/hghData/gaze/eve/valpart/val01/step030_video_Wikimedia-Joy-and-Heron-Animated-CGI-Spot-by-Passion-Pictures/webcam_c.h5'

file = h5py.File(fileroot, 'r')
file.keys()

camera_matrix = file['camera_matrix'][:]
# print('camera_matrix:')
# print(camera_matrix)
camera_transformation = file['camera_transformation'][:]
# print('camera_transformation:')
# print(camera_transformation)
head_rvec = file['head_rvec']['data']
# print('head_rvec:')
# print(head_rvec)
head_tvec = file['head_tvec']['data']
# print('head_tvec:')
# print(head_tvec)
tobii = file['face_PoG_tobii']['data']
# print('gaze[0]:')
# print(tobii[0])
gaze_o = file['face_o']['data']
# print('gaze_o[0]:')
# print(gaze_o[0])
gazen = file['face_g_tobii']['data']
# print('gazen[0]:')
# print(gazen[0])
facial_landmarks = file['facial_landmarks']['data']
# print('facial_landmarks:')
# print(facial_landmarks)
pixel_scale = file['millimeters_per_pixel'][:]
# print(pixel_scale)
# 模型读取
trans = transforms.Compose([
        transforms.ToPILImage(),
        transforms.ToTensor(),  # this also convert pixel value from [0,255] to [0,1]
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])
model = gaze_network()
model.cuda()
pre_trained_model_path = './ckpt/epoch_24_ckpt.pth.tar'
ckpt = torch.load(pre_trained_model_path)
model.load_state_dict(ckpt['model_state'], strict=True)
model.eval()

# 数据预处理
video_path = '/home/hgh/hghData/test20240306.mkv'
cap = cv2.VideoCapture(video_path)
res = []
images = []
idx = 0
ret = True
camera_distortion = np.array([-0.16321888, 0.66783406, -0.00121854, -0.00303158, -1.02159927])
preds = gaze_normalize.xmodel()
while True:
    ret,image = cap.read()
    if ret == False:
        break
    gaze_normalize_eve = gaze_normalize.GazeNormalize(image,(0,0), camera_matrix,camera_distortion,preds,is_video=True)
    image_warp = gaze_normalize_eve.norm()
    if gaze_normalize_eve.err:
        text = 'EAR: No Detected Face'
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 1
        font_thickness = 2
        font_color = (255, 255, 255)  # 白色
        text_size = cv2.getTextSize(text, font, font_scale, font_thickness)[0]
        text_position = (image.shape[1] - text_size[0] - 10, image.shape[0] - 10)
        cv2.putText(image, text, text_position, font, font_scale, font_color, font_thickness)
        image_draw = image
    else:
        gaze_normalize_eve.pred('./ckpt/epoch_24_ckpt.pth.tar', image_warp)
        image_draw = gaze_normalize_eve.draw_gaze()
    # res.append(gaze_normalize_eve)
    images.append(image_draw)
    idx += 1
cap.release()

height, width, layers = images[0].shape
video = cv2.VideoWriter('/home/hgh/hghData/output_3_6.mp4', cv2.VideoWriter_fourcc(*'mp4v'), 25, (width, height))

# 将每张图片逐帧写入视频
for image in images:
    video.write(image)
video.release()
# with open('/home/hgh/hghData/eve_cam_c_3_5.pkl', 'wb') as fo:
    # pickle.dump(res,fo)