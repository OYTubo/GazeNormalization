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
import math
from collections import deque

cam_drozy = r"D:\DROZY_and_NTHU\GazeNormalization-cpu_1\testpart\DROZY\kinect-intrinsics.yaml"  #drozy的相机参数
fs_drozy = cv2.FileStorage(cam_drozy, cv2.FILE_STORAGE_READ)
camera_matrix_drozy = fs_drozy.getNode('intrinsics').mat()
k, p = fs_drozy.getNode('k').mat(), fs_drozy.getNode('p').mat()
camera_distortion_drozy = np.zeros((5,1))
for i in range(3):
    camera_distortion_drozy[i]=k[i]
for j in range(2):
    camera_distortion_drozy[j+3]=p[j]
w_drozy = 512
h_drozy = 424
fs_drozy.release()


# fileroot = '/home/hgh/hghData/gaze/eve/valpart/val01/step030_video_Wikimedia-Joy-and-Heron-Animated-CGI-Spot-by-Passion-Pictures/webcam_c.h5'
#
# file = h5py.File(fileroot, 'r')
# file.keys()
#
camera_matrix=camera_matrix_drozy
# camera_matrix = file['camera_matrix'][:]
# # print('camera_matrix:')
# # print(camera_matrix)
# camera_transformation = file['camera_transformation'][:]
# # print('camera_transformation:')
# # print(camera_transformation)
# head_rvec = file['head_rvec']['data']
# # print('head_rvec:')
# # print(head_rvec)
# head_tvec = file['head_tvec']['data']
# # print('head_tvec:')
# # print(head_tvec)
# tobii = file['face_PoG_tobii']['data']
# # print('gaze[0]:')
# # print(tobii[0])
# gaze_o = file['face_o']['data']
# # print('gaze_o[0]:')
# # print(gaze_o[0])
# gazen = file['face_g_tobii']['data']
# # print('gazen[0]:')
# # print(gazen[0])
# facial_landmarks = file['facial_landmarks']['data']
# # print('facial_landmarks:')
# # print(facial_landmarks)
# pixel_scale = file['millimeters_per_pixel'][:]
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
pre_trained_model_path = r"D:\DROZY_and_NTHU\GazeNormalization-cpu_1\ckpt\epoch_24_ckpt.pth.tar"
ckpt = torch.load(pre_trained_model_path)
model.load_state_dict(ckpt['model_state'], strict=True)
model.eval()

def popHead(lst):
    if len(lst)>=1800:
        lst.pop(0)
    return lst

def returnAlert(blinkin60s,frameCountReturn,nextBegin):
    alert=0
    if frameCountReturn > 5:
        blinkin60s = popHead(blinkin60s)
        blinkin60s.append(2)
        for j in range(nextBegin - i):
            blinkin60s = popHead(blinkin60s)
            blinkin60s.append(0)
    else:
        blinkin60s = popHead(blinkin60s)
        blinkin60s.append(2)
        for j in range(nextBegin - i):
            blinkin60s = popHead(blinkin60s)
            blinkin60s.append(0)
    if frameCountReturn > 10:
        alert=1
    return alert,blinkin60s

def baselineBn(dataY):
    lenX = len(dataY)
    dataY_b = []
    dataY_b.append(dataY[0])
    Alpha = [0] * lenX
    for i in range(1, lenX):
        Alpha[i] = smoothingFactor(dataY_b, i)
        bI = (1 - Alpha[i]) * dataY_b[i - 1] + Alpha[i] * dataY[i]
        #print(f'bI{bI}')
        dataY_b.append(round(bI,4))
    #print(f'bn:{dataY_b}')
    return dataY_b


def smoothingFactor(dataY_b, n):
    a0 = 0.4
    ad = 15
    aa = 0.5
    ab = 2
    am = 0.7
    exp1 = (-1) * ad * ((dataY[n] - dataY[n - 1]) ** 2)
    exp_1 = math.exp(exp1)
    exp2 = (-1) * aa * (dataY[n] - dataY_b[n - 1]) if dataY[n] - dataY_b[n - 1] > 0 else 0
    exp_2 = math.exp(exp2)
    exp3 = (-1) * ab * (dataY_b[n - 1] - dataY[n]) if dataY_b[n - 1] - dataY[n] > 0 else 0
    exp_3 = math.exp(exp3)
    exp4 = dataY[n] - am * getMedian(dataY, n)
    exp_4 = 1 if exp4 >= 0 else 0
    return a0 * exp_1 * exp_2 * exp_3 * exp_4


def getMedian(dataY, n):
    d = []
    d.extend(dataY[1:n+1])
    d.sort()
    if len(d) % 2 == 0:
        return (d[len(d) // 2 - 1] + d[len(d) // 2]) / 2
    else:
        return d[len(d) // 2]

def blinkCounter(data_normalized):
    i=0
    # flag=0
    blinkin60s=[]
    for k in range(1800):
        blinkin60s.append(0)
    alert=0
    frameCount=0
    blinkCount=0
    blinkLonggest=0
    while i<len(data_normalized):
        if(data_normalized[i]<0.65):
            flag=i
            frameCountReturn,nextBegin=frameBackandForth(data_normalized,flag)
            frameCount+=frameCountReturn
            alert,blinkin60s=returnAlert(blinkin60s,frameCountReturn,nextBegin)
            i=nextBegin
            blinkCount+=1
            if frameCountReturn>blinkLonggest:
                blinkLonggest=frameCountReturn
        i+=1
        blinkin60s.append(0)
    c1=0
    c2=0
    for x in blinkin60s:
        if x==1:
            c1+=1
        elif x==2:
            c2+=1
    if c1>0 and c2>0:
        if c2/(c1+c2)>0.25:
            alert=1
    return frameCount,blinkCount,blinkLonggest,alert

def frameBackandForth(data_normalized,flag):
    frameCount=0
    nextBegin=flag
    for i in range(1,(flag+1 if flag<len(data_normalized)-flag else len(data_normalized)-flag)):
        if(data_normalized[flag-i]<=0.75):
            frameCount+=1
        if(data_normalized[flag+i]<=0.75):
            frameCount+=1
            nextBegin+=1
        if(data_normalized[flag-i]>0.75 and data_normalized[flag+i]>0.75):
            break
    return frameCount,nextBegin

# 数据预处理
# video_path = '/home/hgh/hghData/test20240306.mkv'
video_path="./testpart/3-1.mp4"
cap = cv2.VideoCapture(video_path)
res = []
images = []
idx = 0
ret = True
# camera_distortion = np.array([-0.16321888, 0.66783406, -0.00121854, -0.00303158, -1.02159927])
camera_distortion=camera_distortion_drozy
preds = gaze_normalize.xmodel()
counter=0
dataX=[]
dataY=[]
dataY_b=[]
while counter<300:
    dataX.append(counter)
    counter+=1
    print(f'counter:{counter}')
    ret,image = cap.read()
    # print(f'image:{image}')
    if ret == False:
        break
    gaze_normalize_eve = gaze_normalize.GazeNormalize(image,(0,0), camera_matrix,camera_distortion,preds,is_video=True,image=image) ##True to False
    image_warp, real_eyelip_distance = gaze_normalize_eve.norm()
    dataY.append(real_eyelip_distance)
    dataY_b=baselineBn(dataY)
    data_normalized = []
    for i in range(len(dataY)):
        opening = dataY[i] / dataY_b[i]
        data_normalized.append(round(opening, 4))
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
        frameCount, blinkCount, blinkLonggest,alert=blinkCounter(data_normalized)
        image_draw = gaze_normalize_eve.draw_gaze(frameCount=frameCount, blinkCount=blinkCount, blinkLonggest=blinkLonggest,alert=alert,data_normalized=data_normalized)
        print(f'{data_normalized}')
        # image_draw=gaze_normalize_eve.draw_norm_gaze()
    # res.append(gaze_normalize_eve)
    images.append(image_draw)
    idx += 1
cap.release()

height, width, layers = images[0].shape
# video = cv2.VideoWriter('/home/hgh/hghData/output_3_6.mp4', cv2.VideoWriter_fourcc(*'mp4v'), 25, (width, height))
video = cv2.VideoWriter("./test/output_3-1.mp4", cv2.VideoWriter_fourcc(*'mp4v'), 25, (width, height))


# 将每张图片逐帧写入视频
for image in images:
    video.write(image)
video.release()
# with open('/home/hgh/hghData/eve_cam_c_3_5.pkl', 'wb') as fo:
    # pickle.dump(res,fo)