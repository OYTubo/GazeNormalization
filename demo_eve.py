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
import utils
import pickle


fileroot = '/home/hgh/hghData/gaze/eve/valpart/val01/step030_video_Wikimedia-Joy-and-Heron-Animated-CGI-Spot-by-Passion-Pictures/basler.h5'

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
video_path = fileroot[:-3] + '.mp4'
cap = cv2.VideoCapture(video_path)
pred_norm = []
pred = []
idx = 0
ret = True
while True:
    ret,image = cap.read()
    if ret == False:
        break
    hr, ht = warp_norm.xnorm(image, camera_matrix)
    if(hr.all() == 0 and ht.all() == 0):
        warp_image = np.zeros((224,224,3), dtype=np.byte)
        gcn = np.zeros((3,1))
        R = np.zeros((3,3))
        pred_gaze_np = [0,0]
        pred_norm.append(pred_gaze_np)
        pred.append(np.dot(np.linalg.inv(R),warp_norm.pitchyaw_to_vector(np.array([pred_gaze_np])).T))
        continue
    face_model_load = np.loadtxt('./modules/face_model.txt')  # Generic face model with 3D facial landmarks
    landmark_use = [20, 23, 26, 29, 15, 19]  # we use eye corners and nose conners
    face_model = face_model_load[landmark_use, :]
    warp_image,_,gcn,R = warp_norm.xtrans(image, face_model, hr, ht, camera_matrix, pixel_scale,gc = np.array([100,100,0]))

    # 模型推理
    input_var = warp_image[:, :, [2, 1, 0]]  # from BGR to RGB
    input_var = trans(input_var)
    input_var = torch.autograd.Variable(input_var.float().cuda())
    input_var = input_var.view(1, input_var.size(0), input_var.size(1), input_var.size(2))  # the input must be 4-dimension
    pred_gaze = model(input_var)  # get the output gaze direction, this is 2D output as pitch and raw rotation
    pred_gaze = pred_gaze[0] # here we assume there is only one face inside the image, then the first one is the prediction
    pred_gaze_np = pred_gaze.cpu().data.numpy()  # convert the pytorch tensor to numpy array
    print('Predict normalization gaze vector(pitch yaw):', pred_gaze_np)
    pred_norm.append(pred_gaze_np)
    pred.append(np.dot(np.linalg.inv(R),warp_norm.pitchyaw_to_vector(np.array([pred_gaze_np])).T))
    print('Ground truth gaze vector:', gazen[0])
    idx += 1
cap.release()
pred = np.array(pred)
pred_norm = np.array(pred_norm)
print(pred_norm)
label_norm = gazen
print(label_norm)
tinydict = {'pred_norm': pred_norm, 'pred': pred}
with open('./eve_val01_pred.pkl', 'wb') as fo:
    pickle.dump(tinydict,fo)