import os
import cv2
import csv
import numpy as np
import pandas as pd
from PIL import Image
import warp_norm
import pickle
import gaze_normalize
from ipdb import set_trace as st


cam_chen = '/home/hgh/hghData/Datasets/camChen.xml'  # this is camera calibration information file obtained with OpenCV
fs_chen = cv2.FileStorage(cam_chen, cv2.FILE_STORAGE_READ)
camera_matrix_chen = fs_chen.getNode('Camera_Matrix').mat() # camera calibration information is used for data normalization
camera_distortion_chen = fs_chen.getNode('Distortion_Coefficients').mat()

cam_tan = '/home/hgh/hghData/Datasets/camTan.xml'  # this is camera calibration information file obtained with OpenCV
fs_tan = cv2.FileStorage(cam_tan, cv2.FILE_STORAGE_READ)
camera_matrix_tan = fs_tan.getNode('Camera_Matrix').mat() # camera calibration information is used for data normalization
camera_distortion_tan = fs_tan.getNode('Distortion_Coefficients').mat()


def get_condition_number(file_dict):
    if (1 <= file_dict <= 100):
        return 0
    if (101 <= file_dict <= 200):
        return 1
    if (201 <= file_dict <= 300):
        return 2
    if (301 <= file_dict <= 400):
        return 3
    if (401 <= file_dict <= 500):
        return 4
    if (501 <= file_dict <= 600):
        return 5
    if (601 <= file_dict <= 700):
        return 6
    if (701 <= file_dict <= 800):
        return 7
    if (801 <= file_dict <= 850):
        return 8
    if (851 <= file_dict <= 900):
        return 9
    if (901 <= file_dict <= 950):
        return 10
    if (951 <= file_dict <= 1000):
        return 11
    if (1001 <= file_dict <= 1050):
        return 12
    if (1051 <= file_dict <= 1100):
        return 13
    if (1101 <= file_dict <= 1150):
        return 14
    if (1151 <= file_dict <= 1200):
        return 15
    if (1201 <= file_dict <= 1250):
        return 16
    if (1251 <= file_dict <= 1300):
        return 17
    if (1301 <= file_dict <= 1350):
        return 18
    if (1351 <= file_dict <= 1400):
        return 19
    if (1401 <= file_dict <= 1440):
        return -1
    if (1441 <= file_dict <= 1520):
        return 20
    if (1521 <= file_dict <= 1600):
        return 21
    if (1601 <= file_dict <= 1620):
        return 22
    if (1621 <= file_dict <= 1640):
        return 23
    if (1641 <= file_dict <= 1700):
        return 24
    if (1701 <= file_dict <= 1760):
        return 25
    if (1761 <= file_dict <= 1780):
        return 26
    if (1781 <= file_dict <= 1800):
        return 27
    if (1801 <= file_dict <= 1820):
        return 28
    if (1821 <= file_dict <= 1840):
        return 29
    if (1841 <= file_dict <= 1900):
        return 30
    if (1901 <= file_dict <= 1960):
        return 31
    if (1961 <= file_dict <= 1980):
        return 32
    if (1981 <= file_dict <= 2000):
        return 33
    if (2001 <= file_dict <= 2050):
        return 34
    if (2051 <= file_dict <= 2100):
        return 35
    if (2101 <= file_dict <= 2150):
        return 36
    if (2151 <= file_dict <= 2200):
        return 37
    if (2201 <= file_dict <= 2250):
        return 38
    if (2251 <= file_dict <= 2300):
        return 39
    if (2301 <= file_dict <= 2350):
        return 40
    if (2351 <= file_dict <= 2400):
        return 41
    if (2401 <= file_dict <= 2450):
        return 42
    if (2451 <= file_dict <= 2500):
        return 43
    if (2501 <= file_dict <= 2550):
        return 44
    if (2551 <= file_dict <= 2600):
        return 45

def get_camera(path):
    path = os.path.basename(path)
    number,ext = os.path.splitext(path)
    number = get_condition_number(int(number))
    try:
        if(number % 2 == 0):
            return camera_matrix_tan, camera_distortion_tan
        else:
            return camera_matrix_chen, camera_distortion_chen
    except:
        st()

# 图像文件所在的文件夹路径
image_folder_path = '/home/hgh/hghData/Datasets2/Photo'

# 预处理后的数据存储路径
save_dir = '/home/hgh/hghData/Datasets2/preprocessed_images'
os.makedirs(save_dir, exist_ok=True)

# 数据集列表
res = []
load_labels = []
# 标签列表
csv_file_path = '/home/hgh/hghData/Datasets2/coordinate.csv'
df = pd.read_csv(csv_file_path, header=None)
preds = gaze_normalize.xmodel()
# 遍历图像文件夹
for filename in sorted(os.listdir(image_folder_path), key=lambda x: int(os.path.splitext(x)[0])):
    if filename.endswith(".jpg"):
        # 构建图像文件的完整路径
        image_path = os.path.join(image_folder_path, filename)
        print(image_path)
        try:
            row = df.iloc[int(os.path.splitext(filename)[0]) - 1].tolist()
            label = (int(row[3]),int(row[4]))
        except:
            st()
        print(label)

        camera_matrix,camera_distortion = get_camera(image_path)
        gaze_normalize_new = gaze_normalize.GazeNormalize(filename,label,camera_matrix,camera_distortion,preds)
        save_path = os.path.join('/home/hgh/hghData/pre_3_5', f'{filename}')
        warp_image = gaze_normalize_new.norm(image_folder_path)
        if gaze_normalize_new.err == False:
            cv2.imwrite(save_path, warp_image)
            res.append(gaze_normalize_new)
        
with open('/home/hgh/hghData/all_3_5.pkl', 'wb') as fo:
    pickle.dump(res,fo)

print('Preprocessing and saving complete.')