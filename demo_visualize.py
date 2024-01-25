import os
import pandas as pd
import numpy as np
import pickle
import warp_norm
import utils
import cv2

pred = []
pred_path = './results.txt'

with open(pred_path, 'r') as file:
    lines = file.readlines()

for line in lines:
    data_array = [float(x) for x in line.split()]
    pred.append(data_array)
pred = np.array(pred)


pickle_file_path = '/home/hgh/hghData/Datasets/dataset_dict.pkl'
with open(pickle_file_path, 'rb') as file:
    loaded_dataset_dict = pickle.load(file)

image_path = '/home/hgh/hghData/Datasets/preprocessed_images'
save_path = '/home/hgh/hghData/Datasets/visual'

idx = 0
for data in loaded_dataset_dict:
    # print(f"Sample {key}: {value}")
    # 取出男生的数据
    image_name = os.path.basename(data['image_path'])
    print(image_name)
    # print(image_name)
    # number = int((int(os.path.splitext(image_name)[0][19:]) - 1)/100)
    ground_truth = data['label'].reshape((1,3))[0]
    ground_truth = warp_norm.vector_to_pitchyaw(np.array([ground_truth]))
    img = cv2.imread(os.path.join(image_path, image_name))
    warp_norm.draw_gaze(img, pred[idx])
    warp_norm.draw_gaze(img, ground_truth[0], color=(0,255,0))
    cv2.imwrite(os.path.join(save_path, image_name), img)
    idx += 1
