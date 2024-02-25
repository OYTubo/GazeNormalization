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
import pandas as pd
import copy
import os
from pt_module import StNet,StRefine

colors = plt.cm.viridis(np.linspace(0, 1, 4))
plt.rcParams['font.sans-serif']=['DejaVu Sans'] #用来正常显示中文标签

model_dir = './ckpt'
state_name = 'spatical_transform_model_fake_eyetracking_dataset_1kg_128_72_error00_valid_random_ii_in_sequence_person_gt_lr_0.1_99_full.pt'
state_path = os.path.join(model_dir,state_name)
condition_label=['glass && upright tan','glass && upright chen',
                 'no glass && upright tan','no glass && upright chen',
                 'glass && not upright tan','glass && not upright chen',
                 'no glass && not upright tan','no glass && not upright chen',
                 'glass && indoor daylight tan', 'glass && indoor daylight chen',
                 'no glass && indoor daylight tan','no glass && indoor daylight chen',
                 'glass && only lamp tan','glass && only lamp chen',
                 'no glass && only lamp tan','no glass && only lamp chen',
                 'glass && only external lighting tan','glass && only external lighting chen',
                 'no glass && only external lighting tan','no glass && only external lighting chen',
                 'Normal indoor lighting at night && glass && no mask tan', 'Normal indoor lighting at night && glass && no mask chen',
                 'Normal indoor lighting at night && glass && large camera distance(70cm+) tan','Normal indoor lighting at night && glass && large camera distance(70cm+) chen',
                 'Normal indoor lighting at night && glass && middle camera distance(45-48cm) tan','Normal indoor lighting at night && glass && middle camera distance(45-48cm) chen',
                 'Normal indoor lighting at night && glass && small camera distance(32-35cm) tan','Normal indoor lighting at night && glass && small camera distance(32-35cm) chen',
                 'Normal indoor lighting at night && glass && large inclination angle(45°) tan','Normal indoor lighting at night && glass && large inclination angle(45°) chen',
                 'Normal indoor lighting at night && glass && middle inclination angle(30°) tan','Normal indoor lighting at night && glass && middle inclination angle(30°) chen',
                 'Normal indoor lighting at night && glass && small inclination angle(15°) tan','Normal indoor lighting at night && glass && small inclination angle(15°) chen',
                 'Normal indoor lighting at night && glass && multi device normal distance(1cm<x<10cm)gaze at the computer tan','Normal indoor lighting at night && glass && multi device normal distance(1cm<x<10cm)gaze at the computer chen',
                 'Normal indoor lighting at night && glass && multi device normal distance(1cm<x<10cm)gaze at the phone tan','Normal indoor lighting at night && glass && multi device normal distance(1cm<x<10cm)gaze at the phone chen',
                 'Normal indoor lighting at night && glass && multi device large distance(x<1cm)gaze at the computer tan','Normal indoor lighting at night && glass && multi device large distance(x<1cm)gaze at the computer chen',
                 'Normal indoor lighting at night && glass && multi device large distance(x<1cm)gaze at the phone tan','Normal indoor lighting at night && glass && multi device large distance(x<1cm)gaze at the phone chen',
                 'Normal indoor lighting at night && glass && multi device phone laid(x<1cm)gaze at the computer tan','Normal indoor lighting at night && glass && multi device phone laid(x<1cm)gaze at the computer chen',
                 'Normal indoor lighting at night && glass && multi device phone laid(x<1cm)gaze at the phone tan','Normal indoor lighting at night && glass && multi device phone laid(x<1cm)gaze at the phone chen']
print(len(condition_label))

# 读取数据
with open('./gaze_pred_new.pkl', 'rb') as fo:
    tinydict = pickle.load(fo, encoding='bytes')#que
file_names = tinydict['file_name']
file_dict = []
for file_name in file_names:
    file_name = str(file_name).strip('()').split(',')
    # print(file_name)
    for file in file_name:
        if(file!=''):
            # print(file.strip("''").split('/')[4].split('.')[0][40:])
            # print(file.strip("''").split('/')[4].split('.')[0][40:])
            file_dict.append(int(file.strip("''").split('/')[6].split('.')[0][19:]))
print(file_dict)
print(len(file_dict))

with open('/home/hgh/hghData/Datasets/dataset_dict_eve.pkl', 'rb') as fo:
    tinydict2 = pickle.load(fo, encoding='bytes')#que

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

ground_truth = [[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[]]
pred = [[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[]]
RMat = [[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[]]

for i in range(len(file_dict)):
    number = get_condition_number(file_dict[i])
    ground_truth[number].append(tinydict['label'][i])
    pred[number].append(tinydict['pred_gaze'][i])
    RMat[number].append(tinydict2[i]['R'])

# for i in range(46):
#     print(ground_truth[i])



for i in range(len(ground_truth)):
    ground_truth[i] = np.vstack(ground_truth[i])
    pred[i] = np.vstack(pred[i])
# print(RMat[0][0])


# 将pitchyaw转换成vector
for i in range(len(ground_truth)):
    pred[i] = warp_norm.pitchyaw_to_vector(pred[i])

# print(pred[0][0])


epi = 0.7
# 将归一化向量还原
org_pred =[[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[]]
for i in range(len(ground_truth)):
    for j in range(len(RMat[i])):
        # print(RMat[i][j])
        RMat[i][j][2] *= epi
        org_pred[i].append(np.dot(np.linalg.inv(RMat[i][j]), pred[i][j].T))
# print(org_pred[0][0])

pixel_scale_tan = np.array([0.202 , 0.224])
pixel_scale_chen = np.array([0.22 , 0.235])

pred_gc = [[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[]]
for i in range(len(ground_truth)):
    for j in range(len(pred[i])):
        if i%2 == 0:
            pred_gc[i].append(warp_norm.vector_to_gc(org_pred[i][j], pixel_scale_tan))
        else:
            pred_gc[i].append(warp_norm.vector_to_gc(org_pred[i][j], pixel_scale_chen))
# print(pred_gc[0])


#
# i = 0
# for j in range(len(pred[i])):
#     if j < 50:
#         plt.scatter(pred_gc[i][j][0], pred_gc[i][j][1], marker='o',color = colors[i], label=f'Pred')
#         plt.scatter(ground_truth[i][j][0], ground_truth[i][j][1], marker='x',color = colors[i], label=f'True')
#         plt.arrow(ground_truth[i][j][0], ground_truth[i][j][1], pred_gc[i][j][0] - ground_truth[i][j][0], pred_gc[i][j][1] - ground_truth[i][j][1], color=colors[i], alpha=0.5)
#
# plt.title('True and Pred pog')
# plt.xlabel('X')
# plt.ylabel('Y')
# # plt.legend()
#
# plt.tight_layout()
# plt.show()





org_tan = np.array([800,0])#tan 1600*825
org_chen = np.array([650,0])#chen 1300*720


pred_gc_org = [[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[]]
for i in range(46):
    if i%2 == 0:
        pred_gc_org[i] = org_tan + pred_gc[i]
    else:
        pred_gc_org[i] = org_chen + pred_gc[i]

# aver_errors=[]
# for i in range(46):
#     total_errors=0
#     for j in range(len(pred[i])):
#         total_errors=total_errors+((pred_gc_org[i][j][0]-ground_truth[i][j][0])**2+(pred_gc_org[i][j][1]-ground_truth[i][j][1])**2)**0.5
#     aver_errors.append(total_errors/len(pred[i]))
# print(aver_errors)


# aver_errors=[]
# for i in range(46):
#     total_errors=0
#     for j in range(int(len(pred[i])/2),len(pred[i])):
#         total_errors=total_errors+((pred_gc_org[i][j][0]-ground_truth[i][j][0])**2+(pred_gc_org[i][j][1]-ground_truth[i][j][1])**2)**0.5
#     aver_errors.append(total_errors/(len(pred[i])-int(len(pred[i])/2)))
# print(aver_errors)


pred_errors=[]
pred_xerrors=[]
pred_yerrors=[]
for i in range(46):
    total_errors=0
    total_xerrors = 0
    total_yerrors = 0
    for j in range(len(pred[i])):
        total_errors=total_errors+((pred_gc_org[i][j][0]-ground_truth[i][j][0])**2+(pred_gc_org[i][j][1]-ground_truth[i][j][1])**2)**0.5
        total_xerrors=total_xerrors+abs(pred_gc_org[i][j][0]-ground_truth[i][j][0])
        total_yerrors = total_yerrors + abs(pred_gc_org[i][j][1] - ground_truth[i][j][1])
    pred_errors.append(total_errors/(len(pred[i])))
    pred_xerrors.append(total_xerrors / (len(pred[i])))
    pred_yerrors.append(total_yerrors / (len(pred[i])))
pred_xerrors_cm=[]
pred_yerrors_cm=[]
for i in range(46):
    if i % 2 == 0:
        pred_xerrors_cm.append(pred_xerrors[i] * 0.1 * pixel_scale_tan[0])
        pred_yerrors_cm.append(pred_yerrors[i] * 0.1 * pixel_scale_tan[1])
    else:
        pred_xerrors_cm.append(pred_xerrors[i] * 0.1 * pixel_scale_chen[0])
        pred_yerrors_cm.append(pred_yerrors[i] * 0.1 * pixel_scale_chen[1])
# print(pred_errors)
print('pred errors:')
print(pred_xerrors_cm)
print(pred_yerrors_cm)

in_screen_net = [[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[]]
for i in range(46):
    for j in range(len(pred[i])):
        if i%2==0:#tan
            if (0 - 0.05 * 1600 <= pred_gc_org[i][j][0] <= 1600 + 0.05 * 1600 and 0 - 0.05 * 825 <= pred_gc_org[i][j][1] <= 825 + 0.05 * 825):
                in_screen_net[i].append(1)
            else:
                in_screen_net[i].append(0)
        else:
            if (0 - 0.05 * 1300 <= pred_gc_org[i][j][0] <= 1300 + 0.05 * 1300 and 0 - 0.05 * 720 <= pred_gc_org[i][j][1] <= 720 + 0.05 * 720):
                in_screen_net[i].append(1)
            else:
                in_screen_net[i].append(0)
# print(in_screen)

device_level_accuracy=[]
for i in range(46):
    total_in=0
    for j in range(len(pred[i])):
        total_in=total_in+in_screen_net[i][j]
    device_level_accuracy.append(total_in/len(pred[i]))
    if(i==36 or i==37 or i==40 or i==41 or i==44 or i==45):
        device_level_accuracy[i]=1-device_level_accuracy[i]
print(device_level_accuracy)


#before SC
# for i in range(46):
#     plt.figure(figsize=(10, 12))
#     for j in range(len(pred[i])):
#         plt.scatter(pred_gc_org[i][j][0], pred_gc_org[i][j][1], marker='o',color = colors[i], label=f'Measured')
#         plt.scatter(ground_truth[i][j][0], ground_truth[i][j][1], marker='x',color = colors[i], label=f'True')
#         plt.arrow(ground_truth[i][j][0], ground_truth[i][j][1], pred_gc_org[i][j][0] - ground_truth[i][j][0], pred_gc_org[i][j][1] - ground_truth[i][j][1], color=colors[i], alpha=0.5)
#         if(j==0):
#             plt.legend()
#     plt.title(f'Error {aver_errors[i]}', fontsize=10)
#     plt.suptitle(f'Condition {i+1} True and Pred PoG', fontsize=20)
#     plt.xlabel('X')
#     plt.ylabel('Y')
#     # plt.legend()
#     plt.tight_layout()
#     plt.show()


#SC Module
gtr=[[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[]]
aver_pred=[[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[]]
offset=[[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[]]
for i in range(46):
    total_truth=[0,0]
    total_pred=[0,0]
    for j in range(int(len(pred[i])/2)):
        total_truth=total_truth + ground_truth[i][j]
        total_pred = total_pred + pred_gc_org[i][j]
    if i%2==0:
        gtr[i] = [800, 412.5]
    else:
        gtr[i] = [650, 360]
    aver_pred[i] = total_pred / int(len(pred[i])/2)
    if i < 8:
        offset[i] = aver_pred[i % 4] - gtr[i]
    elif 8 <= i < 20:
        offset[i] = aver_pred[i] - gtr[i]
    else:
        offset[i] = aver_pred[20+i%2] - gtr[i]

# print(gtr)
# print(aver_pred)
# print(offset)

refine_pred=[[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[]]
for i in range(46):
    refine_pred[i]=pred_gc_org[i] - offset[i]

# SC_errors=[]
# for i in range(23):
#     total_errors=0
#     for j in range(int(len(pred[i])/2),len(pred[i])):
#         total_errors=total_errors+((refine_pred[i][j][0]-ground_truth[i][j][0])**2+(refine_pred[i][j][1]-ground_truth[i][j][1])**2)**0.5
#     SC_errors.append(total_errors/(len(pred[i])-int(len(pred[i])/2)))
# print(SC_errors)



SC_errors=[]
SC_xerrors=[]
SC_yerrors=[]
for i in range(46):
    total_errors = 0
    total_xerrors = 0
    total_yerrors = 0
    for j in range(len(pred[i])):
        total_errors=total_errors+((refine_pred[i][j][0]-ground_truth[i][j][0])**2+(refine_pred[i][j][1]-ground_truth[i][j][1])**2)**0.5
        total_xerrors = total_xerrors + abs(refine_pred[i][j][0] - ground_truth[i][j][0])
        total_yerrors = total_yerrors + abs(refine_pred[i][j][1] - ground_truth[i][j][1])
    SC_errors.append(total_errors/len(pred[i]))
    SC_xerrors.append(total_xerrors / len(pred[i]))
    SC_yerrors.append(total_yerrors / len(pred[i]))
SC_xerrors_cm=[]
SC_yerrors_cm=[]
for i in range(46):
    if i % 2 == 0:
        SC_xerrors_cm.append(SC_xerrors[i] * 0.1 * pixel_scale_tan[0])
        SC_yerrors_cm.append(SC_yerrors[i] * 0.1 * pixel_scale_tan[1])
    else:
        SC_xerrors_cm.append(SC_xerrors[i] * 0.1 * pixel_scale_chen[0])
        SC_yerrors_cm.append(SC_yerrors[i] * 0.1 * pixel_scale_chen[1])
print('SC errors:')
# print(SC_errors_cm)
print(SC_xerrors_cm)
print(SC_yerrors_cm)

in_screen_SC = [[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[]]
for i in range(46):
    for j in range(len(pred[i])):
        if i%2==0:#tan
            if (0 - 0.05 * 1600 <= refine_pred[i][j][0] <= 1600 + 0.05 * 1600 and 0 - 0.05 * 825 <= refine_pred[i][j][1] <= 825 + 0.05 * 825):
                in_screen_SC[i].append(1)
            else:
                in_screen_SC[i].append(0)
        else:
            if (0 - 0.05 * 1300 <= refine_pred[i][j][0] <= 1300 + 0.05 * 1300 and 0 - 0.05 * 720 <= refine_pred[i][j][1] <= 720 + 0.05 * 720):
                in_screen_SC[i].append(1)
            else:
                in_screen_SC[i].append(0)
# print(in_screen)

device_level_accuracy=[]
for i in range(46):
    total_in=0
    for j in range(len(pred[i])):
        total_in=total_in+in_screen_SC[i][j]
    device_level_accuracy.append(total_in/len(pred[i]))
    if(i==36 or i==37 or i==40 or i==41 or i==44 or i==45):
        device_level_accuracy[i]=1-device_level_accuracy[i]
print(device_level_accuracy)


# for i in range(46):
#     plt.figure(figsize=(10, 12))
#     for j in range(len(pred[i])):
#         plt.scatter(refine_pred[i][j][0], refine_pred[i][j][1], marker='o',color = colors[i], label=f'Pred')
#         plt.scatter(ground_truth[i][j][0], ground_truth[i][j][1], marker='x',color = colors[i], label=f'True')
#         plt.arrow(ground_truth[i][j][0], ground_truth[i][j][1], refine_pred[i][j][0] - ground_truth[i][j][0], refine_pred[i][j][1] - ground_truth[i][j][1], color=colors[i], alpha=0.5)
#         if (j == 0):
#             plt.legend()
#     plt.title(f'Error {SC_errors[i]}', fontsize=10)
#     plt.suptitle(f'Condition {i + 1} True and Refine_Pred PoG', fontsize=20)
#     plt.xlabel('X')
#     plt.ylabel('Y')
#     # plt.legend()
#     plt.tight_layout()
#     plt.show()




#history_heatmap
def create_history_gaze_path_map(PoG_pxs, history_trajectory_map_size=(256, 144), actual_screen_size=(1920, 1080),
                                 guassian_blur=(15, 15)):
    # xys = sample['PoG_history_gt'][sample['PoG_history_gt_validity']]
    xys = PoG_pxs
    # history_trajectory_map_size = 256, 144
    # actual_screen_size = 1920, 1080
    w, h = 256, 144

    trajmap = np.zeros((h, w))
    xys_copy = copy.deepcopy(xys)

    xys_copy[:, 0] *= (w / actual_screen_size[0])
    xys_copy[:, 1] *= (h / actual_screen_size[1])
    arrPt = np.array(xys_copy, np.int32).reshape((-1, 1, 2))

    trajmap = cv2.polylines(trajmap, [arrPt], isClosed=False, color=(1.0,), thickness=2)
    trajmap = cv2.GaussianBlur(trajmap, guassian_blur, 3)
    if (w, h) != history_trajectory_map_size:
        trajmap = cv2.resize(trajmap, history_trajectory_map_size)
    trajmap = normalise_arr(trajmap)
    plt.imshow(trajmap, origin='upper')
    plt.show()
    trajmap.shape
    return trajmap

def normalise_arr(arr):
    mmax, mmin = np.max(arr), np.min(arr)
    assert mmax > mmin
    arr = (arr - mmin +1e-8)/(mmax - mmin + 2e-8)
    return arr

history=[[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[]]
for i in range(46):
    for j in range(int(len(refine_pred[i]) / 2)):
        history[i].append(np.array(refine_pred[i][j]))
    # print(np.array(history[i]))
# create_history_gaze_path_map(history[0])



#PT Module
st_refine_tan = StRefine(StNet_path=state_path, full_screen_size=(1600, 825))
st_refine_chen = StRefine(StNet_path=state_path, full_screen_size=(1300, 720))
# result,valid=st_refine.refine(np.array(refine_pred[0][0]),history[0])
# print(refine_pred[0][0])
# print(result.detach().cpu().numpy())


final_pred=[[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[]]
for i in range(46):
    for j in range(len(refine_pred[i])):
        if i < 20:
            if i % 2 == 0:
                output, valid = st_refine_tan.refine(np.array(refine_pred[i][j]), np.array(history[i % 4]))
            else:
                output, valid = st_refine_chen.refine(np.array(refine_pred[i][j]), np.array(history[i % 4]))
        else:
            if i % 2 == 0:
                output, valid = st_refine_tan.refine(np.array(refine_pred[i][j]), np.array(history[i % 2]))
            else:
                output, valid = st_refine_chen.refine(np.array(refine_pred[i][j]), np.array(history[i % 2]))

        # if(i==2 or i==3):
        #     output, valid = st_refine.refine(np.array(refine_pred[i][j]), np.array(history[i-2]))
        # else:
        #     output, valid = st_refine.refine(np.array(refine_pred[i][j]), np.array(history[i]))
        # print(refine_pred[i][j],output,valid)
        # print(output.detach().cpu().numpy())
        if valid:
            final_pred[i].append(output.detach().cpu().numpy())
        else:
            final_pred[i].append(output)
    # print(np.array(final_pred[i]).shape)


aver_pred=[[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[]]

for i in range(46):
    total_xy=[0,0]
    for j in range(len(pred[i])):
        total_xy=total_xy+final_pred[i][j]
    aver_pred[i]=total_xy/len(pred[i])
# print(aver_pred)

PT_errors=[]
for i in range(46):
    total_errors=0
    for j in range(len(pred[i])):
        total_errors=total_errors+((final_pred[i][j][0]-ground_truth[i][j][0])**2+(final_pred[i][j][1]-ground_truth[i][j][1])**2)**0.5
    PT_errors.append(total_errors/len(pred[i]))
# PT_errors_cm = np.multiply(PT_errors, 0.0264583333)
# print(PT_errors_cm)

for i in range(46):
    for j in range(len(pred[i])):
        if i % 2 == 0:
            final_pred[i][j] = pixel_scale_tan / 2 + (final_pred[i][j] - pixel_scale_tan / 2)/0.7
        else:
            final_pred[i][j] = pixel_scale_chen / 2 + (final_pred[i][j] - pixel_scale_chen / 2)/0.7





PT_magnify_errors=[]
PT_magnify_xerrors=[]
PT_magnify_yerrors=[]
for i in range(46):
    total_errors = 0
    total_xerrors = 0
    total_yerrors = 0

    for j in range(len(pred[i])):
        total_errors=total_errors+((final_pred[i][j][0]-ground_truth[i][j][0])**2+(final_pred[i][j][1]-ground_truth[i][j][1])**2)**0.5
        total_xerrors = total_xerrors +abs(final_pred[i][j][0] - ground_truth[i][j][0])
        total_yerrors = total_yerrors + abs(final_pred[i][j][1] - ground_truth[i][j][1])
    PT_magnify_errors.append(total_errors/len(pred[i]))
    PT_magnify_xerrors.append(total_xerrors / len(pred[i]))
    PT_magnify_yerrors.append(total_yerrors / len(pred[i]))

PT_magnify_xerrors_cm=[]
PT_magnify_yerrors_cm=[]
for i in range(46):
    if i % 2 == 0:
        PT_magnify_xerrors_cm.append(PT_magnify_xerrors[i] * 0.1*pixel_scale_tan[0])
        PT_magnify_yerrors_cm.append(PT_magnify_yerrors[i] * 0.1*pixel_scale_tan[1])
    else:
        PT_magnify_xerrors_cm.append(PT_magnify_xerrors[i] * 0.1*pixel_scale_chen[0])
        PT_magnify_yerrors_cm.append(PT_magnify_yerrors[i] * 0.1*pixel_scale_chen[1])
print('PT errors:')
# print(PT_magnify_errors_cm)
print(PT_magnify_xerrors_cm)
print(PT_magnify_yerrors_cm)



in_screen_PT = [[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[]]
for i in range(46):
    for j in range(len(pred[i])):
        if i%2==0:#tan
            if (0 - 0.05 * 1600 <= final_pred[i][j][0] <= 1600 + 0.05 * 1600 and 0 - 0.05 * 825 <= final_pred[i][j][1] <= 825 + 0.05 * 825):
                in_screen_PT[i].append(1)
            else:
                in_screen_PT[i].append(0)
        else:
            if (0 - 0.05 * 1300 <= final_pred[i][j][0] <= 1300 + 0.05 * 1300 and 0 - 0.05 * 720 <= final_pred[i][j][1] <= 720 + 0.05 * 720):
                in_screen_PT[i].append(1)
            else:
                in_screen_PT[i].append(0)
# print(in_screen)

device_level_accuracy=[]
for i in range(46):
    total_in=0
    for j in range(len(pred[i])):
        total_in=total_in+in_screen_PT[i][j]
    device_level_accuracy.append(total_in/len(pred[i]))
    if(i==36 or i==37 or i==40 or i==41 or i==44 or i==45):
        device_level_accuracy[i]=1-device_level_accuracy[i]
print(device_level_accuracy)

# pred_gc_org
# i=37
# fig = plt.figure(figsize=(10, 12))
# ax = fig.add_subplot()
# if i%2==0:
#     rect = plt.Rectangle((0, 0), 1600, 825, edgecolor='r', facecolor='None')
# else:
#     rect = plt.Rectangle((0, 0), 1300, 720, edgecolor='r', facecolor='None')
# ax.add_patch(rect)
# for j in range(len(pred[i])):
#     plt.scatter(pred_gc_org[i][j][0], pred_gc_org[i][j][1], marker='o', color=colors[0], label=f'gaze at the phone Pred')
#     plt.scatter(pred_gc_org[i%2][j][0], pred_gc_org[i%2][j][1], marker='o', color=colors[2], label=f'upright Pred')
#     # plt.scatter(refine_pred[i][j][0], refine_pred[i][j][1], marker='o', color=colors[0], label=f'gaze at the phone SC Pred')
#     # plt.scatter(refine_pred[i%2][j][0], refine_pred[i%2][j][1], marker='o', color=colors[2], label=f'upright SC Pred')
#     # plt.scatter(final_pred[i][j][0], final_pred[i][j][1], marker='o',color = colors[0], label=f'gaze at the phone PT Pred')
#     # plt.scatter(final_pred[i%2][j][0], final_pred[i%2][j][1], marker='o', color=colors[2], label=f'upright PT Pred')
#     plt.scatter(ground_truth[i%2][j][0], ground_truth[i%2][j][1], marker='x',color = colors[1], label=f'upright True')
#     # plt.arrow(ground_truth[i][j][0], ground_truth[i][j][1], refine_pred[i][j][0] - ground_truth[i][j][0], refine_pred[i][j][1] - ground_truth[i][j][1], color=colors[0], alpha=0.5)
#     # plt.arrow(ground_truth[i][j][0], ground_truth[i][j][1], final_pred[i][j][0] - ground_truth[i][j][0], final_pred[i][j][1] - ground_truth[i][j][1], color=colors[2], alpha=0.5)
#     # plt.arrow(ground_truth[i][j][0], ground_truth[i][j][1], pred_gc_org[i][j][0] - ground_truth[i][j][0],pred_gc_org[i][j][1] - ground_truth[i][j][1], color=colors[3], alpha=0.5)
#     if (j == 0):
#         plt.legend()
#
# plt.title(f'SC Error {SC_errors[i]:.4f} ,PT Error {PT_errors[i]:.4f}', fontsize=10)
# plt.suptitle(f'Condition {condition_label[i]} ', fontsize=15)
# plt.xlabel('X')
# plt.ylabel('Y')
# # plt.legend()
# plt.tight_layout()
# plt.show()




for i in range(46):
    fig = plt.figure(figsize=(10, 12))
    ax = fig.add_subplot()
    if i%2==0:
        rect = plt.Rectangle((0, 0), 1600, 825, edgecolor='r', facecolor='None')
    else:
        rect = plt.Rectangle((0, 0), 1300, 720, edgecolor='r', facecolor='None')
    ax.add_patch(rect)
    for j in range(len(pred[i])):
        # plt.scatter(pred_gc_org[i][j][0], pred_gc_org[i][j][1], marker='o', color=colors[3], label=f'Pred')
        plt.scatter(refine_pred[i][j][0], refine_pred[i][j][1], marker='o', color=colors[0], label=f'SC Pred')
        plt.scatter(final_pred[i][j][0], final_pred[i][j][1], marker='o',color = colors[2], label=f'PT Pred')
        plt.scatter(ground_truth[i][j][0], ground_truth[i][j][1], marker='x',color = colors[1], label=f'True')
        plt.arrow(ground_truth[i][j][0], ground_truth[i][j][1], refine_pred[i][j][0] - ground_truth[i][j][0], refine_pred[i][j][1] - ground_truth[i][j][1], color=colors[0], alpha=0.5)
        plt.arrow(ground_truth[i][j][0], ground_truth[i][j][1], final_pred[i][j][0] - ground_truth[i][j][0], final_pred[i][j][1] - ground_truth[i][j][1], color=colors[2], alpha=0.5)
        if (j == 0):
            plt.legend()
    plt.title(f'SC Error {SC_errors[i]:.4f} ,PT Error {PT_errors[i]:.4f}', fontsize=10)
    plt.suptitle(f'{condition_label[i]} ', fontsize=15)
    plt.xlabel('X')
    plt.ylabel('Y')
    # plt.legend()
    plt.tight_layout()
    plt.show()