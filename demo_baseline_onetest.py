import os
import cv2
import dlib
from imutils import face_utils
import numpy as np
import torch
from torchvision import transforms
from model import gaze_network
import warp_norm
import pandas as pd
import time
import pickle


trans = transforms.Compose([
        transforms.ToPILImage(),
        transforms.ToTensor(),  # this also convert pixel value from [0,255] to [0,1]
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])

cam_chen = './testpart/camChen.xml'  # this is camera calibration information file obtained with OpenCV
fs_chen = cv2.FileStorage(cam_chen, cv2.FILE_STORAGE_READ)
w_chen = 1300
h_chen = 720
pixel_scale_chen = np.array([0.22, 0.235])
camera_matrix_chen = fs_chen.getNode('Camera_Matrix').mat() # camera calibration information is used for data normalization
camera_distortion_chen = fs_chen.getNode('Distortion_Coefficients').mat()
cam_tan = './testpart/camTan.xml'  # this is camera calibration information file obtained with OpenCV
fs_tan = cv2.FileStorage(cam_tan, cv2.FILE_STORAGE_READ)
w_tan = 1600
h_tan = 825
pixel_scale_tan = np.array([0.202, 0.224])
camera_matrix_tan = fs_tan.getNode('Camera_Matrix').mat() # camera calibration information is used for data normalization
camera_distortion_tan = fs_tan.getNode('Distortion_Coefficients').mat()

if __name__ == '__main__':
    faces = []
    for i in range(50):
        image_id  = 2101+i
        image_path = '/home/hgh/hghData/Datasets2/preprocessed_images/preprocessed_image_'+ str(image_id)+'.jpg'
        # print(target_row)
        print('load input face image: ', image_path)
        sub = 'Tan'
        start_time = time.time()

        image = cv2.imread(image_path)


        if sub == 'Tan':
            camera_matrix = camera_matrix_tan
            camera_distortion = camera_distortion_tan
            w = w_tan
            h = h_tan
        else:
            camera_matrix = camera_matrix_chen
            camera_distortion = camera_distortion_chen
            w = w_chen
            h = h_chen
        gc = np.array([-1,-1])
        model1,model2,model3 = warp_norm.xmodel()
        img_normalized, gcn, R, Ear, Face= warp_norm.GazeNormalization(image, camera_matrix, camera_distortion,gc, w, h,predictor=model1,face_detector=model2, eve_detector= model3)
        faces.append(Face)
        print('load gaze estimator')
        model = gaze_network()
        model.cuda()
        pre_trained_model_path = './ckpt/epoch_24_ckpt.pth.tar'
        if not os.path.isfile(pre_trained_model_path):
            print('the pre-trained gaze estimation model does not exist.')
            exit(0)
        else:
            print('load the pre-trained model: ', pre_trained_model_path)
        ckpt = torch.load(pre_trained_model_path)
        model.load_state_dict(ckpt['model_state'], strict=True)  # load the pre-trained model
        model.eval()  # change it to the evaluation mode
        input_var = img_normalized[:, :, [2, 1, 0]]  # from BGR to RGB
        input_var = trans(input_var)
        input_var = torch.autograd.Variable(input_var.float().cuda())
        input_var = input_var.view(1, input_var.size(0), input_var.size(1), input_var.size(2))  # the input must be 4-dimension
        pred_gaze = model(input_var)  # get the output gaze direction, this is 2D output as pitch and raw rotation
        pred_gaze = pred_gaze[0] # here we assume there is only one face inside the image, then the first one is the prediction
        pred_gaze_np = pred_gaze.cpu().data.numpy()  # convert the pytorch tensor to numpy array
        end_time = time.time()
        elapsed_time = end_time - start_time
        print(f"Elapsed Time: {elapsed_time} seconds")
        print('prepare the output')
        print('Predict normalization gaze vector(pitch yaw):', pred_gaze_np * 180.0 / np.pi)
        print('True normalization gaze vector(pitch yaw):', warp_norm.vector_to_pitchyaw(np.array([gcn]))[0])
        e = warp_norm.angular_error(np.array([pred_gaze_np]),warp_norm.vector_to_pitchyaw(np.array([gcn])))[0]
        print('error:', e)


        print('true vector', gcn)
        print('pred vector', warp_norm.pitchyaw_to_vector(np.array([pred_gaze_np])))



        # print('Normalization true gaze point:', warp_norm.vector_to_gc(gcn.reshape((1,3))[0],w,h))
        # print('Normalization pred gaze point:', warp_norm.vector_to_gc(pred_gaze_np,w,h))
        

        # save the normalization image
        face_patch_gaze = warp_norm.draw_gaze(img_normalized, pred_gaze_np)  # draw gaze direction on the normalized face image
        # face_patch_gaze = warp_norm.draw_gaze(img_normalized, gcn, color=(0,255,0))  # draw gaze direction on the normalized face image
        output_path = './test/results_gaze_'+ str(image_id) + '.jpg'
        print('save output image to: ', output_path)
        cv2.imwrite(output_path, face_patch_gaze)
        
        # compare the gaze point
        print('Original Gaze Point:', gc)
        pred_gaze = warp_norm.pitchyaw_to_vector(np.array([pred_gaze_np]))[0]
        pred_gaze2 = np.dot(np.linalg.inv(R),pred_gaze)
        pred_gaze_point = warp_norm.vector_to_gc(pred_gaze2,pixel_scale_tan)
        pred_gaze_point += [800, 0]
        print('Fake Predict Gaze Point:', warp_norm.vector_to_gc(pred_gaze,pixel_scale_tan) + [800,0])
        print('Predict Gaze Point:', pred_gaze_point)
    with open('./tan_face_data_outscreen.pkl', 'wb') as file:
        pickle.dump(faces, file)