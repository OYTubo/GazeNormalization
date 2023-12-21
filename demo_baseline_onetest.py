import os
import cv2
import dlib
from imutils import face_utils
import numpy as np
import torch
from torchvision import transforms
from model import gaze_network
import warp_norm


trans = transforms.Compose([
        transforms.ToPILImage(),
        transforms.ToTensor(),  # this also convert pixel value from [0,255] to [0,1]
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])

if __name__ == '__main__':
    img_file_name = '/home/hgh/hghData/Datasets/Photo/2.jpg'
    print('load input face image: ', img_file_name)
    image = cv2.imread(img_file_name)
    cam_tan = '/home/hgh/hghData/Datasets/camTan.xml'  # this is camera calibration information file obtained with OpenCV
    fs_tan = cv2.FileStorage(cam_tan, cv2.FILE_STORAGE_READ)
    camera_matrix_tan = fs_tan.getNode('Camera_Matrix').mat() # camera calibration information is used for data normalization
    camera_distortion_tan = fs_tan.getNode('Distortion_Coefficients').mat()     
    img_normalized, gcn = warp_norm.GazeNormalization(image, camera_matrix_tan, camera_distortion_tan, gc = np.array([1047,572]))
    print('load gaze estimator')
    model = gaze_network()
    model.cuda() # comment this line out if you are not using GPU
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

    print('prepare the output')
    # draw the facial landmarks
    face_patch_gaze = warp_norm.draw_gaze(img_normalized, pred_gaze_np)  # draw gaze direction on the normalized face image
    face_patch_gaze = warp_norm.draw_gaze(img_normalized, gcn, color=(0,255,0))  # draw gaze direction on the normalized face image
    output_path = './test/results_gaze.jpg'
    print('save output image to: ', output_path)
    cv2.imwrite(output_path, face_patch_gaze)
