import sys
sys.path.append("./FaceAlignment")
import face_alignment
from imutils import face_utils
import cv2
import dlib
import numpy as np
from ipdb import set_trace as st
import warp_norm
import os
from model import gaze_network
import torch
from torchvision import transforms


def xmodel():
    predictor = dlib.shape_predictor('./modules/shape_predictor_68_face_landmarks.dat')
    face_detector = dlib.get_frontal_face_detector()
    fa = face_alignment.FaceAlignment(face_alignment.LandmarksType.TWO_D, flip_input=False)
    preds = {'landmarks':predictor, 'dlib_detector': face_detector, 'fa_detector': fa}
    return preds

class GazeNormalize():
    def __init__(self, image_name, label, camera_matrix, camera_distortion, predictor, is_video = False):
        self.err = False
        self.is_video = is_video
        self.image_name = image_name
        self.label = label
        self.camera_matrix = camera_matrix
        self.camera_distortion = camera_distortion
        if predictor is None:
            self.predictor = xmodel()
        self.predictor = predictor
    
    def __getstate__(self):
            state = self.__dict__.copy()
            del state['predictor']
            return state
    
    def norm(self, dataset_path = None):
        # 1.face detection
        # detected_faces = self.predictor['dlib_detector'](cv2.cvtColor(self.image, cv2.COLOR_BGR2RGB), 1) ## convert BGR image to RGB for dlib
        # if len(detected_faces) == 0:
        #    print('warning: no detected face by dlib')
        if self.is_video == False:
            self.image_path = os.path.join(dataset_path, self.image_name)
            image = cv2.imread(self.image_path)
        else:
            image = self.image_name
        preds = self.predictor['fa_detector'].get_landmarks(image)
        if preds is None:
            print('warning: no detected face by fa')
            self.hr = np.zeros((1,3))
            self.ht = np.zeros((1,3))
            self.err = True
            return 0
        max_face_index = np.argmax(np.apply_along_axis(self.calculate_face_area, axis=1, arr=preds))
        landmarks = preds[max_face_index]
        # else:
        #     largest_face = max(detected_faces, key=lambda rect: rect.width() * rect.height())
        #     print("max face position:", largest_face)
        #     shape = self.predictor['landmarks'](self.image, largest_face)
        #     shape = face_utils.shape_to_np(shape)
        #     landmarks = []
        #     for (x, y) in shape:
        #         landmarks.append((x, y))
        #     landmarks = np.asarray(landmarks)
        landmarks_sub = landmarks[[36, 39, 42, 45, 31, 35], :]
        landmarks_sub = landmarks_sub.astype(float)
        self.landmarks_sub = landmarks_sub.reshape(6, 1, 2)

        # Get Ear
        Ear = []
        for i in range(2):
            Ear.append((np.linalg.norm(landmarks[41+6*i]-landmarks[37+6*i],2) + np.linalg.norm(landmarks[40+6*i]-landmarks[38+6*i],2))/(2*np.linalg.norm(landmarks[36+6*i]-landmarks[39+6*i],2)))
        self.Ear = np.mean(np.asarray(Ear))

        # 2.Estimate Head Pose
        face_model_load = np.loadtxt('./modules/face_model.txt')  # Generic face model with 3D facial landmarks
        landmark_use = [20, 23, 26, 29, 15, 19]  # we use eye corners and nose conners
        face_model = face_model_load[landmark_use, :]
        facePts = face_model.reshape((6, 1, 3))
        hr,ht = self.estimateHeadPose(facePts)
        self.hr = hr.reshape((1,3))
        self.ht = ht.reshape((1,3))

        # 3.Normalize Image
        self.warp_image = self.xtrans(face_model)

        return self.warp_image
    
    def calculate_face_area(self, face):
        min_x, max_x = np.min(face[0]), np.max(face[0])
        min_y, max_y = np.min(face[1]), np.max(face[1])
        area = (max_x - min_x) * (max_y - min_y)
        return area

    def estimateHeadPose(self, face_model, iterate=True):
        ret, rvec, tvec = cv2.solvePnP(face_model, self.landmarks_sub, self.camera_matrix, self.camera_distortion, flags=cv2.SOLVEPNP_EPNP)
        ## further optimize
        if iterate:
            ret, rvec, tvec = cv2.solvePnP(face_model, self.landmarks_sub, self.camera_matrix, self.camera_distortion, rvec, tvec, True)
        return rvec, tvec

    def xtrans(self, face_model):
        # normalized camera parameters
        focal_norm = 960
        distance_norm = 600  # normalized distance between face and camera
        roiSize = (224, 224)

        ht = self.ht.reshape((3,1))
        hr = self.hr.reshape((3,1))
        hR = cv2.Rodrigues(hr)[0]  # rotation matrix, [3,3]
        Fc = np.dot(hR, face_model.T) + ht # [3,50]
        self.face_center = np.mean(Fc,axis=1).reshape((3, 1))
        # normalize image
        distance = np.linalg.norm(self.face_center)  # actual distance between and original camera
        z_scale = distance_norm / distance

        cam_norm = np.array([
            [focal_norm, 0, roiSize[0] / 2],
            [0, focal_norm, roiSize[1] / 2],
            [0, 0, 1.0],
        ])
        S = np.array([  # scaling matrix
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, z_scale],
        ])

        hRx = hR[:, 0]
        forward = (self.face_center / distance).reshape(3)
        down = np.cross(forward, hRx)
        down /= np.linalg.norm(down)
        right = np.cross(down, forward)
        right /= np.linalg.norm(right)

        self.R = np.c_[right, down, forward].T  # rotation matrix R
        self.W = np.dot(np.dot(cam_norm, S), np.dot(self.R, np.linalg.inv(self.camera_matrix)))  # transformation matrix
        if self.is_video:
            image = self.image_name
        else:
            image = cv2.imread(self.image_path)
        img_warped = cv2.warpPerspective(image, self.W, roiSize)  # image normalization
        return img_warped
    
    def pred(self, model_path, img_warped):
        '''This method is used for real-time video gaze direction prediction'''
        print('load gaze estimator')
        model = gaze_network()
        model.cuda()
        pre_trained_model_path = model_path
        if not os.path.isfile(pre_trained_model_path):
            print('the pre-trained gaze estimation model does not exist.')
            exit(0)
        else:
            print('load the pre-trained model: ', pre_trained_model_path)
        ckpt = torch.load(pre_trained_model_path)
        model.load_state_dict(ckpt['model_state'], strict=True)  # load the pre-trained model
        model.eval()  # change it to the evaluation mode
        input_var = img_warped[:, :, [2, 1, 0]]  # from BGR to RGB
        trans = transforms.Compose([
                transforms.ToPILImage(),
                transforms.ToTensor(),  # this also convert pixel value from [0,255] to [0,1]
                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                            std=[0.229, 0.224, 0.225]),
                ])
        input_var = trans(input_var)
        input_var = torch.autograd.Variable(input_var.float().cuda())
        input_var = input_var.view(1, input_var.size(0), input_var.size(1), input_var.size(2))  # the input must be 4-dimension
        pred_gaze = model(input_var)  # get the output gaze direction, this is 2D output as pitch and raw rotation
        self.pred_gaze = pred_gaze[0].cpu().data.numpy() # here we assume there is only one face inside the image, then the first one is the prediction
        return self.pred_gaze
    
    def draw_gaze(self, thickness = 2, color=(0,0,255)):
        '''Draw gaze angle on given image with a given eye positions.'''
        # 1.Reduction
        gaze_vector = warp_norm.pitchyaw_to_vector(self.pred_gaze.reshape((1,2)))
        org_pred = np.dot(np.linalg.inv(self.R), gaze_vector.T)
        pitchyaw = warp_norm.vector_to_pitchyaw(org_pred.reshape((1,3)))[0]

        # 2.Draw vector
        image_out = self.image_name
        length = 112
        # pos = (int(w / 2.0), int(h / 2.0))
        pos = tuple(np.mean(self.landmarks_sub.reshape((6,2)),axis=0).reshape((1,2)).tolist())[0]
        if len(image_out.shape) == 2 or image_out.shape[2] == 1:  # to draw on the image, we need to convert to RGB
            image_out = cv2.cvtColor(image_out, cv2.COLOR_GRAY2BGR)
        dx = -length * np.sin(pitchyaw[1]) * np.cos(pitchyaw[0])
        dy = -length * np.sin(pitchyaw[0])
        cv2.arrowedLine(image_out, tuple(np.round(pos).astype(int)),
                    tuple(np.round([pos[0] + dx, pos[1] + dy]).astype(int)), color,
                    thickness, cv2.LINE_AA, tipLength=0.2)
        
        # 3.Draw landmarks&EAR
        for landmark in self.landmarks_sub.reshape((6,2)):
            cv2.circle(image_out, tuple(np.round(landmark).astype(int)), 5, (255,0,0), -1)
        # 添加一行文本
        text = 'EAR:{:2f}'.format(self.Ear)
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 1
        font_thickness = 2
        font_color = (255, 255, 255)  # 白色
        text_size = cv2.getTextSize(text, font, font_scale, font_thickness)[0]
        text_position = (image_out.shape[1] - text_size[0] - 10, image_out.shape[0] - 10)
        cv2.putText(image_out, text, text_position, font, font_scale, font_color, font_thickness)
        return image_out