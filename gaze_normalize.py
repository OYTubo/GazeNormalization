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

def xmodel():
    predictor = dlib.shape_predictor('./modules/shape_predictor_68_face_landmarks.dat')
    face_detector = dlib.get_frontal_face_detector()
    fa = face_alignment.FaceAlignment(face_alignment.LandmarksType.TWO_D, flip_input=False)
    preds = {'landmarks':predictor, 'dlib_detector': face_detector, 'fa_detector': fa}
    return preds

class GazeNormalize():
    def __init__(self, image_name, label, camera_matrix, camera_distortion, predictor):
        self.err = False
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
    
    def norm(self, dataset_path):
        # 1.face detection
        # detected_faces = self.predictor['dlib_detector'](cv2.cvtColor(self.image, cv2.COLOR_BGR2RGB), 1) ## convert BGR image to RGB for dlib
        # if len(detected_faces) == 0:
        #    print('warning: no detected face by dlib')
        self.image_path = os.path.join(dataset_path, self.image_name)
        image = cv2.imread(self.image_path)
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
        Ear = np.mean(np.asarray(Ear))

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
        image = cv2.imread(self.image_path)
        img_warped = cv2.warpPerspective(image, self.W, roiSize)  # image normalization
        return img_warped
    
        