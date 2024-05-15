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
    def __init__(self, image_name, label, camera_matrix, camera_distortion, predictor, is_video = False, image = None):
        self.err = False
        self.is_video = is_video
        self.image_name = image_name
        self.label = label
        self.camera_matrix = camera_matrix
        self.camera_distortion = camera_distortion
        self.image = image
        if predictor is None:
            self.predictor = xmodel()
        self.predictor = predictor
    
    def __getstate__(self):
            state = self.__dict__.copy()
            del state['predictor']
            return state

    def pitchyaw_to_vector(self,pitchyaws):
        r"""Convert given yaw (:math:`\theta`) and pitch (:math:`\phi`) angles to unit gaze vectors.

        Args:
            pitchyaws (:obj:`numpy.array`): yaw and pitch angles :math:`(n\times 2)` in radians.

        Returns:
            :obj:`numpy.array` of shape :math:`(n\times 3)` with 3D vectors per row.
        """
        n = pitchyaws.shape[0]
        sin = np.sin(pitchyaws)
        cos = np.cos(pitchyaws)
        out = np.empty((n, 3))
        out[:, 0] = np.multiply(cos[:, 0], sin[:, 1])
        out[:, 1] = sin[:, 0]
        out[:, 2] = np.multiply(cos[:, 0], cos[:, 1])
        return out

    def norm(self, dataset_path = None):
        # 1.face detection
        # detected_faces = self.predictor['dlib_detector'](cv2.cvtColor(self.image, cv2.COLOR_BGR2RGB), 1) ## convert BGR image to RGB for dlib
        # if len(detected_faces) == 0:
        #    print('warning: no detected face by dlib')

        if self.is_video == False:
            self.image_path = os.path.join(dataset_path, self.image_name)
            image = cv2.imread(self.image_path)
        else:
            image = self.image
            # print(f'image{self.image}')

        predictor = dlib.shape_predictor('./modules/shape_predictor_68_face_landmarks.dat')
        face_detector = dlib.get_frontal_face_detector()
        # face detection
        detected_faces = face_detector(cv2.cvtColor(image, cv2.COLOR_BGR2RGB),1)  ## convert BGR image to RGB for dlib

        shape = predictor(image, detected_faces[0])  ## only use the first detected face (assume that each input image only contains one face)
        shape = face_utils.shape_to_np(shape)
        landmarks = []
        for (x, y) in shape:
            landmarks.append((x, y))
        landmarks = np.asarray(landmarks)

        # preds = self.predictor['fa_detector'].get_landmarks_from_image(image)  #做了修改
        # # preds = self.predictor[].get_landmarks_from_image(image)
        # if preds is None:
        #     print('warning: no detected face by fa')
        #     self.hr = np.zeros((1,3))
        #     self.ht = np.zeros((1,3))
        #     self.err = True
        #     return 0
        # max_face_index = np.argmax(np.apply_along_axis(self.calculate_face_area, axis=1, arr=preds))
        # landmarks = preds[max_face_index]


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
        self.landmarks=landmarks.astype(float)
        self.landmarks_sub = landmarks_sub.reshape(6, 1, 2)

        # Get Ear
        Ear = []
        for i in range(2):
            Ear.append((np.linalg.norm(landmarks[41+6*i]-landmarks[37+6*i],2) + np.linalg.norm(landmarks[40+6*i]-landmarks[38+6*i],2))/(2*np.linalg.norm(landmarks[36+6*i]-landmarks[39+6*i],2)))
        self.Ear = np.mean(np.asarray(Ear))

        # print('ear:{}'.format(self.Ear))

        # 2.Estimate Head Pose
        face_model_load = np.loadtxt('./modules/face_model.txt')  # Generic face model with 3D facial landmarks
        landmark_use = [20, 23, 26, 29, 15, 19]  # we use eye corners and nose conners
        face_model = face_model_load[landmark_use, :]
        landmarks_eye_corner = [20, 23, 26, 29]
        face_model_eye_corner = face_model_load[landmarks_eye_corner, :]
        facePts = face_model.reshape((6, 1, 3))
        hr,ht = self.estimateHeadPose(facePts)
        self.hr = hr.reshape((1,3))
        self.ht = ht.reshape((1,3))

        # 3.Normalize Image
        self.warp_image, Fc, Fc_lip= self.xtrans(face_model,face_model_eye_corner)


        #Get lip distance and eye corner distance
        lip_distance = []
        for i in range(2):
            lip_distance.append(((np.linalg.norm(landmarks[41 + 6 * i] - landmarks[37 + 6 * i], 2) + np.linalg.norm(
                landmarks[40 + 6 * i] - landmarks[38 + 6 * i], 2)) / 2))
        lip_distance = np.mean(np.array(lip_distance))
        landmarks_eye_corner = []
        for i in range(2):
            landmarks_eye_corner.append(np.linalg.norm(landmarks[36 + 6 * i] - landmarks[39 + 6 * i]))
        landmarks_eye_corner = np.mean(np.array(landmarks_eye_corner))

        distance_baseline = (np.linalg.norm(Fc_lip[:, 0] - Fc_lip[:, 1], ord=2, axis=0) + np.linalg.norm(Fc_lip[:, 2] - Fc_lip[:, 3],
                                                                                                 ord=2, axis=0)) / 2
        transfer_facter = distance_baseline / landmarks_eye_corner
        real_eyelip_distance = lip_distance * transfer_facter

        return self.warp_image,real_eyelip_distance,self.Ear,self.R
    
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

    def xtrans(self, face_model,face_model_lip):
        # normalized camera parameters
        focal_norm = 960
        distance_norm = 600  # normalized distance between face and camera
        roiSize = (224, 224)

        ht = self.ht.reshape((3,1))
        hr = self.hr.reshape((3,1))
        hR = cv2.Rodrigues(hr)[0]  # rotation matrix, [3,3]
        Fc = np.dot(hR, face_model.T) + ht # [3,50]
        Fc_lip=np.dot(hR, face_model_lip.T)
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
            image = self.image
        else:
            image = cv2.imread(self.image_path)
        img_warped = cv2.warpPerspective(image, self.W, roiSize)  # image normalization
        return img_warped, Fc ,Fc_lip


    def pred(self, model_in, img_warped):
        '''This method is used for real-time video gaze direction prediction'''
        if model_in == None:
            # print('load gaze estimator')
            model = gaze_network()
            # model.cuda()
            pre_trained_model_path = './ckpt/epoch_24_ckpt.pth.tar'
            if not os.path.isfile(pre_trained_model_path):
                print('the pre-trained gaze estimation model does not exist.')
                exit(0)
            # else:
            #     print('load the pre-trained model: ', pre_trained_model_path)
            ckpt = torch.load(pre_trained_model_path,map_location=torch.device('cpu'))
            model.load_state_dict(ckpt['model_state'], strict=True)  # load the pre-trained model
            model.eval()  # change it to the evaluation mode
        #quan na dao xia mian
        model = gaze_network()
        pre_trained_model_path = './ckpt/epoch_24_ckpt.pth.tar'
        ckpt = torch.load(pre_trained_model_path, map_location=torch.device('cpu'))
        model.load_state_dict(ckpt['model_state'], strict=True)  # load the pre-trained model
        model.eval()  # change it to the evaluation mode
        ##dao zhe li wei zhi
        input_var = img_warped[:, :, [2, 1, 0]]  # from BGR to RGB
        trans = transforms.Compose([
                transforms.ToPILImage(),
                transforms.ToTensor(),  # this also convert pixel value from [0,255] to [0,1]
                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                            std=[0.229, 0.224, 0.225]),
                ])
        input_var = trans(input_var)
        # input_var = torch.autograd.Variable(input_var.float().cuda())
        input_var = torch.autograd.Variable(input_var.float())
        input_var = input_var.view(1, input_var.size(0), input_var.size(1), input_var.size(2))  # the input must be 4-dimension
        model_1=model
        pred_gaze = model_1(input_var)  # get the output gaze direction, this is 2D output as pitch and raw rotation
        self.pred_gaze = pred_gaze[0].cpu().data.numpy() # here we assume there is only one face inside the image, then the first one is the prediction
        return self.pred_gaze
    
    def draw_norm_gaze(self, thickness = 2, color=(0,0,255), frameCount=0, blinkCount=0, blinkLonggest=0,alert=0,data_normalized=[],add_hr = False):
        '''Draw gaze angle on given image with a given eye positions.'''
        # 1.Reduction landmarks
        landmarks = self.landmarks_sub
        landmarks_warped = cv2.perspectiveTransform(landmarks, self.W)
        landmarks_warped = landmarks_warped.reshape((-1, 2))
        pitchyaw = self.pred_gaze.reshape((1,2))[0]

        # 2.Draw vector
        image_out = self.warp_image
        r = 56
        pos = int(224 - 56)
        if len(image_out.shape) == 2 or image_out.shape[2] == 1:  # to draw on the image, we need to convert to RGB
            image_out = cv2.cvtColor(image_out, cv2.COLOR_GRAY2BGR)

        # 处理数据
        theta_circle = np.linspace(0, 2 * np.pi, 100)
        x_circle = r * np.sin(theta_circle)
        y_circle = r * np.cos(theta_circle)
        theta_new = pitchyaw[1]
        phi_new = pitchyaw[0]

        # 根据phi值计算y和z坐标
        x = -r * np.cos(phi_new) * np.sin(theta_new)
        y = r * np.sin(phi_new)
        z = r * np.cos(phi_new) * np.cos(theta_new) 
        # 根据真实坐标还原角度并投影
        # 绘制竖圆
        phi_1 = np.linspace(0, np.pi, 200)
        proj_v = np.array([x,z])
        x_axis = np.array([1, 0])
        cos_theta_1 = np.dot(proj_v,x_axis) /(np.linalg.norm(proj_v) * np.linalg.norm(x_axis))
        theta_1 = np.arccos(cos_theta_1)
        x_1 = r * np.sin(phi_1) * np.cos(theta_1)
        y_1 = r * np.cos(phi_1)

        # 绘制横圆
        phi_2 = np.linspace(0, np.pi, 200)
        proj_y = np.array([y,z])
        y_axis = np.array([1, 0])
        cos_theta_2 = np.dot(proj_y,y_axis) /(np.linalg.norm(proj_y) * np.linalg.norm(y_axis))
        theta_2 = np.arccos(cos_theta_2)
        x_2 = r * np.cos(phi_2)
        y_2 = r * np.sin(phi_2) * np.cos(theta_2)

        # 绘制点和线
        cv2.circle(image_out, (int(x+pos), int(-y+pos)), radius=5, color=(0, 0, 255), thickness=-1)  # 红色点
        cv2.circle(image_out, (int(0+pos), int(0+pos)), radius=1, color=(255, 0, 0), thickness=-1)  # 蓝色点
        cv2.line(image_out, (pos, pos), (int(x+pos), int(-y+pos)), color=(0, 0, 0), lineType = cv2.LINE_AA)  # 黑色虚线

        # 绘制圆形投影（根据x_circle, y_circle的值绘制）
        for i in range(len(x_circle)-1):
            cv2.line(image_out, (int(x_circle[i]+pos), int(-y_circle[i]+pos)), (int(x_circle[i+1]+pos), int(-y_circle[i+1]+pos)), color=(0, 255, 0), thickness=1)
        for i in range(len(x_1) - 1):
            cv2.line(image_out, (int(x_1[i]+pos), int(-y_1[i]+pos)), (int(x_1[i + 1]+pos), int(-y_1[i + 1]+pos)), (0, 0, 255), 2)
        for i in range(len(x_2) - 1):
            cv2.line(image_out, (int(x_2[i]+pos), int(-y_2[i]+pos)), (int(x_2[i + 1]+pos), int(-y_2[i + 1]+pos)), (0, 0, 255), 2)

        # 3.Draw landmarks&EAR
        for landmark in landmarks_warped:
            cv2.circle(image_out, tuple(np.round(landmark).astype(int)), 5, (255,0,0), -1)
        

        # add Hr,Ht
        if add_hr == True:
            hr = cv2.Rodrigues(self.R)[0].reshape((1,3))
            eular = np.ravel(warp_norm.hr_to_pitchyaw(hr))
            text = 'Eular:[{:.2f},{:.2f},{:.2f}]'.format(eular[0],eular[1],eular[2])
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.5
            font_thickness = 2
            font_color = (255, 255, 255)  # 白色
            text_size = cv2.getTextSize(text, font, font_scale, font_thickness)[0]
            text_position = (image_out.shape[1] - text_size[0] - 10, image_out.shape[0] - 10)
            cv2.putText(image_out, text, text_position, font, font_scale, font_color, font_thickness)
            text = 'Ht:[{:.2f},{:.2f},{:.2f}]'.format(self.ht[0][0],self.ht[0][1],self.ht[0][2])
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.5
            font_thickness = 2
            font_color = (255, 255, 255)  # 白色
            text_size = cv2.getTextSize(text, font, font_scale, font_thickness)[0]
            text_position = (image_out.shape[1] - text_size[0] - 10, image_out.shape[0] - 30)
            cv2.putText(image_out, text, text_position, font, font_scale, font_color, font_thickness)
            text2 = 'blinkCount:{:2f}'.format(blinkCount)
            text_position2 = (image_out.shape[1] - text_size[0] - 10, image_out.shape[0] - 30)
            cv2.putText(image_out, text2, text_position2, font, font_scale, font_color, font_thickness)
            text3 = 'frameCount:{:2f}'.format(frameCount)
            text_position3 = (image_out.shape[1] - text_size[0] - 10, image_out.shape[0] - 40)
            cv2.putText(image_out, text3, text_position3, font, font_scale, font_color, font_thickness)
            text4 = 'blinkLonggest:{:2f}'.format(blinkLonggest)
            text_position4 = (image_out.shape[1] - text_size[0] - 10, image_out.shape[0] - 50)
            cv2.putText(image_out, text4, text_position4, font, font_scale, font_color, font_thickness)
            text5 = 'alert:{}'.format(alert)
            text_position5 = (image_out.shape[1] - text_size[0] - 10, image_out.shape[0] - 60)
            cv2.putText(image_out, text5, text_position5, font, font_scale, font_color, font_thickness)
        return image_out


    def draw_gaze(self, thickness = 2, color=(0,0,255), frameCount=0, blinkCount=0, blinkLonggest=0,alert=0,data_normalized=[]):
        '''Draw gaze angle on given image with a given eye positions.'''
        # 1.Reduction
        gaze_vector = warp_norm.pitchyaw_to_vector(self.pred_gaze.reshape((1,2)))
        org_pred = np.dot(np.linalg.inv(self.R), gaze_vector.T) 
        pitchyaw = warp_norm.vector_to_pitchyaw(org_pred.reshape((1,3)))[0]

        # # 2.Draw vector
        image_out = self.image
        # length = 112
        # # pos = (int(w / 2.0), int(h / 2.0))
        # pos = tuple(np.mean(self.landmarks_sub.reshape((6,2)),axis=0).reshape((1,2)).tolist())[0]
        # if len(image_out.shape) == 2 or image_out.shape[2] == 1:  # to draw on the image, we need to convert to RGB
        #     image_out = cv2.cvtColor(image_out, cv2.COLOR_GRAY2BGR)
        # dx = -length * np.sin(pitchyaw[1]) * np.cos(pitchyaw[0])
        # dy = -length * np.sin(pitchyaw[0])
        # cv2.arrowedLine(image_out, tuple(np.round(pos).astype(int)),
        #             tuple(np.round([pos[0] + dx, pos[1] + dy]).astype(int)), color,
        #             thickness, cv2.LINE_AA, tipLength=0.2)

        r = 56
        pos = int(224 - 56)

        theta_circle = np.linspace(0, 2 * np.pi, 100)
        x_circle = r * np.sin(theta_circle)
        y_circle = r * np.cos(theta_circle)
        theta_new = pitchyaw[1]
        phi_new = pitchyaw[0]

        # 根据phi值计算y和z坐标
        x = -r * np.cos(phi_new) * np.sin(theta_new)
        y = r * np.sin(phi_new)
        z = r * np.cos(phi_new) * np.cos(theta_new)
        # 根据真实坐标还原角度并投影
        # 绘制竖圆
        phi_1 = np.linspace(0, np.pi, 200)
        proj_v = np.array([x,z])
        x_axis = np.array([1, 0])
        cos_theta_1 = np.dot(proj_v,x_axis) /(np.linalg.norm(proj_v) * np.linalg.norm(x_axis))
        theta_1 = np.arccos(cos_theta_1)
        x_1 = r * np.sin(phi_1) * np.cos(theta_1)
        y_1 = r * np.cos(phi_1)

        # 绘制横圆
        phi_2 = np.linspace(0, np.pi, 200)
        proj_y = np.array([y,z])
        y_axis = np.array([1, 0])
        cos_theta_2 = np.dot(proj_y,y_axis) /(np.linalg.norm(proj_y) * np.linalg.norm(y_axis))
        theta_2 = np.arccos(cos_theta_2)
        x_2 = r * np.cos(phi_2)
        y_2 = r * np.sin(phi_2) * np.cos(theta_2)

        # 绘制点和线
        cv2.circle(image_out, (int(x+pos), int(-y+pos)), radius=5, color=(0, 0, 255), thickness=-1)  # 红色点
        cv2.circle(image_out, (int(0+pos), int(0+pos)), radius=1, color=(255, 0, 0), thickness=-1)  # 蓝色点
        cv2.line(image_out, (pos, pos), (int(x+pos), int(-y+pos)), color=(0, 0, 0), lineType = cv2.LINE_AA)  # 黑色虚线

        # 绘制圆形投影（根据x_circle, y_circle的值绘制）
        for i in range(len(x_circle)-1):
            cv2.line(image_out, (int(x_circle[i]+pos), int(-y_circle[i]+pos)), (int(x_circle[i+1]+pos), int(-y_circle[i+1]+pos)), color=(0, 255, 0), thickness=1)
        for i in range(len(x_1) - 1):
            cv2.line(image_out, (int(x_1[i]+pos), int(-y_1[i]+pos)), (int(x_1[i + 1]+pos), int(-y_1[i + 1]+pos)), (0, 0, 255), 2)
        for i in range(len(x_2) - 1):
            cv2.line(image_out, (int(x_2[i]+pos), int(-y_2[i]+pos)), (int(x_2[i + 1]+pos), int(-y_2[i + 1]+pos)), (0, 0, 255), 2)

        # 3.Draw landmarks&EAR
        # for landmark in self.landmarks_sub.reshape((6,2)):
        #     cv2.circle(image_out, tuple(np.round(landmark).astype(int)), 1, (255,0,0), -1)
        for landmark in self.landmarks.reshape((68, 2)):
              cv2.circle(image_out, tuple(np.round(landmark).astype(int)), 1, (255,0,0), -1)
        # add EAR
        # text = 'EAR:{:2f}'.format(self.Ear)

        text = 'EAR:{:2f}'.format(self.Ear)
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.4
        font_thickness = 1
        font_color = (255, 255, 255)  # 白色
        text_size = cv2.getTextSize(text, font, font_scale, font_thickness)[0]
        text_position = (image_out.shape[1] - text_size[0] - 10, image_out.shape[0] -20)
        cv2.putText(image_out, text, text_position, font, font_scale, font_color, font_thickness)
        text2= 'blinkCount:{:2f}'.format(blinkCount)
        text_position2 = (image_out.shape[1] - text_size[0] - 10, image_out.shape[0] - 30 )
        cv2.putText(image_out, text2, text_position2, font, font_scale, font_color, font_thickness)
        text3= 'frameCount:{:2f}'.format(frameCount)
        text_position3 = (image_out.shape[1] - text_size[0] - 10, image_out.shape[0] - 40)
        cv2.putText(image_out, text3, text_position3, font, font_scale, font_color, font_thickness)
        text4= 'blinkLonggest:{:2f}'.format(blinkLonggest)
        text_position4 = (image_out.shape[1] - text_size[0] - 10, image_out.shape[0] - 50)
        cv2.putText(image_out, text4, text_position4, font, font_scale, font_color, font_thickness)
        text5= 'alert:{}'.format(alert)
        text_position5 = (image_out.shape[1] - text_size[0] - 10, image_out.shape[0] - 60)
        cv2.putText(image_out, text5, text_position5, font, font_scale, font_color, font_thickness)
        return image_out
    
    def vector_to_screen(self,pixel_scale=np.array([0.215,0.215])):
        # gaze_vector为屏幕坐标系注视向量
        gaze_vector = warp_norm.pitchyaw_to_vector(self.pred_gaze.reshape((1,2)))
        org_pred = np.dot(np.linalg.inv(self.R), gaze_vector.T)
        org_pred = org_pred.reshape((1,3))
        # 先反转为真实方向 反向+反转x轴
        org_pred = -org_pred[0]
        org_pred[0] = -org_pred[0]

        self.org_pred = org_pred

        # 坐标系方向转换至与屏幕坐标系一致 反转x轴
        fc = np.array([-self.face_center[0,0], self.face_center[1,0], self.face_center[2,0]])
        # 新版处理流程
        scale = np.abs(fc[2]/org_pred[2])
        gaze_vector_org = org_pred * scale
        gp = fc + gaze_vector_org
        # 旧版处理流程
        # theta = np.arcsin(np.linalg.norm(np.cross(org_pred,z))/(np.linalg.norm(org_pred)*np.linalg.norm(z)))
        # scale = np.linalg.norm(z)/(np.cos(theta)*np.linalg.norm(org_pred))
        # gp = scale * org_pred - z #单位为mm
        # print(gp)
        gp = np.delete(gp, 2, axis=0)
        gp = gp/pixel_scale
        self.gaze_point = gp
        return gp