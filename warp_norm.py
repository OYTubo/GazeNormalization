import sys
sys.path.append("./face-alignment")
import face_alignment
from imutils import face_utils
import cv2
import dlib
import numpy as np


def pitchyaw_to_vector(pitchyaws):
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


def vector_to_pitchyaw(vectors):
    r"""Convert given gaze vectors to yaw (:math:`\theta`) and pitch (:math:`\phi`) angles.

    Args:
        vectors (:obj:`numpy.array`): gaze vectors in 3D :math:`(n\times 3)`.

    Returns:
        :obj:`numpy.array` of shape :math:`(n\times 2)` with values in radians.
    """
    n = vectors.shape[0]
    out = np.empty((n, 2))
    vectors = np.divide(vectors, np.linalg.norm(vectors, axis=1).reshape(n, 1))
    out[:, 0] = np.arcsin(vectors[:, 1])  # theta
    out[:, 1] = np.arctan2(vectors[:, 0], vectors[:, 2])  # phi
    return out


def estimateHeadPose(landmarks, face_model, camera, distortion = np.array([-0.16321888, 0.66783406, -0.00121854, -0.00303158, -1.02159927]), iterate=True):
    ret, rvec, tvec = cv2.solvePnP(face_model, landmarks, camera, distortion, flags=cv2.SOLVEPNP_EPNP)

    ## further optimize
    if iterate:
        ret, rvec, tvec = cv2.solvePnP(face_model, landmarks, camera, distortion, rvec, tvec, True)

    return rvec, tvec

def xnorm(input, camera_matrix, camera_distortion = np.array([-0.16321888, 0.66783406, -0.00121854, -0.00303158, -1.02159927])):
    # face detection
    predictor = dlib.shape_predictor('./shape_predictor_68_face_landmarks.dat')
    # face_detector = dlib.cnn_face_detection_model_v1('./modules/mmod_human_face_detector.dat')
    face_detector = dlib.get_frontal_face_detector()  ## this face detector is not very powerful
    detected_faces = face_detector(cv2.cvtColor(input, cv2.COLOR_BGR2RGB), 1) ## convert BGR image to RGB for dlib
    if len(detected_faces) == 0:
        print('warning: no detected face')
        exit(0)
    print('detected one face')
    shape = predictor(input, detected_faces[0]) ## only use the first detected face (assume that each input image only contains one face)
    shape = face_utils.shape_to_np(shape)
    landmarks = []
    for (x, y) in shape:
        landmarks.append((x, y))
    landmarks = np.asarray(landmarks)
    
    # load face model
    face_model_load = np.loadtxt('face_model.txt')  # Generic face model with 3D facial landmarks
    landmark_use = [20, 23, 26, 29, 15, 19]  # we use eye corners and nose conners
    face_model = face_model_load[landmark_use, :]
    # estimate the head pose,
    ## the complex way to get head pose information, eos library is required,  probably more accurrated
    # landmarks = landmarks.reshape(-1, 2)
    # head_pose_estimator = HeadPoseEstimator()
    # hr, ht, o_l, o_r, _ = head_pose_estimator(image, landmarks, camera_matrix[cam_id])
    ## the easy way to get head pose information, fast and simple
    facePts = face_model.reshape(6, 1, 3)
    landmarks_sub = landmarks[[36, 39, 42, 45, 31, 35], :]
    landmarks_sub = landmarks_sub.astype(float)  # input to solvePnP function must be float type
    landmarks_sub = landmarks_sub.reshape(6, 1, 2)  # input to solvePnP requires such shape
    hr, ht = estimateHeadPose(landmarks_sub, facePts, camera_matrix, camera_distortion)
    return hr,ht


def enorm(input, camera_matrix, camera_distortion = np.array([-0.16321888, 0.66783406, -0.00121854, -0.00303158, -1.02159927])):
    '''
    input: 待处理的图片，可以不经过人脸提取
    camera_matrix: 相机内参矩阵
    camera_distortion: 相机畸变矩阵，这里的默认值是xGaze的相机参数
    '''
    # 标定人脸关键点
    fa = face_alignment.FaceAlignment(face_alignment.LandmarksType.TWO_D, flip_input=False)
    # input = cv2.imread(img_path)
    preds = fa.get_landmarks(input)
    # 选取指定的6个点(左右眼角，嘴角)
    landmark_use = [36,39,42,45,48,54]
    lm = preds[0]
    lm = lm[landmark_use, :]
    # load the generic face model, which includes 6 facial landmarks: four eye corners and two mouth corners
    face = np.loadtxt('./data-preprocessing-gaze/data/faceModelGeneric.txt')
    num_pts = face.shape[1]
    facePts = face.T.reshape(num_pts, 3)
    # fid = cv2.FileStorage('./data-preprocessing-gaze//data/calibration/cameraCalib.xml', cv2.FileStorage_READ)
    # camera_matrix = fid.getNode("camera_matrix").mat()
    # camera_distortion = fid.getNode("cam_distortion").mat()
    lm = lm.astype(np.float32)
    lm = lm.reshape(num_pts, 1, 2)
    hr, ht = estimateHeadPose(lm, facePts, camera_matrix, camera_distortion)    
    return hr, ht

    # normalization function for the face images
def xtrans(img, face_model, hr, ht, cam, gc = np.array([100,100]), pixel_scale = np.array([0.28802082, 0.28796297])):
    '''
    img: 人脸图片
    face_model: 人脸模型,[68,2]
    hr: 来自annotation, 旋转向量, [3,1]
    ht: 来自annotation, 平移向量, [3,1]
    cam: Camera_Matrix
    gc: 来自annotation, gaze point on the screen coordinate system, [2,1]
    '''
    # normalized camera parameters
    focal_norm = 960  # focal length of normalized camera
    distance_norm = 600  # normalized distance between face and camera
    roiSize = (224, 224)  # size of cropped image

    # compute estimated 3D positions of the landmarks
    ht = ht.reshape((3, 1))
    if(gc.size == 2):
        # should change depend on the camera position
        gc = np.array([- gc[0] + 1920/2, gc[1] - 1080])
        gc = gc * pixel_scale
        gc = np.r_[gc, np.array([0])]
        gc = gc.reshape((3, 1))
    # 将二维人脸（原图）套入三维模型
    hR = cv2.Rodrigues(hr)[0]  # rotation matrix, [3,3]
    Fc = np.dot(hR, face_model.T) + ht # [3,6]
    # 取得关键点
    two_eye_center = np.mean(Fc[:, 0:4], axis=1).reshape((3, 1))
    mouth_center = np.mean(Fc[:, 4:6], axis=1).reshape((3, 1))
    face_center = np.mean(np.concatenate((two_eye_center, mouth_center), axis=1), axis=1).reshape((3, 1))

    # normalize image
    distance = np.linalg.norm(face_center)  # actual distance between and original camera
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
    forward = (face_center / distance).reshape(3)
    down = np.cross(forward, hRx)
    down /= np.linalg.norm(down)
    right = np.cross(down, forward)
    right /= np.linalg.norm(right)
    R = np.c_[right, down, forward].T  # rotation matrix R

    W = np.dot(np.dot(cam_norm, S), np.dot(R, np.linalg.inv(cam)))  # transformation matrix

    img_warped = cv2.warpPerspective(img, W, roiSize)  # image normalization

    # normalize rotation 
    hR_norm = np.dot(R, hR)  # rotation matrix in normalized space
    hr_norm = cv2.Rodrigues(hR_norm)[0]  # convert rotation matrix to rotation vectors

    # normalize gaze vector
    gc_normalized = gc - face_center  # gaze vector
    gc_normalized = np.dot(R, gc_normalized) # 这里只追求旋转，所以没有与相机矩阵相乘
    gc_normalized = gc_normalized / np.linalg.norm(gc_normalized) #归一化

    return img_warped, hr_norm, gc_normalized, R


def draw_gaze(image_in, gc_normalized, thickness=2, color=(0, 0, 255)):
    '''Draw gaze angle on given image with a given eye positions.'''
    gaze_theta = np.arcsin((-1) * gc_normalized[1])
    gaze_phi = np.arctan2((-1) * gc_normalized[0], (-1) * gc_normalized[2])
    pitchyaw = np.array([gaze_theta[0], gaze_phi[0]])
    image_out = image_in
    (h, w) = image_in.shape[:2]
    length = np.min([h, w]) / 2.0
    pos = (int(w / 2.0), int(h / 2.0))
    if len(image_out.shape) == 2 or image_out.shape[2] == 1:  # to draw on the image, we need to convert to RGB
        image_out = cv2.cvtColor(image_out, cv2.COLOR_GRAY2BGR)
    dx = -length * np.sin(pitchyaw[1]) * np.cos(pitchyaw[0])
    dy = -length * np.sin(pitchyaw[0])
    cv2.arrowedLine(image_out, tuple(np.round(pos).astype(int)),
                   tuple(np.round([pos[0] + dx, pos[1] + dy]).astype(int)), color,
                   thickness, cv2.LINE_AA, tipLength=0.2)

    return image_out


def draw_gaze_pitchyaw(image_in, pitchyaw, thickness=2, color=(0, 0, 255)):
    '''Draw gaze angle on given image with a given eye positions.'''
    image_out = image_in
    (h, w) = image_in.shape[:2]
    length = np.min([h, w]) / 2.0
    pos = (int(w / 2.0), int(h / 2.0))
    if len(image_out.shape) == 2 or image_out.shape[2] == 1:  # to draw on the image, we need to convert to RGB
        image_out = cv2.cvtColor(image_out, cv2.COLOR_GRAY2BGR)
    dx = -length * np.sin(pitchyaw[1]) * np.cos(pitchyaw[0])
    dy = -length * np.sin(pitchyaw[0])
    cv2.arrowedLine(image_out, tuple(np.round(pos).astype(int)),
                   tuple(np.round([pos[0] + dx, pos[1] + dy]).astype(int)), color,
                   thickness, cv2.LINE_AA, tipLength=0.2)

    return image_out

def GazeNormalization(image, camera_matrix, camera_distortion, gc, method='xgaze'):
    if(method == 'xgaze'):
        hr, ht = xnorm(image, camera_matrix, camera_distortion)
        face_model_load = np.loadtxt('face_model.txt')  # Generic face model with 3D facial landmarks
        landmark_use = [20, 23, 26, 29, 15, 19]  # we use eye corners and nose conners
        face_model = face_model_load[landmark_use, :]
        warp_image,_,gcn,_ = xtrans(image, face_model, hr, ht, camera_matrix, gc)
    else:   
        hr, ht = enorm(image, camera_matrix, camera_distortion)
        face = np.loadtxt('./faceModelGeneric.txt')
        num_pts = face.shape[1]
        face_model = face.T.reshape(num_pts, 3)
        warp_image,_,gcn,_ = xtrans(image, face_model, hr, ht, camera_matrix, gc)
    return warp_image, gcn