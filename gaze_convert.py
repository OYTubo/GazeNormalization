import numpy as np
import scipy.io as sio
import sys
import cv2

def Gaze2DTo3D(point, origin, rmat, tmat):
  """
    Usage:
      This function convert 2D gaze to 3D gaze.
    Algorithm:
      First compute the 3D coordinate T of 2D point in CCS.
      The 3D gaze g = T - origin.
      Note the 3D gaze is defined in CCS.
    Inputs:
      point: numpy array, [x, y]. Defined in SCS.
      origin: numpy array, [x, y, z]. Defined in CCS.
      rmat: numpy array, 3*3. Convert points in SCS to points CCS.
      tmat: numpy array, shape is 3*1.
    Return:
      a numpy array, the shape is (3,).
  """

  assert type(point) == type(np.zeros(0)) and point.shape == (2, ), "There is an error about point."
  assert type(origin) == type(np.zeros(0)) and origin.shape == (3, ), "There is an error about origin."
  assert type(rmat) == type(np.zeros(0)) and rmat.shape == (3, 3), "There is an error about rmat."
  assert type(tmat) == type(np.zeros(0)) and tmat.size == 3, "There is an error about tmat."

  target = __Point2DTo3D(point, rmat, tmat)
  gaze = target - origin  
  return gaze

def Gaze3DTo2D(gaze, origin, rmat, tmat, require3d=False):
  """
    Usage:
      Convert 3D gaze direction to 2D gaze point in screen.
    Algorithm:
      We fisrt compute the equation of screen plane and then compute the intersection of the plane and gaze.
    Input:
      gaze: numpy array, the shape is (3,).
      origin: numpy array, the shape is (3,). It is the origin of gaze direction.
      rmat: numpy array, the shape is (3, 3). Convert SCS to CCS.
      tmat: numpy array, the shape is (3,).
  """
  assert type(gaze) == type(np.zeros(0)) and gaze.shape == (3,), "There is an error about gaze."
  assert type(origin) == type(np.zeros(0)) and origin.shape == (3,), "There is an error about origin."
  assert type(rmat) == type(np.zeros(0)) and rmat.shape == (3, 3), "There is an error about rmat."
  assert type(tmat) == type(np.zeros(0)) and tmat.size==3, "There is an error about tmat."
  
  tmat = np.reshape(tmat, (3,1))

  plane = __PlaneEquation(rmat, tmat)
  plane_w = plane[0:3]
  plane_b = plane[3]
  
  a11 = gaze[1]
  a12 = -gaze[0]
  a13 = 0
  b1 = gaze[1] * origin[0] - gaze[0] * origin[1]

  a21 = 0
  a22 = gaze[2]
  a23 = -gaze[1]
  b2 = gaze[2] * origin[1] - gaze[1] * origin[2]

  line_w = np.array([[a11, a12, a13], [a21, a22, a23]])
  line_b = np.array([[b1], [b2]])


  matrix = np.insert(line_w, 2, plane_w, axis=0)
  bias = np.insert(line_b, 2, plane_b, axis=0)

  point = np.linalg.solve(matrix, bias)
  point = np.reshape(point, (3,1))
  
  if not require3d:
      result = np.dot(np.linalg.inv(rmat), point-tmat)
      return np.array([result[0], result[1]])
  else:
      return point

 
def __Point2DTo3D(point, rmat, tmat):
  """
  Usage:
    This function convert a 2D co. in plane to a 3D co. in space.
  Algorithm:
    Get the conversion matrix R, which convert SCS to CCS.
    The 3D co. can be computed by Q = R*[x, y, 0].T
  Inputs:
    point: numpy array, [x, y].
    rmat:  numpy array, one rotation matrix, the shape is 3*3
    tmat:  numpy array, one translation matrix, the shape is 3*1
  Return:
    a numpy array, the shape is [3, 1].
  """
  
  assert type(point) == type(np.zeros(0)) and point.shape == (2, ), "There is an error about point."
  assert type(rmat) == type(np.zeros(0)) and rmat.shape == (3,3), "There is an error about rmat."
  assert type(tmat) == type(np.zeros(0)) and tmat.size == 3, "There is an error about tmat."

  point3D = np.insert(point, 2, 0)
  point3D = np.reshape(point3D, (3,1))
  tvec = np.reshape(tmat, (3,1))
  Q = np.dot(rmat, point3D) + tvec
  return Q.T

def __PlaneEquation(rmat, tmat):
  """
    Usage: 
      Given rotation matrix, this function compute the equation of x-y plane.
    Algorithm:
      The normal vector of the plane is z-axis in rotation matrix. And tmat provide on point in the plane. It is easy to infer the equation.
    Input:
      matrix: numpy array, the shape is (3,3)
    Return: (a, b, c, d), where the equation of plane is ax + by + cz = d
  """

  assert type(rmat) == type(np.zeros(0)) and rmat.shape == (3, 3), "There is an error about rmat."
  assert type(tmat) == type(np.zeros(0)) and tmat.size == 3, "There is an error about tmat."

  n = rmat[:,2]
  origin = np.reshape(tmat, (3))

  a = n[0] 
  b = n[1] 
  c = n[2] 

  d = origin[0] * n[0] + \
      origin[1] * n[1] + \
      origin[2] * n[2] 
  return np.array([a, b, c, d])
   

def __FuncTest__():
  # MPIIGaze Dataset
  # sys.argv[1] monitorPose.mat
  # sys.argv[2] screenSize.mat
  # sys.argv[3] annotation.txt

  monitorpose = sio.loadmat(sys.argv[1])
  rvec = monitorpose["rvects"]
  tvec = monitorpose["tvecs"]
  rmat = cv2.Rodrigues(rvec)[0]

  screen = sio.loadmat(sys.argv[2])
  w_pixel = screen["width_pixel"][0][0]
  h_pixel = screen["height_pixel"][0][0]
  w_mm = screen["width_mm"][0][0]
  h_mm = screen["height_mm"][0][0]
  w_ratio = w_mm / w_pixel
  h_ratio = h_mm / h_pixel


  print("Point PrePoin Direction PreDirection")
  with open(sys.argv[3]) as infile:
    lines = infile.readlines()
    for line in lines:
      line = line.strip().split(" ")
      point = np.array(line[24:26]).astype("float")
      point = np.array([point[0] * w_ratio, point[1] * h_ratio])

      direction = np.array(line[26: 29]).astype("float") - np.array(line[35: 38]).astype("float")
      #direction = direction / np.linalg.norm(direction)
      origin = np.array(line[35: 38]).astype("float")
      target = np.array(line[26: 29]).astype("float")
      target = np.append(target, -1)
      

      pre_direc = Gaze2DTo3D(point, origin, rmat, tvec)
      pre_point = Gaze3DTo2D(direction, origin, rmat, tvec)
      print(f"{point}\t{pre_point.flatten()}\t{direction}\t{pre_direc.flatten()}")
    


if __name__ == "__main__":
  __FuncTest__()
  







