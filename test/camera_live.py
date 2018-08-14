#coding:utf-8
import sys
sys.path.append('..')
sys.path.insert(0, '/usr/lib/python2.7/dist-packages')
from Detection.MtcnnDetector import MtcnnDetector
from Detection.detector import Detector
from Detection.fcn_detector import FcnDetector
from train_models.mtcnn_model import P_Net, R_Net, O_Net
import cv2
from scipy.misc import imresize
import numpy as np
import math

def _rotation_matrix(rad_x, rad_y, rad_z):
    cosx, cosy, cosz = math.cos(rad_x), math.cos(rad_y), math.cos(rad_z)
    sinx, siny, sinz = math.sin(rad_x), math.sin(rad_y), math.sin(rad_z)
    rotz = np.array([[cosz, -sinz, 0],
                     [sinz, cosz, 0],
                     [0, 0, 1]], dtype=np.float32)
    roty = np.array([[cosy, 0, siny],
                     [0, 1, 0],
                     [-siny, 0, cosy]], dtype=np.float32)
    rotx = np.array([[1, 0, 0],
                     [0, cosx, -sinx],
                     [0, sinx, cosx]], dtype=np.float32)
    return rotx.dot(roty).dot(rotz)

def _project_plane_yz(vec):
    x = vec.dot(np.array([0, 1, 0], dtype=np.float32))
    y = vec.dot(np.array([0, 0, 1], dtype=np.float32))
    return np.array([x, -y], dtype=np.float32)  # y flip

def _draw_line(img, pt1, pt2, color, thickness=2):
    pt1 = (int(pt1[0]), int(pt1[1]))
    pt2 = (int(pt2[0]), int(pt2[1]))
    cv2.line(img, pt1, pt2, color, int(thickness))

def draw_pose(img, pose, size=30, idx=0):
    # parallel projection (something wrong?)
    rotmat = _rotation_matrix(-pose[0], -pose[1], -pose[2])
    # print(rotmat)
    zvec = np.array([0, 0, 1], np.float32)
    yvec = np.array([0, 1, 0], np.float32)
    xvec = np.array([1, 0, 0], np.float32)
    zvec = _project_plane_yz(rotmat.dot(zvec))
    yvec = _project_plane_yz(rotmat.dot(yvec))
    xvec = _project_plane_yz(rotmat.dot(xvec))

    # Lower left
    org_pt = ((size + 5) * (2 * idx + 1), img.shape[0] - size - 5)
    # _draw_line(img, org_pt, org_pt + zvec * size, (255, 0, 0), 3)
    # _draw_line(img, org_pt, org_pt + yvec * size, (0, 255, 0), 3)
    # _draw_line(img, org_pt, org_pt + xvec * size, (0, 0, 255), 3)
    _draw_line(img, org_pt, org_pt + zvec * [-1,1] * size, (255, 0, 0), 3) #blue
    _draw_line(img, org_pt, org_pt + yvec * [1,-1] * size, (0, 255, 0), 3) #green
    _draw_line(img, org_pt, org_pt + xvec * [1,1] * size, (0, 0, 255), 3) #red

    return img

def rotationMatrixToEulerAngles(R) :
  
        #assert(isRotationMatrix(R))
      
        #To prevent the Gimbal Lock it is possible to use
        #a threshold of 1e-6 for discrimination
        sy = math.sqrt(R[0,0] * R[0,0] +  R[1,0] * R[1,0])    
        singular = sy < 1e-6
  
        if  not singular :
            x = math.atan2(R[2,1] , R[2,2])
            y = math.atan2(-R[2,0], sy)
            z = math.atan2(R[1,0], R[0,0])
        else :
            x = math.atan2(-R[1,2], R[1,1])
            y = math.atan2(-R[2,0], sy)
            z = 0
  
        return np.array([x, y, z])

def face_orientation(frame, landmarks):
    size = frame.shape #(height, width, color_channel)
    # eyes_cy = int((landmarks[5] + landmarks[6])/2)
    # mouth_cy = int((landmarks[8] + landmarks[9])/2)
    # mouth2eyes = int(mouth_cy - eyes_cy)
    # Chin_x = int((landmarks[3] + landmarks[4])/2)
    # Chin_y = int(mouth_cy + mouth2eyes*9/16)

    image_points = np.array([
                            (landmarks[4], landmarks[5]),     # Nose tip
                            (landmarks[10], landmarks[11]),   # Chin
                            (landmarks[0], landmarks[1]),     # Left eye left corner
                            (landmarks[2], landmarks[3]),     # Right eye right corne
                            (landmarks[6], landmarks[7]),     # Left Mouth corner
                            (landmarks[8], landmarks[9])      # Right mouth corner
                        ], dtype="double")

    # 3D model points.
    model_points = np.array([
                            (0.0, 0.0, 0.0),             # Nose tip
                            (0.0, -330.0, -65.0),        # Chin
                            (-165.0, 170.0, -135.0),     # Left eye left corner
                            (165.0, 170.0, -135.0),      # Right eye right corne
                            (-150.0, -150.0, -125.0),    # Left Mouth corner
                            (150.0, -150.0, -125.0)      # Right mouth corner                         
                        ])

    # Camera internals
 
    focal_length = (size[1], size[0])
    center = (size[1]/2, size[0]/2)
    camera_matrix = np.array(
                         [[focal_length[0], 0, center[0]],
                         [0, focal_length[0], center[1]],
                         [0, 0, 1]], dtype = "double"
                         )

    dist_coeffs = np.zeros((4,1)) # Assuming no lens distortion
    (success, rotation_vector, translation_vector) = cv2.solvePnP(model_points, image_points, camera_matrix, dist_coeffs, flags=cv2.CV_ITERATIVE)
    (nose_end_point2D, jacobian) = cv2.projectPoints(np.array([(0.0, 0.0, 1000.0)]), rotation_vector, translation_vector, camera_matrix, dist_coeffs)
    axis = np.float32([[300,0,0], 
                          [0,300,0], 
                          [0,0,300]])
    (imgpts, jacobian2) = cv2.projectPoints(axis, rotation_vector, translation_vector, camera_matrix, dist_coeffs)
    
    # pitch, roll, yaw = rotation_vector
    # pitch_degrees = math.degrees(math.asin(math.sin(pitch)))
    # roll_degrees = math.degrees(math.asin(math.sin(roll)))
    # yaw_degrees = math.degrees(math.asin(math.sin(yaw)))


    # No. 1 
    rvec_matrix = cv2.Rodrigues(rotation_vector)[0]
    proj_matrix = np.hstack((rvec_matrix, translation_vector))
    euler_angle = cv2.decomposeProjectionMatrix(proj_matrix, camera_matrix, rotation_vector, translation_vector)[6] 

    pitch, yaw, roll = [math.radians(_) for _ in euler_angle]

    pitch = math.degrees(math.asin(math.sin(pitch)))
    roll = math.degrees(math.asin(math.sin(roll)))
    yaw = math.degrees(math.asin(math.sin(yaw)))


    # pitch_degrees = euler_angle[0]
    # yaw_degrees = euler_angle[1]
    # roll_degrees = euler_angle[2]

    # No. 2
    # pitch_degrees = math.degrees(rotationMatrixToEulerAngles(rvec_matrix)[0]) #ok top: +180 +179 ... down: -180 -179 -178
    # # print(pitch_degrees)
    # yaw_degrees = math.degrees(rotationMatrixToEulerAngles(rvec_matrix)[1]) #ok left: +0 +1 ... right: -1 -2 -3
    # # print(yaw_degrees)
    # roll_degrees = math.degrees(rotationMatrixToEulerAngles(rvec_matrix)[2]) #ok left:-1 -2 ... right:+1 +2 ...
    # if pitch_degrees > 0:
    #     pitch_degrees = 180 - pitch_degrees
    # elif pitch_degrees < 0:
    #     pitch_degrees = -180 - pitch_degrees
    # roll_degrees = -roll_degrees 


    # print(roll_degrees, pitch_degrees, yaw_degrees)
    # print()
    # print(pitch_degrees, yaw_degrees, roll_degrees) 

    # p1 = ( int(image_points[0][0]), int(image_points[0][1]))
    # p2 = ( int(nose_end_point2D[0][0][0]), int(nose_end_point2D[0][0][1]))


    return imgpts, (roll, pitch, yaw)
    # return p1, p2, (math.radians(roll_degrees), math.radians(pitch_degrees), math.radians(yaw_degrees))

test_mode = "onet"
thresh = [0.6, 0.6, 0.7]
# thresh = [0.6, 0.15, 0.05]
min_face_size = 100
stride = 2
slide_window = False
shuffle = False
#vis = True
detectors = [None, None, None]
prefix = ['../data/MTCNN_model/20180213(60)/PNet_landmark/PNet', '../data/MTCNN_model/20180213(60)/RNet_landmark/RNet', '../data/MTCNN_model/20180213(60)/ONet_landmark/ONet']
epoch = [30, 22, 22]
model_path = ['%s-%s' % (x, y) for x, y in zip(prefix, epoch)]
PNet = FcnDetector(P_Net, model_path[0])
detectors[0] = PNet
RNet = Detector(R_Net, 24, 1, model_path[1])
detectors[1] = RNet
ONet = Detector(O_Net, 48, 1, model_path[2])
detectors[2] = ONet
videopath = "./video_test.avi"
mtcnn_detector = MtcnnDetector(detectors=detectors, min_face_size=min_face_size,
                               stride=stride, threshold=thresh, slide_window=slide_window)

video_capture = cv2.VideoCapture(0)
qq
# Define the codec and create VideoWriter object
fourcc = cv2.cv.CV_FOURCC(*'XVID')
out = cv2.VideoWriter('output2.avi',fourcc, 20.0, (640,480))

corpbbox = None
while True:
    # fps = video_capture.get(cv2.CAP_PROP_FPS)
    t1 = cv2.getTickCount()
    ret, frame = video_capture.read()
    frame = cv2.flip(frame, 1)
    if ret:
        image = np.array(frame)
        boxes_c,landmarks = mtcnn_detector.detect(image)
        
        t2 = cv2.getTickCount()
        t = (t2 - t1) / cv2.getTickFrequency()
        fps = 1.0 / t
        for i in range(boxes_c.shape[0]):
            bbox = boxes_c[i, :4]
            score = boxes_c[i, 4]
            corpbbox = [int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])]
            # if score > thresh:
            cv2.rectangle(frame, (corpbbox[0], corpbbox[1]),
                          (corpbbox[2], corpbbox[3]), (0, 255, 0), 2)
            # cv2.putText(frame, '{:.3f}'.format(score), (corpbbox[0], corpbbox[1] - 2), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
        # cv2.putText(frame, '{:.4f}'.format(t) + " " + '{:.3f}'.format(fps), (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 255), 2)
        for i in range(landmarks.shape[0]):
            for j in range(len(landmarks[i])/2):
                cv2.circle(frame, (int(landmarks[i][2*j]),int(int(landmarks[i][2*j+1]))), 3, (0,0,255),-1)            
        # time end

            # p1, p2, rotation_vector = face_orientation(frame, landmarks[i])
            imgpts, rotate_degree = face_orientation(frame, landmarks[i])

    	    # cv2.putText(frame, str(rotate_degree[0])+' '+str(rotate_degree[1])+' '+str(rotate_degree[2]), (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), thickness=2, lineType=2)
            # cv2.putText(frame, '{:05.2f}'.format(rotate_degree), (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), thickness=2, lineType=2)
            for j in xrange(len(rotate_degree)):
                cv2.putText(frame, ('{:05.2f}').format(float(rotate_degree[j])), (10, 30 + (50 * j)), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), thickness=2, lineType=2)
            

            cv2.line(frame, (landmarks[i][4],landmarks[i][5]), tuple(imgpts[1].ravel()), (0,255,0), 3) #GREEN
            cv2.line(frame, (landmarks[i][4],landmarks[i][5]), tuple(imgpts[0].ravel()), (255,0,0), 3) #BLUE
            cv2.line(frame, (landmarks[i][4],landmarks[i][5]), tuple((imgpts[2]).ravel()), (0,0,255), 3) #RED



            # print(rotation_vector)
            # cv2.line(frame, p1, p2, (255,0,0), 2)
            # frame = draw_pose(frame, rotation_vector)

        # out.write(frame)
        cv2.imshow("", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    else:
        print 'device not find'
        break
video_capture.release()
cv2.destroyAllWindows()
