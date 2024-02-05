import pybullet as p
import pybullet_data
import numpy as np
from PIL import Image
import os
import math
import random
from pybullet_planning.interfaces.robots.collision import pairwise_collision
import argparse
from types import SimpleNamespace
from scipy.spatial.transform import Rotation as Rot
from ycb_objects.pybullet_object_models import ycb_objects
import time
from pyquaternion import Quaternion
from scipy.spatial.transform import Rotation as R

def parse_args():
    parser = argparse.ArgumentParser(description="Simple example of a data generation script for InstructPix2Pix.")
    parser.add_argument(
        "--id",
        type=str,
        default=None,
        required=True,
        help="Id of the urdf",
    )
    parser.add_argument(
        "--texture_id",
        type=str,
        default=None,
        required=False,
        help="Id of the urdf",
    )
    args = parser.parse_args()
    return args

class Image_Extractor:
    def __init__(self, urfd_path, r, h, target_h , num_frames, folder, plane, ptr, texture, ht_gnd = 0.0):
        # p.connect(p.GUI)
        p.connect(p.DIRECT)

        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        radius=10
        self.texture = texture
        dt = 1./240
        self.floor = SimpleNamespace()
        self.floor.ori = [0, 0, 0]
        self.floor.pos = [0, 0, 0]
        self.floor.id = p.loadURDF(plane, self.floor.pos, self.euler2quat(self.floor.ori))
        # self.floor.texture_id = p.loadTexture(self.texture)
        # p.changeVisualShape(self.floor.id, -1, textureUniqueId=self.floor.texture_id)

        p.configureDebugVisualizer(shadowMapWorldSize=5)
        p.configureDebugVisualizer(shadowMapResolution=8192)
        p.configureDebugVisualizer(lightPosition=[radius*math.sin(3*dt),radius*math.cos(3*dt),33])

        self.ht_gnd = 0
        # Load an R2D2 droid at the position at 0.5 meters height in the z-axis.
        self.r2d2 = p.loadURDF(urfd_path, [0, 0, self.ht_gnd])
        while pairwise_collision(self.floor.id, self.r2d2) == True:
            self.ht_gnd += 0.025
            # print(self.ht_gnd)
            p.resetBasePositionAndOrientation(self.r2d2, [0, 0, self.ht_gnd], p.getQuaternionFromEuler([0,0,0]))

        self.ht_gnd = 0.2

        p.setGravity(0, 0, -9.807)
        self.r = r
        self.h = 2*self.ht_gnd
        self.num_frames = num_frames
        self.target_h = self.ht_gnd
        self.folder = folder
        self.itr = ptr

    def euler2quat(self, angles):
        rot = Rot.from_euler('xyz', angles, degrees=True)
        rot_quat = rot.as_quat()
        return rot_quat
    
    def interpolate_points(self, point1, point2, num_points):
        t_values = np.linspace(0, 1, num_points)
        interpolated_points = [(1 - t) * point1 + t * point2 for t in t_values]
        return interpolated_points
    
    def cvK2BulletP(self, w=256, h=256, near=0.01, far=100):
        """
        cvKtoPulletP converst the K interinsic matrix as calibrated using Opencv
        and ROS to the projection matrix used in openGL and Pybullet.
        :param K:  OpenCV 3x3 camera intrinsic matrix
        :param w:  Image width
        :param h:  Image height
        :near:     The nearest objects to be included in the render
        :far:      The furthest objects to be included in the render
        :return:   4x4 projection matrix as used in openGL and pybullet
        """ 
        f_x = 126.0
        f_y = 126.
        c_x = 126.
        c_y = 126.
        A = (near + far)/(near - far)
        B = 2 * near * far / (near - far)
        projection_matrix = np.array([[2/w * f_x,  0,          (w - 2*c_x)/w,  0], 
                            [0,          2/h * f_y,  (2*c_y - h)/h,  0], 
                            [0,          0,          A,              B], 
                            [0,          0,          -1,             0]])
        #The transpose is needed for respecting the array structure of the OpenGL
        return np.array(projection_matrix).T.reshape(16).tolist()
    
    def get_pose(self, rounds):
        trajs = []
        t_heights = []
        angles = np.arange(0, 2*np.pi, 2*np.pi/self.num_frames)
        # angles = [0, np.pi/6, 2*np.pi/6, 4*np.pi/6, 5*np.pi/6, 6*np.pi/6, 7*np.pi/6, 8*np.pi/6, 10*np.pi/6, 11*np.pi/6] 
        angles = [np.pi/6] 

        for _ in range(rounds):
            for n in angles:
                t_height = []
                # target
                if n > np.pi/2 and n < 3*np.pi/2:
                    theta = np.pi
                else:
                    theta = 0
                target_pos = [(self.r)*np.cos(theta), (self.r)*np.sin(theta), self.h/2]
                # Initial
                dh = round(random.uniform(-self.h/2 + 0.1, self.h), 2)
                dr = round(random.uniform(self.r/4, 2*self.r), 2)
                dtheta = round(random.uniform(-np.pi/20, np.pi/20), 2)
                intial_pos = [(self.r + dr)*np.cos(n+dtheta), (self.r + dr)*np.sin(n+dtheta), self.h/2 + dh]
                
                interp_points = self.interpolate_points(np.array(intial_pos), np.array(target_pos), 9)
                trajs.append(interp_points)

                for _ in range(len(interp_points)):
                    # dth = round(random.uniform(self.h/2 - 0.5, self.h/2 + 0.5), 2)
                    t_height.append(self.h/2)
                t_heights.append(t_height)

        return trajs, t_heights
    
    def change_texture(self, txtre_id):
        self.floor.texture_id = p.loadTexture(self.texture.replace("00", txtre_id))
        p.changeVisualShape(self.floor.id, -1, textureUniqueId=self.floor.texture_id)

    def cvPose2BulletView(self, t=None, q=None, view=None):
        """
        cvPose2BulletView gets orientation and position as used 
        in ROS-TF and opencv and coverts it to the view matrix used 
        in openGL and pyBullet.
        
        :param q: ROS orientation expressed as quaternion [qx, qy, qz, qw] 
        :param t: ROS postion expressed as [tx, ty, tz]
        :return:  4x4 view matrix as used in pybullet and openGL
        
        """
        def quat2rot(quat):
            r = Rot.from_quat(quat)
            if hasattr(r, 'as_matrix'):
                return r.as_matrix()
            else:
                return r.as_dcm()
        def euler2rot(euler, axes='xyz'):
            r = Rot.from_euler(axes, euler)
            if hasattr(r, 'as_matrix'):
                return r.as_matrix()
            else:
                return r.as_dcm()
        def to_rot_mat(ori):
            ori = np.array(ori)
            if ori.size == 3:
                # [roll, pitch, yaw]
                ori = euler2rot(ori)
            elif ori.size == 4:
                ori = quat2rot(ori)
            elif ori.shape != (3, 3):
                raise ValueError('Orientation should be rotation matrix, '
                                'euler angles or quaternion')
            return ori
        def create_se3(ori, trans=None):
            rot = to_rot_mat(ori)
            out = np.eye(4)
            out[:3, :3] = rot
            if trans is not None:
                trans = np.array(trans)
                out[:3, 3] = trans.flatten()
            return out
        def rotvec2rot(vec):
            r = Rot.from_rotvec(vec)
            if hasattr(r, 'as_matrix'):
                return r.as_matrix()
            else:
                return r.as_dcm()
        
        # R = q.rotation_matrix
        # T = np.vstack([np.hstack([R, np.array(t).reshape(3,1)]),
        #                         np.array([0, 0, 0, 1])])
        # Tc = np.array([[1,   0,    0,  0],
        #             [0,   1,    0,  0],
        #             [0,   0,    1,  0],
        #             [0,   0,    0,  1]]).reshape(4,4)
        # T=Tc@np.linalg.inv(T)
        # viewMatrix = T.T.reshape(16)

        view_matrix = np.array(view).reshape(4, 4)
        rot = create_se3(rotvec2rot(np.pi * np.array([1, 0, 0])))
        view_matrix_T = view_matrix.T
        cam_ext_mat = np.dot(np.linalg.inv(view_matrix_T), rot)
        print("Cam ext mat: ", cam_ext_mat)
        r = R.from_matrix(cam_ext_mat[:3, :3])
        print("r: ", r.as_euler("xyz", degrees=True), "\n")
        
        return None
    
    def save_images(self, trajs, t_heights):
        itr = self.itr
        itr_global = -1
        for eye_pos_edited in trajs:
            itr_global += 1
            itr_h = -1
            for pos in eye_pos_edited:
                itr_h+=1
                # break
                p.configureDebugVisualizer(lightPosition=[pos[0], pos[1], 3])
                width = 256
                height = 256
                t = pos
                # q1 = Quaternion(axis=[0, 1, 0], angle=np.pi/2)
                # q2 = Quaternion(axis=[0, 0, 1], angle=np.pi/2)
                # q3 = Quaternion(axis=[1, 0, 0], angle=-np.pi/24)
                # q = q1*q2*q3
                # print("t = ", t, "q = ", q)
                print("Pos: ", pos)
                view_mat = p.computeViewMatrix(cameraEyePosition=[pos[0], pos[1], pos[2]],
                                               cameraTargetPosition=[0,0,t_heights[itr_global][itr_h]],
                                               cameraUpVector=[0,0,1],)
                self.cvPose2BulletView(view=view_mat)
                img_arr = p.getCameraImage(
                    width,
                    height,
                    viewMatrix=view_mat,
                    projectionMatrix=p.computeProjectionMatrixFOV(
                        fov=60,
                        aspect=width/height,
                        nearVal=0.01,
                        farVal=10,
                    ),
                    renderer=p.ER_BULLET_HARDWARE_OPENGL,
                    shadow = False,
                )

                width, height, rgba, depth, mask = img_arr
                rgba = bytes(rgba)
                im_rgb = Image.frombytes('RGBA', (width, height), rgba)

                fol = "trajs/"
                if not os.path.isdir(fol):
                    os.makedirs(fol)
                im_rgb = im_rgb.save(fol + str(itr_global).zfill(4) + "_" + str(itr_h).zfill(4) + ".png")
                itr+=1
                time.sleep(1)

    def capture_images(self):
        rounds = 1
        self.itr = 0

        trajs, t_heights = self.get_pose(rounds)
        self.save_images(trajs, t_heights)

if __name__ == "__main__":
    plane = "plane.urdf"
    num_frames = 10

    save_folder =   "./imgs/"

    if not os.path.exists(save_folder):
        os.makedirs(save_folder)

    mul = -1
    mul+=1

    # Camera params 
    # r = 1.5
    r = 0.45
    h = -0.1

    target_h = h - 0.3
    ptr = 0 + num_frames*9*mul
    path = os.path.join(ycb_objects.getDataPath(), "YcbChipsCan", "model.urdf")
    # path = "./door/8897/mobility.urdf"
    texture = "./textures/floor/00.jpg"
    image_extarctor = Image_Extractor(path, r, h, target_h, num_frames, save_folder, plane, ptr, texture)
    image_extarctor.capture_images()
    p.resetSimulation()