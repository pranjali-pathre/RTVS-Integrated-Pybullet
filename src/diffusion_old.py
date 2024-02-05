import os
import shutil
from types import SimpleNamespace
import pybullet_data
import time
from utils.photo_error import mse_
from PIL import Image
from scipy.spatial.transform import Rotation
import math
import numpy as np
import pybullet as p
from pybullet_planning.interfaces.robots.collision import pairwise_collision
import matplotlib.pyplot as plt 
from utils.logger import logger
from utils.sim_utils import get_random_config
from ycb_objects.pybullet_object_models import ycb_objects
from pyquaternion import Quaternion
from airobot.sensor.camera.rgbdcam_pybullet import RGBDCameraPybullet

np.set_string_function(
    lambda x: repr(np.round(x, 4))
    .replace("(", "")
    .replace(")", "")
    .replace("array", "")
    .replace("       ", " "),
    repr=False,
)

class Diffusion:
    def __init__(
        self,
        urfd_path,
        cam_eye_pos,
        des_img,
        start_itr=0,
        folder="./result/",
        controller_type="diffusion",
    ):
        self.radius = 10.0
        self.dt = 1./240
        
        p.connect(p.DIRECT)
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        # p.configureDebugVisualizer(shadowMapWorldSize=5)
        # p.configureDebugVisualizer(shadowMapResolution=8192)
        p.configureDebugVisualizer(lightPosition=[self.radius*math.sin(3*self.dt), self.radius*math.cos(3*self.dt), 33])
        p.setGravity(0, 0, -9.807)
        p.configureDebugVisualizer(p.COV_ENABLE_RENDERING, 0)
        p.configureDebugVisualizer(p.COV_ENABLE_SHADOWS,0) 

        self.floor = SimpleNamespace()
        self.floor.ori = [0, 0, 0]
        self.floor.pos = [0, 0, 0]
        self.floor.id = p.loadURDF('plane.urdf', self.floor.pos, self.euler2quat(self.floor.ori))
        # self.floor.texture_id = p.loadTexture(self.texture)
        # p.changeVisualShape(self.floor.id, -1, textureUniqueId=self.floor.texture_id)

        self.ht_gnd = 0
        self.r2d2 = p.loadURDF(urfd_path, [0, 0, self.ht_gnd])
        while pairwise_collision(self.floor.id, self.r2d2) == True:
            self.ht_gnd += 0.025
            p.resetBasePositionAndOrientation(self.r2d2, [0, 0, self.ht_gnd], p.getQuaternionFromEuler([0,0,0]))
        
        self.controller_type = controller_type
        
        self.h = 2*self.ht_gnd
        self.target_h = self.ht_gnd
        self.folder = folder
        self.itr = start_itr
        self.img_goal_path = des_img
        self.cam_eye_pos = cam_eye_pos
        
        self.width = 256
        self.height = 256
        self.cam = RGBDCameraPybullet
        self.cam.setup_camera(height=256 , width=256)
        self.cam_ori = np.deg2rad([-105, 0, 0])
        self.cam_to_gt_R = R.from_euler("xyz", self.cam_ori)

        # q1 = Quaternion(axis=[0, 1, 0], angle=np.pi/2)
        # q2 = Quaternion(axis=[0, 0, 1], angle=np.pi/2)
        # self.q = q1*q2
        # self.cameraTargetPosition = [0, 0, self.target_h]
        # self.projectionmatrix = self.cvK2BulletP()

        if not os.path.isdir(self.folder):
            os.mkdir(self.folder)
        else:
            shutil.rmtree(self.folder, ignore_errors=True)
            os.makedirs(self.folder, exist_ok=True)

        if not os.path.isdir("./imgs"):
            os.mkdir("./imgs")
        else:
            shutil.rmtree("./imgs", ignore_errors=True)
            os.makedirs("./imgs", exist_ok=True)
            
        self._set_controller()

    def euler2quat(self, angles):
        rot = Rotation.from_euler('xyz', angles, degrees=True)
        rot_quat = rot.as_quat()
        return rot_quat
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
        f_x = 126.
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
    
    def cvPose2BulletView(self):
        """
        cvPose2BulletView gets orientation and position as used 
        in ROS-TF and opencv and coverts it to the view matrix used 
        in openGL and pyBullet.
        
        :param q: ROS orientation expressed as quaternion [qx, qy, qz, qw] 
        :param t: ROS postion expressed as [tx, ty, tz]
        :return:  4x4 view matrix as used in pybullet and openGL
        
        """
        # t = self.cam_eye_pos
        # q = Quaternion(axis=self.axis, angle=self.theta)
        R = self.q.rotation_matrix
        # print("R = ", R)
        T = np.vstack([np.hstack([R, np.array(self.cam_eye_pos).reshape(3,1)]),
                                np.array([0, 0, 0, 1])])
        # print("T = ", T)
        # Convert opencv convention to python convention
        # By a 180 degrees rotation along X
        Tc = np.array([[1,   0,    0,  0],
                    [0,   1,    0,  0],
                    [0,   0,    1,  0],
                    [0,   0,    0,  1]]).reshape(4,4)
        # pybullet pse is the inverse of the pose from the ROS-TF
        T=Tc@np.linalg.inv(T)
        # The transpose is needed for respecting the array structure of the OpenGL
        viewMatrix = T.T.reshape(16)
        return viewMatrix

    def _set_controller(self):
        logger.info(controller_type=self.controller_type)

        if self.controller_type == "diffusion":
            from controllers.rtvs import DiffusionController

            self.controller = DiffusionController(
                np.asarray(Image.open(self.img_goal_path).convert("RGB")),
                self.cam_to_gt_R,
            )
        else:
            logger.info("Controller not found!")

    def render(self):
        far = 100
        near = 0.01
        print(self.projectionmatrix)
        img_arr = p.getCameraImage(
                    self.width,
                    self.height,
                    viewMatrix = self.cvPose2BulletView(),
                    projectionMatrix=self.projectionmatrix,
                    renderer=p.ER_BULLET_HARDWARE_OPENGL,
                    shadow = False,
                )

        width, height, rgba, depth, mask = img_arr
        # print(depth.shape, len(depth))
        depth_buffer_tiny = np.reshape(depth, [self.width, self.height])
        depth_tiny = far * near / (far - (far - near) * depth_buffer_tiny)
        rgba = bytes(rgba)
        im_rgb = Image.frombytes('RGBA', (width, height), rgba).convert("RGB")
        return np.asarray(im_rgb), np.asarray(depth_buffer_tiny)

    def take_step(self, v):
        V = v[0]
        p.stepSimulation()
        vx, vy, vz = 1, 1, 1
        # v1, v2, v3 = V[2], V[0], V[1]
        v1, v2, v3 = V[0], V[1], V[2]

        new_pos = [self.cam_eye_pos[0] + vx*self.dt*v1, self.cam_eye_pos[1] + vy*self.dt*v2, self.cam_eye_pos[2] + vz*self.dt*v3]
        self.cam_eye_pos = new_pos
    
    def run(self):
        logger.info("Run start", cam_eye_pos=self.cam_eye_pos)
        
        img_src, depth_src = self.render()
        self.itr+=1
        Image.fromarray(img_src).save(self.folder + "%06d.png" % (self.itr))
        plt.imsave(self.folder + "%06d_depth.png" % (0), depth_src)

        pre_img_src = img_src

        img_goal = np.asarray(Image.open(self.img_goal_path).convert("RGB"))
        photo_error_val = mse_(img_src, img_goal)
        perrors = []
        logger.info("Initial Photometric Error: ", photo_error_val)
       
        start_time = time.time()
        step = 0

        while photo_error_val > 2000 and step < 300:
            self.itr+=1
            step+=1
            vel = self.controller.get_vel(img_src, pre_img_src, depth_src)
            
            photo_error_val = mse_(img_src, img_goal)
            perrors.append(photo_error_val)

            
            self.take_step([vel])

            logger.info("Cam Eye Position: ", self.cam_eye_pos)
            logger.info("Step Number: ", step)
            logger.info("Velocity : ", vel.round(8))
            logger.info("Photometric Error : ", photo_error_val)


            pre_img_src = img_src
            img_src, depth_src = self.render()
            Image.fromarray(img_src).save(self.folder + "%06d.png" % (self.itr))
            # img_src = np.asarray(img_src)
            
        time_taken = time.time() - start_time
        plt.plot(perrors)
        plt.savefig(self.folder + "000000.png")
        p.disconnect()
        return self.cam_eye_pos, self.itr

    def __del__(self):
        return

def simulate(urfd_path, cam_pos, des_img, start_itr, **kwargs):
    env = Diffusion(urfd_path, cam_pos, des_img, start_itr)
    new_cam_eye, itr = env.run()
    return new_cam_eye, itr

def main(urfd_path, des_img, cam_pos, start_itr):
    logger.info("Urdf: ", urfd_path, " des_img: ", des_img, " cam_pos: ", cam_pos)
    new_cam_eye, itr = simulate(
        urfd_path,
        cam_pos,
        des_img,
        start_itr,
    )
    return new_cam_eye, itr

if __name__ == "__main__":
    main(os.path.join(ycb_objects.getDataPath(), "YcbChipsCan", "model.urdf"), "./0000_0008.png", [0.45, 0.,   0.12], 0)