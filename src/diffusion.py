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
from airobot import Robot
from airobot.arm.ur5e_pybullet import UR5ePybullet as UR5eArm
from scipy.spatial.transform import Rotation as R
from airobot.cfgs.assets.pybullet_camera import get_sim_cam_cfg
import cfg
from airobot.utils.pb_util import create_pybullet_client

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
        folder="./results_pybullet_omega/",
        controller_type="diffusion",
    ):
        self.radius = 10.0
        self.dt = 1./240
        self.pb_client = create_pybullet_client(gui=False,
                                            realtime=False,
                                            opengl_render=False)
        
        self.pb_client.setAdditionalSearchPath(pybullet_data.getDataPath())
        # p.configureDebugVisualizer(shadowMapWorldSize=5)
        # p.configureDebugVisualizer(shadowMapResolution=8192)
        self.pb_client.configureDebugVisualizer(lightPosition=[self.radius*math.sin(3*self.dt), self.radius*math.cos(3*self.dt), 33])
        self.pb_client.setGravity(0, 0, -9.807)
        self.pb_client.configureDebugVisualizer(self.pb_client.COV_ENABLE_RENDERING, 1)
        self.pb_client.configureDebugVisualizer(self.pb_client.COV_ENABLE_SHADOWS,0) 

        self.cam_eye_pos = cam_eye_pos
        self.width = 256
        self.height = 256

        self.cam = RGBDCameraPybullet(cfg.get_cfg(), self.pb_client)
        self.cam.setup_camera(height=256 , width=256)
       
        self.angles_deg = [-9.32119081e+01, -3.54676998e-07,  9.59774447e+01]
        self.cam_ori = np.deg2rad(self.angles_deg)
        self.cam_to_gt_R = R.from_euler("xyz", self.cam_ori)

        self.floor = SimpleNamespace()
        self.floor.ori = [0, 0, 0]
        self.floor.pos = [0, 0, 0]
        self.floor.id = self.pb_client.loadURDF('plane.urdf', self.floor.pos, self.euler2quat(self.floor.ori))

        self.ht_gnd = 0
        self.r2d2 = self.pb_client.loadURDF(urfd_path, [0, 0, self.ht_gnd])
        while pairwise_collision(self.floor.id, self.r2d2) == True:
            self.ht_gnd += 0.025
            self.pb_client.resetBasePositionAndOrientation(self.r2d2, [0, 0, self.ht_gnd], self.pb_client.getQuaternionFromEuler([0,0,0]))
        
        self.controller_type = controller_type
        
        self.folder = folder
        self.itr = start_itr
        self.img_goal_path = des_img

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
    
    def _set_controller(self):
        logger.info(controller_type=self.controller_type)

        if self.controller_type == "diffusion":
            from controllers.rtvs import DiffusionController

            self.controller = DiffusionController(
                np.asarray(Image.open(self.img_goal_path).convert("RGB")),
                self.cam_to_gt_R,
                self.cam.get_cam_int(),
            )
        else:
            logger.info("Controller not found!")

    def render(self, get_rgb=True, get_depth=True, get_seg=True, for_video=True, noise=None):
        self.cam.set_cam_ext(pos=np.asarray(self.cam_eye_pos), ori=self.cam_ori)
        cam_eye = self.cam_eye_pos

        rgb, depth, seg = self.cam.get_images(
            get_rgb, get_depth, get_seg, shadow=0, lightDirection=[0, 0, 2]
        )
        if noise is not None:
            depth *= np.random.normal(loc=1, scale=noise, size=depth.shape)
        return rgb, depth, seg

    def step(self, v):
        V = v[0]
        self.pb_client.stepSimulation()
        rf = 180/np.pi

        vx, vy, vz = 1, 1, 1
        v1, v2, v3 = V[0], V[1], V[2]
        # w1, w2, w3 = -rf*V[3], rf*V[4], rf*V[5]
        w1, w2, w3 = rf*V[4], -rf*V[3], rf*V[5]

        new_pos = [self.cam_eye_pos[0] + vx*self.dt*v1, self.cam_eye_pos[1] + vy*self.dt*v2, self.cam_eye_pos[2] + vz*self.dt*v3]
        self.cam_eye_pos = new_pos

        new_w = [self.angles_deg[0] + self.dt*w1, self.angles_deg[1] + self.dt*w2, self.angles_deg[2] + self.dt*w3]
        # new_w[1] = 0
        logger.info("Taking step before: ", self.angles_deg, " after: ", new_w)
        self.angles_deg = new_w
        self.cam_ori = np.deg2rad(self.angles_deg)
        self.cam_to_gt_R = R.from_euler("xyz", self.cam_ori)
    
    def run(self):
        logger.info("Run start", cam_eye_pos=self.cam_eye_pos)
        
        img_src, depth_src, _ = self.render()
        self.itr+=1
        Image.fromarray(img_src).save(self.folder + "%06d.png" % (self.itr))

        pre_img_src = img_src

        img_goal = np.asarray(Image.open(self.img_goal_path).convert("RGB"))
        photo_error_val = mse_(img_src, img_goal)
        perrors = []
        logger.info("Initial Photometric Error: ", photo_error_val)
       
        start_time = time.time()
        step = 0

        while photo_error_val > 650 and step < 300:
            self.itr+=1
            step+=1
            photo_error_val = mse_(img_src, img_goal)
            perrors.append(photo_error_val)

            vel = self.controller.get_vel(img_src, pre_img_src, depth_src)
            self.step([vel])

            logger.info("Cam Eye Position: ", self.cam_eye_pos)
            logger.info("Step Number: ", step)
            logger.info("Velocity : ", vel.round(4))
            logger.info("Photometric Error : ", photo_error_val)

            pre_img_src = img_src
            img_src, depth_src, _ = self.render()
            Image.fromarray(img_src).save(self.folder + "%06d.png" % (self.itr))
            
        time_taken = time.time() - start_time
        # plt.plot(perrors)
        # plt.savefig(self.folder + "000000.png")
        self.pb_client.disconnect()
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
    # main("./door/8897/mobility.urdf", "./test/000007.png", [1.9482712939381668, 0.016201307992615745, 0.9478492264367028], 584)
    main(os.path.join(ycb_objects.getDataPath(), "YcbChipsCan", "model.urdf"), "./trajs/0000_0008.png", [0.53168941, 0.05567119, 0.23], 0)
