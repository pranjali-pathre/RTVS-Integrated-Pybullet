import os
import cv2
import numpy as np
import pybullet as p
from gym import spaces
from airobot import Robot
from airobot.utils.common import ang_in_mpi_ppi
from airobot.arm.ur5e_pybullet import UR5ePybullet as UR5eArm
from airobot.utils.common import clamp
from airobot.utils.common import euler2quat, quat2euler, euler2rot
from airobot.utils.common import quat_multiply
from airobot.utils.common import rotvec2quat
from airobot.sensor.camera.rgbdcam_pybullet import RGBDCameraPybullet
from ibvs_helper import IBVSHelper
from utils.sim_utils import get_random_config
from utils.logger import logger
from scipy.spatial.transform import Rotation as R
from PIL import Image
import argparse

np.set_string_function(
    lambda x: repr(x)
    .replace("(", "")
    .replace(")", "")
    .replace("array", "")
    .replace("       ", " "),
    repr=False,
)


class URRobotGym:
    def __init__(
        self,
        # belt_init_pose=[0.35, -0.05, 0.851],
        belt_init_pose=[0.35, -0.07, 0.851],
        belt_vel=[0.02, 0.05, 0],
        grasp_time=4,
        gui=False,
        config: dict = {},
        inference_mode=False,
        record=True,
    ):
        try:
            self.robot = Robot(
                "ur5e_2f140",
                pb_cfg={"gui": gui, "realtime": False, "opengl_render": True},
            )
        except:
            self.robot = Robot(
                "ur5e_2f140",
                pb_cfg={"gui": gui, "realtime": False, "opengl_render": False},
            )

        self.cam: RGBDCameraPybullet = self.robot.cam
        self.arm: UR5eArm = self.robot.arm
        self.pb_client = self.robot.pb_client
        self.constants_set()
        self.config_vals_set(belt_init_pose, belt_vel, grasp_time)
        p.setTimeStep(self.step_dt)
        self.reset()
        self.inference_mode = inference_mode
        self.record_mode = record
        self.depth_noise = config.get("dnoise", 0)
        if self.inference_mode:
            self.ibvs = IBVSHelper(
                "./dest.png", self.cam.get_cam_int(), lm_params={"lambda": 0.4}
            )

    def config_vals_set(self, belt_init_pose, belt_vel, grasp_time=4):
        self.grasp_time = grasp_time
        self.belt_vel = np.array(belt_vel)
        self.belt_vel[2] = 0
        self.belt_init_pos = np.array(belt_init_pose)
        self.belt_init_pos[2] = 0.851

    def constants_set(self):
        self.step_dt = 0.01
        self._action_repeat = 10

        # useless stuff
        self.ee_ori = [-np.sqrt(2) / 2, np.sqrt(2) / 2, 0, 0]
        self._action_bound = 1.0
        self._ee_pos_scale = 0.14
        self._ee_ori_scale = np.deg2rad(1)
        self._action_high = np.array([self._action_bound] * 5)
        self.action_space = spaces.Box(
            low=-self._action_high, high=self._action_high, dtype=np.float32
        )
        state_low = np.full(len(self._get_obs()), -float("inf"))
        state_high = np.full(len(self._get_obs()), float("inf"))
        self.observation_space = spaces.Box(state_low, state_high, dtype=np.float32)

    def reset(self):
        p.resetDebugVisualizerCamera(
            cameraTargetPosition=[0, 0, 0],
            cameraDistance=2,
            cameraPitch=-40,
            cameraYaw=90,
        )
        self.arm.reset()
        self.arm.go_home(ignore_physics=True)
        self.ref_ee_ori = self.robot.arm.get_ee_pose()[1]
        self.gripper_ori = 0
        self.belt_poses = []
        self.step_cnt = 0
        for i in range(1):
            self.apply_action([0, 0, 0, 90, 0], False)
        self.step_cnt = 0
        self.table_id = self.robot.pb_client.load_urdf(
            "table/table.urdf",
            [0.5, 0, -5.4],
            euler2quat([0, 0, np.pi / 2]),
            scaling=10,
        )
        self.pb_client.changeDynamics(
            bodyUniqueId=self.table_id,
            linkIndex=-1,
            lateralFriction=0,
            spinningFriction=0,
        )
        self.belt_id = self.pb_client.load_geom(
            "box",
            size=[0.3, 0.3, 0.001],
            mass=0,
            base_pos=self.belt_init_pos,
            rgba=[0, 1, 1, 1],
        )
        self.pb_client.changeDynamics(
            bodyUniqueId=self.belt_id,
            linkIndex=-1,
            lateralFriction=2,
            spinningFriction=2,
        )
        box_pos = self.belt_init_pos
        box_pos[2] = 0.9
        self.box_size = [0.015, 0.03, 0.03]
        self.box_id = self.robot.pb_client.load_geom(
            "box",
            size=self.box_size,
            mass=1,
            base_pos=box_pos,
            rgba=[1, 0, 0, 1],
            # base_ori=euler2quat([0, 0, np.arctan2(self.belt_vel[1], self.belt_vel[0])]),
            base_ori=euler2quat([0, 0, np.pi / 2]),
        )
        # self.box_id = p.loadURDF(
        #     os.path.join(ycb_objects.getDataPath(), "YcbTomatoSoupCan", "model.urdf"),
        #     basePosition=[0.5, 0.12, 0.9],
        #     baseOrientation=euler2quat([0, 0, 0]),
        #  )
        logger.info(init_belt_pose=self.pb_client.get_body_state(self.belt_id)[0])
        self.render(for_video=False)
        return self._get_obs()

    @property
    def sim_time(self):
        return self.step_cnt * self.step_dt

    def step(self, action):
        # self.cam.setup_camera(
        #     focus_pt=self.arm.get_ee_pose()[0], dist=0.3, pitch=-30, yaw=-90
        # )
        # rgb = self.cam.get_images()[0]
        # cam_pose = self.cam.get_cam_ext()[:3, 3]
        self.apply_action(action)
        self.belt_poses.append(self.pb_client.get_body_state(self.belt_id)[0])
        # logger.debug(f"Belt pose = {self.pb_client.get_body_state(self.belt_id)[0]}")
        state = self._get_obs()
        done = False
        info = dict()
        reward = -1
        return state, reward, done, info

    def _get_obs(self):
        jpos = self.robot.arm.get_jpos()
        jvel = self.robot.arm.get_jvel()
        state = jpos + jvel
        return state

    def apply_action(self, action, use_belt=True):
        if not isinstance(action, np.ndarray):
            action = np.array(action).flatten()
        if action.size != 5:
            raise ValueError(
                "Action should be [d_x, d_y, d_z, angle, open/close gripper]."
            )
        pos, quat, rot_mat, euler = self.robot.arm.get_ee_pose()
        pos += action[:3] * self._ee_pos_scale

        self.gripper_ori = ang_in_mpi_ppi(action[3] * self._ee_ori_scale)
        rot_vec = np.array([0, 0, 1]) * self.gripper_ori
        rot_quat = rotvec2quat(rot_vec)
        ee_ori = quat_multiply(self.ref_ee_ori, rot_quat)
        jnt_pos = self.robot.arm.compute_ik(pos, ori=ee_ori)
        gripper_ang = self._scale_gripper_angle(action[4])

        for step in range(self._action_repeat):
            self.robot.arm.set_jpos(jnt_pos, wait=False)
            self.robot.arm.eetool.set_jpos(gripper_ang, wait=False)
            if use_belt:
                p.resetBaseVelocity(self.belt_id, self.belt_vel)
            self.robot.pb_client.stepSimulation()
            self.step_cnt += 1

    def _scale_gripper_angle(self, command):
        """
        Convert the command in [-1, 1] to the actual gripper angle.
        command = -1 means open the gripper.
        command = 1 means close the gripper.

        Args:
            command (float): a value between -1 and 1.
                -1 means open the gripper.
                1 means close the gripper.

        Returns:
            float: the actual gripper angle
            corresponding to the command.
        """
        command = clamp(command, -1.0, 1.0)
        close_ang = self.robot.arm.eetool.gripper_close_angle
        open_ang = self.robot.arm.eetool.gripper_open_angle
        cmd_ang = (command + 1) / 2.0 * (close_ang - open_ang) + open_ang
        return cmd_ang

    @property
    def obj_pos(self):
        return self.pb_client.get_body_state(self.box_id)[0]

    @property
    def obj_pos_8(self):
        pos, quat = self.pb_client.get_body_state(self.box_id)[:2]
        euler_z = quat2euler(quat)[2]
        rotmat = euler2rot([0, 0, euler_z])

        points_base = [
            [1, 1, 1],
            [1, -1, 1],
            [-1, 1, 1],
            [-1, -1, 1],
        ]
        points_base = np.asarray(points_base) * self.box_size[2] / np.sqrt(2)
        points = points_base @ rotmat + pos
        points = np.asarray(sorted(points.tolist())).round(5)
        return np.asarray(sorted(points.tolist()))

    @property
    def ee_pos(self):
        return self.arm.get_ee_pose()[0]

    def render(
        self, get_rgb=True, get_depth=True, get_seg=True, for_video=True, noise=None
    ):
        if for_video:
            self.robot.cam.setup_camera(
                focus_pt=[0, 0, 0.7], dist=1.5, yaw=90, pitch=-40, roll=0
            )
        else:
            # self.arm.get_jpos('ur5_ee_link-gripper_base') # link 10
            ee_base_pos, ee_base_ori = p.getLinkState(self.arm.robot_id, 22)[:2]
            ee_base_pos = np.asarray(ee_base_pos)
            ee_base_ori = np.asarray(ee_base_ori)
            # ee_base_pos = ee_base_pos + np.asarray([0.3, -0.2, 0.45])
            # ori_wrt_cam = rotvec2quat([0, 0, 0])
            # ori_cam_wrt_world = quat_multiply(ee_base_ori, ori_wrt_cam)
            # cam_yaw, cam_pitch, cam_roll = np.rad2deg(quat2euler(ori_cam_wrt_world))
            focus_point = np.asarray(ee_base_pos) + [0.0, 0, -0.15]
            self.robot.cam.setup_camera(
                focus_pt=focus_point,
                dist=0.2,
                yaw=-0,
                pitch=-60,
                roll=0,
            )
            self.cam_pos = ee_base_pos + [0, -0.2, 0]
            self.cam_ori = [-np.deg2rad(105), 0, 0]
            self.cam.set_cam_ext(pos=self.cam_pos, ori=self.cam_ori)
            self.cam_ori = [-np.deg2rad(105 + 90), 0, 0]

        cam_eye = -(
            (self.robot.cam.view_matrix[:3, :3]) @ self.robot.cam.view_matrix[3, :3]
        ).flatten()
        p.addUserDebugLine(ee_base_pos, cam_eye, [0, 1, 0], 3, 0.5)
        rgb, depth, seg = self.cam.get_images(
            get_rgb=get_rgb, get_depth=get_depth, get_seg=get_seg
        )
        if noise is not None:
            depth *= np.random.normal(loc=1, scale=noise, size=depth.shape)

        return rgb, depth, seg, cam_eye

    @staticmethod
    def get_obj_pos(self, obj_poses, dt):
        vel = np.gradient(obj_poses, axis=1).mean(axis=1)
        next_pos = obj_poses[-1] + vel * dt
        return next_pos, vel

    def run(self):
        self.grasp_time = 8
        get_state = True
        if get_state:
            state = {
                "obj_pos": [],
                "obj_corners": [],
                "ee_pos": [],
                "joint_pos": [],
                "action": [],
                "t": [],
                "cam_eye": [],
                "joint_vel": [],
                "images": {
                    "rgb": [],
                    "depth": [],
                    "seg": [],
                },
            }

        def get_grasp_pos(t, v=None, p=None):
            if v is None:
                v = self.belt_vel
            if p is None:
                p = self.obj_pos
            return (self.grasp_time - t) * v + p + [0, 0, -0.025]

        logger.info("Run start", obj_pose=self.obj_pos)
        logger.info(grasp_pos=get_grasp_pos(0))
        time_steps = 1 / (self.step_dt * self._action_repeat)
        ready_for_grasp = False
        time_after_grasp = 0
        post_grasp_duration = 1
        post_grasp_dest = self.ee_pos[:]
        post_grasp_dest[2] = 1

        # to preserve EE rotation angle in later stages
        prev_action = np.zeros(5)
        prev_action[3] = 90
        if self.record_mode:
            os.makedirs("imgs", exist_ok=True)

        def get_ee_vel(rgb_img, depth_img):
            ibvs_vel, err = self.ibvs.get_velocity(rgb_img, depth_img)[:3]
            gr_ibvs_vel = R.from_euler("xyz", self.cam_ori).inv().apply(ibvs_vel)
            if self.ee_pos[2] < self.obj_pos[2] - 0.0:
                gr_ibvs_vel[2] = 0
            p.addUserDebugLine(
                self.ee_pos, self.ee_pos + gr_ibvs_vel, [1, 0, 0], 3, 0.3
            )
            logger.info(ibvs_vel=gr_ibvs_vel, ee_pos=self.ee_pos, err=err)
            if err < 0.05:
                nonlocal ready_for_grasp
                ready_for_grasp = True
            return gr_ibvs_vel

        for t in np.arange(time_steps * (self.grasp_time + post_grasp_duration)):
            if get_state:
                state["obj_pos"].append(self.obj_pos)
                state["obj_corners"].append(self.obj_pos_8)
                state["ee_pos"].append(self.ee_pos)
                state["joint_pos"].append(self.arm.get_jpos())
                state["joint_vel"].append(self.arm.get_jvel())
                state["t"].append(t / time_steps)
                rgb, depth, seg, cam_eye = self.render(
                    for_video=False, noise=self.depth_noise
                )
                if self.record_mode:
                    Image.fromarray(rgb).save("imgs/" + str(int(t)).zfill(5) + ".png")
                state["cam_eye"].append(cam_eye)
                state["images"]["rgb"].append(rgb)
                state["images"]["depth"].append(depth)
                state["images"]["seg"].append(seg)

            action = np.zeros(5)
            action[:] = prev_action
            if t > time_steps * self.grasp_time:
                ready_for_grasp = True
            if not ready_for_grasp:
                if self.inference_mode:
                    move_dir = get_ee_vel(rgb, depth)
                    if t < time_steps * self.grasp_time / 2:
                        if (
                            self.ee_pos[2]
                            < 0.01 + self.obj_pos[2] + self.box_size[2] / 2
                        ):
                            move_dir[2] = 0
                else:
                    move_dir = get_grasp_pos(self.sim_time) - self.ee_pos
                    logger.debug(grasp_pose=get_grasp_pos(self.sim_time).round(4))
                    logger.debug(
                        ee_pos=self.ee_pos.round(4), obj_pose=self.obj_pos.round(4)
                    )
                    if t < time_steps * self.grasp_time / 2:
                        move_dir[2] = 0
                    else:
                        move_dir[2] *= 2
                        if self.ee_pos[2] < self.obj_pos[2]:
                            move_dir[2] = 0
                action[:3] = move_dir
                action[4] = -1
                action[3] = 90
                # action[3] = (
                #     np.rad2deg(
                #         np.arctan(
                #             np.tan(np.arctan2(self.belt_vel[0], self.belt_vel[1]))
                #         )
                #     )  # in range -90, 90
                #     + 90
                # )
            else:  # move up after grasp time for 2s
                action[4] = 1
                if self.inference_mode:
                    # cv2.imshow("corners", rgb[:, :, ::-1])
                    # cv2.waitKey(1)
                    action[:3] = post_grasp_dest - self.ee_pos
                    if time_after_grasp < 0.1:
                        action[:3] = prev_action[:3]
                        action[2] = -0.035
                    if time_after_grasp < 0.07:
                        action[4] = -1
                else:
                    action[:3] = post_grasp_dest - self.ee_pos
                    if time_after_grasp < 0.1:
                        action[:3] = prev_action[:3]
                        action[2] = 0.08
                action[:3] = np.clip(action[:3], -0.05, 0.05)
                time_after_grasp += self.step_dt
                # if time_after_grasp > post_grasp_duration:
                #     break

            state["action"].append(action)
            logger.info(
                time=round(t * self.step_dt, 3),
                ready_for_grasp=ready_for_grasp,
                action=action.round(4),
            )
            self.step(action)
            prev_action[:] = action
            if self.sim_time == self.grasp_time:
                logger.info(ee_pos=self.ee_pos.round(3))

        logger.info("Run end", obj_pose=self.obj_pos, ee_pos=self.ee_pos.round(3))
        if get_state:
            return state


def main():
    config = {}
    config["dnoise"] = 0
    parser = argparse.ArgumentParser()
    # run in inference mode using model
    parser.add_argument("-i", "--inference", action="store_true")
    parser.add_argument("--random", action="store_true")
    parser.add_argument("--gui", action="store_true", help="show gui")
    parser.add_argument("--no-gui", dest="gui", action="store_false", help="no gui")
    parser.set_defaults(gui=True)
    parser.add_argument("-r", "--record", action="store_true", default=True)
    # parser.add_argument(
    #     "-w",
    #     "--weights",
    #     type=str,
    #     default=os.path.join(
    #         config["log_dir"],
    #         config["experiment_name"],
    #         "train_logs",
    #         "last.pth",
    #     ),
    # )
    parser.add_argument(
        "--seed", type=int, default=np.random.randint(0, int(1e7)), help="seed"
    )
    args = parser.parse_args()
    np.random.seed(args.seed)
    # config["weights_file"] = args.weights
    # config["ds_file"] = config["dataset_log_file"]

    if args.random:
        env = URRobotGym(
            *(get_random_config()[:2]),
            gui=args.gui,
            config=config,
            inference_mode=args.inference,
            record=args.record,
        )
    else:
        env = URRobotGym(
            gui=args.gui,
            config=config,
            inference_mode=args.inference,
            record=args.record,
        )
    # input()
    env.run()


if __name__ == "__main__":
    main()
