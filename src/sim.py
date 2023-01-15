import os
from types import SimpleNamespace
import numpy as np
import pybullet as p
from airobot import Robot
from airobot.arm.ur5e_pybullet import UR5ePybullet as UR5eArm
from airobot.utils.common import clamp
from airobot.utils.common import euler2quat, quat2euler, euler2rot
from airobot.sensor.camera.rgbdcam_pybullet import RGBDCameraPybullet
from controllers import GTController, VSController
from ibvs_helper import IBVSHelper
from utils.sim_utils import get_random_config
from utils.logger import logger
from scipy.spatial.transform import Rotation as R
from PIL import Image
import argparse

np.set_string_function(
    lambda x: repr(np.round(x, 4))
    .replace("(", "")
    .replace(")", "")
    .replace("array", "")
    .replace("       ", " "),
    repr=False,
)


class URRobotGym:
    def __init__(
        self,
        belt_init_pose=[0.45, -0.05, 0.851],
        belt_vel=[0.03, 0.05, 0],
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
            assert p.isConnected(self.robot.pb_client.get_client_id())
        except:
            self.robot = Robot(
                "ur5e_2f140",
                pb_cfg={"gui": gui, "realtime": False, "opengl_render": False},
            )

        self.cam: RGBDCameraPybullet = self.robot.cam
        self.arm: UR5eArm = self.robot.arm
        self.pb_client = self.robot.pb_client
        self.inference_mode = inference_mode
        self.config_vals_set(belt_init_pose, belt_vel, grasp_time)
        self.reset()
        self.record_mode = record
        self.depth_noise = config.get("dnoise", 0)
        if self.inference_mode:
            self.vs_controller = VSController(
                self.grasp_time,
                self.ee_home_pos,
                self.box.size,
                self.conveyor_level,
                self._ee_pos_scale,
                IBVSHelper("./dest.png", self.cam.get_cam_int(), {"lambda": 0.4}, gui),
                self.cam_to_gt_R,
            )
        else:
            self.gt_controller = GTController(
                self.grasp_time,
                self.ee_home_pos,
                self.box.size,
                self.conveyor_level,
                self._ee_pos_scale,
            )

    def config_vals_set(self, belt_init_pose, belt_vel, grasp_time=4):
        self.step_dt = 0.01
        p.setTimeStep(self.step_dt)
        self._action_repeat = 4
        self._ee_pos_scale = self.step_dt * self._action_repeat

        self.cam_link_anchor_id = 22  #  or ('ur5_ee_link-gripper_base') link 10
        self.cam_ori = np.deg2rad([-105, 0, 0])
        # arm config 1
        self.cam_pos_delta = np.array([0, -0.2, 0.02])
        self.ee_home_pos = [0.5, -0.13, 0.9]
        self.ee_home_ori = np.deg2rad([-180, 0, -180])
        self.arm._home_position = [-0.0, -1.66, -1.92, -1.12, 1.57, 1.57]
        self.arm._home_position = self.arm.compute_ik(
            self.ee_home_pos, self.ee_home_ori
        )

        # arm config 2
        # self.cam_pos_delta = np.array([0, -0.08, 0.05])
        # self.ee_home_ori = np.deg2rad([90, 0, -180])
        # self.ee_home_pos = [0.48, -0.3, 0.99]
        # self.arm._home_position = [-0.75, -2.95, -0.59, -2.74, 2.39, -0.0]

        self.cam_to_gt_R = R.from_euler("xyz", self.cam_ori)
        self.ref_ee_ori = self.ee_home_ori[:]
        self.grasp_time = grasp_time
        # if self.inference_mode:
        #     self.grasp_time = 8
        self.post_grasp_duration = 1

        self.belt = SimpleNamespace()
        self.belt.size = np.array([0.6, 0.6, 0.001])
        self.belt.vel = np.array(belt_vel)
        self.belt.vel[2] = 0
        self.belt.init_pos = np.array(belt_init_pose)
        self.belt.init_pos[2] = 0.851
        self.belt.color = [0, 1, 1, 1]

        self.table = SimpleNamespace()
        self.table.pos = np.array([0.5, 0, -5.4])
        self.table.ori = np.deg2rad([0, 0, 90])

        self.box = SimpleNamespace()
        self.box.size = np.array([0.03, 0.06, 0.06])
        self.box.init_pos = np.array([*self.belt.init_pos[:2], 0.9])
        self.box.init_ori = np.deg2rad([0, 0, 90])
        # self.box.init_ori = [0, 0, np.arctan2(self.belt.vel[1], self.belt.vel[0])]
        self.box.color = [1, 0, 0, 1]

    def get_pos(self, obj):
        if isinstance(obj, int):
            obj_id = obj
        elif hasattr(obj, "id"):
            obj_id = obj.id
        return self.pb_client.get_body_state(obj_id)[0]

    def reset(self):
        p.configureDebugVisualizer(p.COV_ENABLE_RENDERING, 0)
        p.resetDebugVisualizerCamera(
            cameraTargetPosition=[0, 0, 0],
            cameraDistance=2,
            cameraPitch=-40,
            cameraYaw=90,
        )
        change_friction = lambda id, lf=0, sf=0: p.changeDynamics(
            bodyUniqueId=id, linkIndex=-1, lateralFriction=lf, spinningFriction=sf
        )
        self.arm.go_home(ignore_physics=True)
        self.arm.eetool.open(ignore_physics=True)
        ee_pos, _, _, ee_euler = self.arm.get_ee_pose()
        logger.info(home_ee_pos=ee_pos, home_ee_euler=ee_euler)
        logger.info(home_jpos=self.arm.get_jpos)
        # exit(0)
        self.table.id: int = self.robot.pb_client.load_urdf(
            "table/table.urdf", self.table.pos, euler2quat(self.table.ori), scaling=10
        )
        change_friction(self.table.id, 0, 0)
        self.belt.id: int = self.pb_client.load_geom(
            "box",
            size=(self.belt.size / 2).tolist(),
            mass=0,
            base_pos=self.belt.init_pos,
            rgba=self.belt.color,
        )
        change_friction(self.belt.id, 2, 2)
        self.box_id = self.robot.pb_client.load_geom(
            "box",
            size=(self.box.size / 2).tolist(),
            mass=1,
            base_pos=self.box.init_pos,
            rgba=self.box.color,
            base_ori=euler2quat(self.box.init_ori),
        )
        logger.info(init_belt_pose=self.get_pos(self.belt))
        p.configureDebugVisualizer(p.COV_ENABLE_RENDERING, 1)
        self.render(for_video=False)
        self.gripper_ori = 0
        self.belt_poses = []
        self.step_cnt = 0
        # exit(0)

    @property
    def conveyor_level(self):
        belt_pos = self.get_pos(self.belt)
        return belt_pos[2] + self.belt.size[2] / 2

    @property
    def sim_time(self):
        return self.step_cnt * self.step_dt

    def step(self, action):
        self.apply_action(action)
        self.belt_poses.append(self.get_pos(self.belt))

    def apply_action(self, action, use_belt=True):
        if not isinstance(action, np.ndarray):
            action = np.array(action).flatten()
        if action.size != 5:
            raise ValueError(
                "Action should be [d_x, d_y, d_z, angle, open/close gripper]."
            )
        pos = self.ee_pos + action[:3] * self._ee_pos_scale
        pos[2] = max(pos[2], 0.01 + self.conveyor_level)

        # ## NOTE: OVERRIDING EE_ORI for now . Hence action[3] is ignored !!! ####
        # self.gripper_ori = ang_in_mpi_ppi(np.deg2rad(action[3]))
        # rot_vec = np.array([0, 0, 1]) * self.gripper_ori
        # rot_quat = rotvec2quat(rot_vec)
        # ee_ori = quat_multiply(self.ref_ee_ori, rot_quat)
        # ee_ori = R.from_euler("xyz", [-np.pi / 2, np.pi, 0]).as_quat()
        # ee_ori = R.from_euler("xyz", np.deg2rad([-75, 180, 0])).as_quat()
        jnt_pos = self.robot.arm.compute_ik(pos, ori=self.ee_home_ori)
        gripper_ang = self._scale_gripper_angle(action[4])

        for step in range(self._action_repeat):
            self.robot.arm.set_jpos(jnt_pos, wait=False)
            self.robot.arm.eetool.set_jpos(gripper_ang, wait=False)
            if use_belt:
                p.resetBaseVelocity(self.belt.id, self.belt.vel)
            self.robot.pb_client.stepSimulation()
            self.step_cnt += 1
        # logger.debug(action_target = pos, action_result = self.ee_pos, delta=(pos-self.ee_pos))

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
        points_base = np.asarray(points_base) * self.box.size[2] / np.sqrt(2)
        points = points_base @ rotmat + pos
        points = np.asarray(sorted(points.tolist())).round(5)
        return np.asarray(sorted(points.tolist()))

    @property
    def ee_pos(self):
        return self.arm.get_ee_pose()[0]

    @property
    def cam_pos(self):
        ee_base_pos = p.getLinkState(self.arm.robot_id, self.cam_link_anchor_id)[0]
        return ee_base_pos + self.cam_pos_delta

    def render(
        self, get_rgb=True, get_depth=True, get_seg=True, for_video=True, noise=None
    ):
        if for_video:
            self.robot.cam.setup_camera(
                focus_pt=[0, 0, 0.7], dist=1.5, yaw=90, pitch=-40, roll=0
            )
        else:
            self.cam.set_cam_ext(pos=self.cam_pos, ori=self.cam_ori)

        cam_eye = self.cam_pos
        cam_dir = cam_eye + self.cam_to_gt_R.apply([0, 0, 0.1])
        p.addUserDebugLine(cam_dir, cam_eye, [0, 1, 0], 3, 0.5)
        rgb, depth, seg = self.cam.get_images(get_rgb, get_depth, get_seg)
        if noise is not None:
            depth *= np.random.normal(loc=1, scale=noise, size=depth.shape)
        return rgb, depth, seg, cam_eye

    def run(self):
        state = {
            "cam_int": [],
            "cam_ext": [],
            "pcd_3d": [],
            "pcd_rgb": [],
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

        def add_to_state(val, *args):
            if isinstance(val, list) or (
                isinstance(val, np.ndarray) and val.dtype == np.float64
            ):
                val = np.asarray(val, np.float32)
            nonlocal state
            list_val = state
            for arg in args:
                list_val = list_val[arg]
            list_val.append(val)

        def multi_add_to_state(*args):
            for arg in args:
                add_to_state(*arg)

        logger.info("Run start", obj_pose=self.obj_pos)
        time_steps = 1 / (self.step_dt * self._action_repeat)

        if self.record_mode:
            os.makedirs("imgs", exist_ok=True)

        total_sim_time = self.grasp_time + self.post_grasp_duration
        for t in range(int(np.ceil(time_steps * total_sim_time))):
            rgb, depth, seg, cam_eye = self.render(
                for_video=False, noise=self.depth_noise
            )
            pcd_3d, pcd_rgb = self.cam.get_pcd()
            multi_add_to_state(
                (self.obj_pos, "obj_pos"),
                (self.obj_pos_8, "obj_corners"),
                (self.ee_pos, "ee_pos"),
                (self.arm.get_jpos(), "joint_pos"),
                (self.arm.get_jvel(), "joint_vel"),
                (t / time_steps, "t"),
                (cam_eye, "cam_eye"),
                (rgb, "images", "rgb"),
                (depth, "images", "depth"),
                (seg, "images", "seg"),
                (pcd_3d, "pcd_3d"),
                (pcd_rgb, "pcd_rgb"),
                (self.cam.get_cam_int(), "cam_int"),
                (self.cam.get_cam_ext(), "cam_ext"),
            )
            if self.record_mode:
                Image.fromarray(rgb).save("imgs/" + str(int(t)).zfill(5) + ".png")

            if not self.inference_mode:
                action = self.gt_controller.get_action(
                    self.ee_pos, self.obj_pos, self.belt.vel, self.sim_time
                )
            else:
                action = self.vs_controller.get_action(
                    rgb, depth, self.sim_time, self.ee_pos
                )

            add_to_state(action, "action")
            logger.info(time=self.sim_time, action=action)
            logger.info(ee_pos=self.ee_pos, obj_pos=self.obj_pos)
            self.step(action)
            if self.sim_time == self.grasp_time:
                logger.info(ee_pos=self.ee_pos)

        logger.info("Run end", obj_pose=self.obj_pos, ee_pos=self.ee_pos)
        return state


def simulate(init_cfg, gui, inf_mode, record):
    env = URRobotGym(
        *init_cfg,
        gui=gui,
        inference_mode=inf_mode,
        record=record,
    )
    return env.run()


def main():
    parser = argparse.ArgumentParser()
    # run in inference mode using model
    parser.add_argument("-i", "--inference", action="store_true")
    parser.add_argument("--random", action="store_true")
    parser.add_argument("--gui", action="store_true", help="show gui")
    parser.add_argument("--no-gui", dest="gui", action="store_false", help="no gui")
    parser.add_argument("--record", action="store_true", help="save imgs")
    parser.add_argument("--no-record", dest="record", action="store_false")
    parser.set_defaults(gui=True, record=True)
    parser.add_argument("--seed", type=int, default=None, help="seed")
    args = parser.parse_args()
    if args.seed is not None:
        np.random.seed(args.seed)

    init_cfg = ([0.45, -0.05, 0.851], [0.03, 0.05, 0])
    if args.random:
        init_cfg = get_random_config()[:2]

    return simulate(init_cfg, args.gui, args.inference, args.record)


if __name__ == "__main__":
    main()
