import argparse
import os
import shutil
from types import SimpleNamespace

import numpy as np
import pybullet as p
from airobot import Robot
from airobot.arm.ur5e_pybullet import UR5ePybullet as UR5eArm
from airobot.sensor.camera.rgbdcam_pybullet import RGBDCameraPybullet
from airobot.utils.common import clamp, euler2quat, euler2rot, quat2euler
from scipy.spatial.transform import Rotation as R

from utils.img_saver import ImageSaver
from utils.logger import logger
from utils.sim_utils import get_random_config
from ycb_objects.pybullet_object_models import ycb_objects

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
        init_cfg: dict,
        grasp_time=4,
        gui=False,
        config: dict = {},
        controller_type="gt",
        record=True,
        flowdepth=False,
    ):
        self.gui = gui
        try:
            self.robot = Robot(
                "ur5e_2f140",
                # Have to keep openGL render off for the texture to work
                pb_cfg={"gui": gui, "realtime": False, "opengl_render": False},
            )
            assert p.isConnected(self.robot.pb_client.get_client_id())
        except:
            self.robot = Robot(
                "ur5e_2f140",
                pb_cfg={"gui": gui, "realtime": False, "opengl_render": False},
            )

        self.cam: RGBDCameraPybullet = self.robot.cam
        self.cam.setup_camera(height=256 , width=256)
        self.arm: UR5eArm = self.robot.arm
        self.pb_client = self.robot.pb_client
        self.config_vals_set(init_cfg, grasp_time)
        self.reset()
        self.record_mode = record
        self.flowdepth = flowdepth
        self.depth_noise = config.get("dnoise", 0)
        self.controller_type = controller_type
        self._set_controller()

    def config_vals_set(self, init_cfg: dict, grasp_time=4):
        self.step_dt = 1 / 250
        self.ground_lvl = 0.851
        p.setTimeStep(self.step_dt)
        self._action_repeat = 10
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
        self.post_grasp_duration = 1

        self.ground = SimpleNamespace()
        self.ground.init_pos = [0, 0, self.ground_lvl - 0.005]
        self.ground.scale = 0.1

        self.belt = SimpleNamespace()
        self.belt.color = [0, 0, 0, 0]
        self.belt.scale = 0.1
        self.belt.motion_type = init_cfg["motion_type"]
        if self.belt.motion_type == "linear":
            self.belt.vel = np.array(init_cfg["obj_vel"])
            self.belt.vel[2] = 0
            self.belt.init_pos = np.array(init_cfg["obj_pos"])
        elif self.belt.motion_type == "circle":
            self.belt.center = np.array(init_cfg["obj_center"])
            self.belt.radius = init_cfg["obj_radius"]
            self.belt.w = init_cfg["obj_w"]
            self.belt.init_pos = self.belt.center + self.belt.radius * np.array(
                [1, 0, 0]
            )
        self.belt.init_pos[2] = self.ground_lvl
        self._control_belt_motion(0)

        # self.wall = SimpleNamespace()
        # self.wall.init_pos = self.belt.init_pos + [0, 1, 0]
        # self.wall.ori = np.deg2rad([90, 0, 0])
        # self.wall.texture_name = "../data/cream.png"
        # self.wall.scale = 1

        self.table = SimpleNamespace()
        self.table.pos = np.array([0.5, 0, -5.4])
        self.table.ori = np.deg2rad([0, 0, 90])

        self.box = SimpleNamespace()
        self.box.size = np.array([0.03, 0.06, 0.06])
        self.box.init_pos = np.array([*self.belt.init_pos[:2], 0.9])
        self.box.init_ori = np.deg2rad([0, 0, 90])
        # self.box.init_ori = [0, 0, np.arctan2(self.belt.vel[1], self.belt.vel[0])]
        self.box.color = [0, 1, 0, 1]

    def _set_controller(self):
        logger.info(controller_type=self.controller_type)
        if self.controller_type == "rtvs":
            from controllers.rtvs import Rtvs, RTVSController

            self.controller = RTVSController(
                self.grasp_time,
                self.ee_home_pos,
                self.box.size,
                self.conveyor_level,
                self._ee_pos_scale,
                Rtvs("./dest.png", self.cam.get_cam_int()),
                self.cam_to_gt_R,
                max_speed=0.7,
            )

        elif self.controller_type == "ours":
            from controllers.rtvs import Ours, OursController

            self.controller = OursController(
                self.grasp_time,
                self.ee_home_pos,
                self.box.size,
                self.conveyor_level,
                self._ee_pos_scale,
                Ours("./dest.png", self.cam.get_cam_int()),
                self.cam_to_gt_R,
                max_speed=0.7,
            )

        elif self.controller_type == "ibvs":
            from controllers.ibvs import IBVSController, IBVSHelper

            self.controller = IBVSController(
                self.grasp_time,
                self.ee_home_pos,
                self.box.size,
                self.conveyor_level,
                self._ee_pos_scale,
                IBVSHelper(
                    "./dest.png", self.cam.get_cam_int(), {"lambda": 0.4}, self.gui
                ),
                self.cam_to_gt_R,
            )

        else:
            from controllers.gt import GTController

            self.controller = GTController(
                self.grasp_time,
                self.ee_home_pos,
                self.box.size,
                self.conveyor_level,
                self._ee_pos_scale,
            )

    def _control_belt_motion(self, t=None, dt=None):
        if t is None:
            t = self.sim_time
        if dt is None:
            dt = self.step_dt * self._action_repeat

        if self.belt.motion_type == "circle":
            r = self.belt.radius
            w = self.belt.w
            self.belt.vel[0] = -w * r * np.sin(w * t)
            self.belt.vel[1] = w * r * np.cos(w * t)

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

        self.textures = {}

        def apply_col_texture(obj):
            assert hasattr(obj, "color") ^ hasattr(obj, "texture_name")
            if hasattr(obj, "color"):
                return p.changeVisualShape(obj.id, -1, rgbaColor=obj.color)
            if obj.texture_name not in self.textures:
                tex_id = p.loadTexture(obj.texture_name)
                assert tex_id >= 0
                self.textures[obj.texture_name] = tex_id
            else:
                tex_id = self.textures[obj.texture_name]
            obj.texture_id = tex_id
            p.changeVisualShape(obj.id, -1, textureUniqueId=tex_id)

        self.arm.go_home(ignore_physics=True)
        self.arm.eetool.open(ignore_physics=True)

        self.ground.id = p.loadURDF(
            "simple_plane.urdf", self.ground.init_pos, globalScaling=self.ground.scale
        )

        self.belt.id: int = p.loadURDF(
            "simple_plane.urdf", self.belt.init_pos, globalScaling=self.belt.scale
        )
        change_friction(self.belt.id, 2, 2)
        apply_col_texture(self.belt)

        # self.wall.id: int = p.loadURDF(
        #     "simple_plane.urdf",
        #     self.wall.init_pos,
        #     euler2quat(self.wall.ori),
        #     globalScaling=self.wall.scale,
        # )
        # apply_col_texture(self.wall)

        # self.box_id = self.robot.pb_client.load_geom(
        #     "box",
        #     size=(self.box.size / 2).tolist(),
        #     mass=1,
        #     base_pos=self.box.init_pos,
        #     rgba=self.box.color,
        #     base_ori=euler2quat(self.box.init_ori),
        # )
        self.box_id = p.loadURDF(
            os.path.join(ycb_objects.getDataPath(), "YcbChipsCan", "model.urdf"),
            basePosition=self.box.init_pos,
            baseOrientation=euler2quat(self.box.init_ori),
            globalScaling=0.5,
        )
        self.box.id = self.box_id

        # apply_col_texture(self.box)
        # self.box.color = [0, 1, 0, 1]
        # apply_col_texture(self.box)

        ee_pos, _, _, ee_euler = self.arm.get_ee_pose()
        logger.info(init_belt_pose=self.get_pos(self.belt), belt_vel=self.belt.vel)
        # to make arm transparent (not works on TINY_RENDERER when getting img)
        # [ p.changeVisualShape(self.arm.robot_id, i, rgbaColor=[0, 1, 1, 0])   for i in range(-1, 23)]
        logger.info(home_ee_pos=ee_pos, home_ee_euler=ee_euler)
        logger.info(home_jpos=self.arm.get_jpos())
        p.configureDebugVisualizer(p.COV_ENABLE_RENDERING, 1)
        self.prev_rgb = self.render(for_video=False)[0]
        self.gripper_ori = 0
        self.belt_poses = []
        self.step_cnt = 0
        # exit(0)

    @property
    def conveyor_level(self):
        return self.ground_lvl

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
            self.arm.set_jpos(jnt_pos, wait=False, ignore_physics=(action[4] != 1))
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
                focus_pt=[0, 0, 0.7], dist=1.5, yaw=90, pitch=-40, roll=0, height=256 , width=256
            )
        else:
            self.cam.set_cam_ext(pos=self.cam_pos, ori=self.cam_ori)

        cam_eye = self.cam_pos
        cam_dir = cam_eye + self.cam_to_gt_R.apply([0, 0, 0.1])
        p.addUserDebugLine(cam_dir, cam_eye, [0, 1, 0], 3, 0.5)
        
        rgb, depth, seg = self.cam.get_images(
            get_rgb, get_depth, get_seg, shadow=0, lightDirection=[0, 0, 2]
        )
        if noise is not None:
            depth *= np.random.normal(loc=1, scale=noise, size=depth.shape)
        return rgb, depth, seg, cam_eye

    def save_img(self, rgb, t):
        if not self.record_mode:
            return
        ImageSaver.save_rgb(rgb, t)

    def run(self):
        state = {
            "obj_motion": {"motion_type": self.belt.motion_type},
            "cam_int": [],
            "cam_ext": [],
            # "pcd_3d": [],
            # "pcd_rgb": [],
            "obj_pos": [],
            "obj_corners": [],
            "ee_pos": [],
            "joint_pos": [],
            "action": [],
            "t": [],
            "err": [],
            "cam_eye": [],
            "joint_vel": [],
            "images": {
                "rgb": [],
                "depth": [],
                "seg": [],
            },
            "grasp_time": self.grasp_time,
            "grasp_success": 0,
        }
        if self.belt.motion_type == "circle":
            state["obj_motion"].update(
                {
                    "radius": self.belt.radius,
                    "center": self.belt.center,
                    "w": self.belt.w,
                }
            )
        elif self.belt.motion_type == "linear":
            state["obj_motion"].update({"vel": self.belt.vel})

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
            shutil.rmtree("imgs", ignore_errors=True)
            os.makedirs("imgs", exist_ok=True)

        total_sim_time = self.grasp_time + self.post_grasp_duration
        grasping = False
        grasping_success = False
        t = 0
        while t < int(np.ceil(time_steps * total_sim_time)):
            rgb, depth, seg, cam_eye = self.render(
                for_video=False, noise=self.depth_noise
            )
            # pcd_3d, pcd_rgb = self.cam.get_pcd(
            #     depth_min=-np.inf, depth_max=np.inf, rgb_image=rgb, depth_image=depth
            # )
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
                # (pcd_3d, "pcd_3d"),
                # (pcd_rgb, "pcd_rgb"),
                (self.cam.get_cam_ext(), "cam_ext"),
                (self.cam.get_cam_int(), "cam_int"),
            )
            self.save_img(rgb, t)
            observations = {
                "cur_t": self.sim_time,
                "ee_pos": self.ee_pos,
            }
            if self.controller_type == "gt":
                observations["obj_pos"] = self.obj_pos
                observations["belt_vel"] = self.belt.vel
                observations["obj_vel"] = self.belt.vel
            elif self.controller_type == "ibvs":
                observations["rgb_img"] = rgb
                observations["depth_img"] = depth
            elif self.controller_type == "rtvs":
                observations["rgb_img"] = rgb
                observations["depth_img"] = depth
                observations["prev_rgb_img"] = self.prev_rgb
            elif self.controller_type == "ours":
                observations["rgb_img"] = rgb
                observations["depth_img"] = depth
                observations["prev_rgb_img"] = self.prev_rgb
                observations["obj_vel"] = self.belt.vel

            if self.flowdepth:
                observations.pop("depth_img")

            action, err = self.controller.get_action(observations)
            if not grasping and self.controller.ready_to_grasp:
                logger.debug("Grasping start")
                grasping = True
                total_sim_time = self.sim_time + self.post_grasp_duration
                self.grasp_time = self.sim_time
                state["grasp_time"] = self.sim_time
            multi_add_to_state((action, "action"), (err, "err"))
            logger.info(time=self.sim_time, action=action)
            logger.info(ee_pos=self.ee_pos, obj_pos=self.obj_pos)
            logger.info(
                dist=np.round(np.linalg.norm(self.ee_pos - self.obj_pos), 3),
                iou_err=err,
            )
            self._control_belt_motion()
            self.step(action)
            self.prev_rgb = rgb
            if (
                grasping
                and not grasping_success
                and ((self.obj_pos - self.belt.init_pos - self.box.size / 2)[2] > 0.02)
            ):
                grasping_success = True
                logger.info("Grasping success")
            t += 1
        state["grasp_success"] = grasping_success

        logger.info("Run end", ee_pos=self.ee_pos, obj_pose=self.obj_pos)
        return state

    def __del__(self):
        p.disconnect(self.pb_client.get_client_id())


def simulate(init_cfg: dict, gui, controller, **kwargs):
    env = URRobotGym(init_cfg, gui=gui, controller_type=controller, **kwargs)
    return env.run()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-c",
        "--controller",
        type=str,
        default="gt",
        help="controller",
        choices=["gt", "rtvs", "ibvs", "ours"],
    )
    parser.add_argument("--random", action="store_true")
    parser.add_argument("--gui", action="store_true", help="show gui")
    parser.add_argument("--circle", action="store_true", help="move in circle")
    parser.add_argument("--no-gui", dest="gui", action="store_false", help="no gui")
    parser.add_argument("--record", action="store_true", help="save imgs")
    parser.add_argument("--flowdepth", action="store_true", help="use flow_depth")
    parser.add_argument("--no-record", dest="record", action="store_false")
    parser.set_defaults(gui=False, record=True)
    parser.add_argument("--seed", type=int, default=None, help="seed")
    args = parser.parse_args()
    if args.seed is not None:
        np.random.seed(args.seed)

    init_cfg = {
        "motion_type": "linear",
        "obj_pos": [0.45, -0.05, 0.851],
        # "obj_vel": [-0.01, 0.03, 0],
        "obj_vel": [0., 0., 0.],
    }
    if args.circle:
        init_cfg = {
            "motion_type": "circle",
            "obj_center": [0.45, -0.05, 0.851],
            "obj_w": 3,
            "obj_radius": 0.03,
        }

    if args.random:
        if args.circle:
            init_cfg.update(
                {
                    "obj_w": np.random.uniform(1, 5),
                    "obj_radius": np.random.uniform(0.01, 0.05),
                }
            )
        else:
            init_cfg.update(
                {
                    "obj_vel": np.random.uniform(
                        [-0.05, -0.05, -0.05], [0.05, 0.05, 0.05]
                    ),
                }
            )
    print(init_cfg)
    return simulate(
        init_cfg,
        args.gui,
        args.controller,
        record=args.record,
        flowdepth=args.flowdepth,
    )


if __name__ == "__main__":
    main()
