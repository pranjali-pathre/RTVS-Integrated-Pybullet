import numpy as np
from ibvs_helper import IBVSHelper
from utils.logger import logger
from scipy.spatial.transform import Rotation as R


class Controller:
    def __init__(
        self,
        grasp_time,
        post_grasp_dest,
        box_size,
        conveyor_level,
        ee_pos_scale,
        max_speed,
    ):
        self.grasp_time = grasp_time
        self.post_grasp_dest = post_grasp_dest
        self.box_size = box_size
        self.max_speed = max_speed
        self.conveyor_level = conveyor_level
        self.ee_pos_scale = ee_pos_scale

    def _action_vel_to_target_pos(self, action_vel, ee_pos):
        return ee_pos + action_vel * self.ee_pos_scale

    def _target_pos_to_action_vel(self, tar_pos, ee_pos):
        return (tar_pos - ee_pos) / self.ee_pos_scale

    def get_action(self):
        raise NotImplementedError


class GTController(Controller):
    def __init__(
        self,
        grasp_time,
        post_grasp_dest,
        box_size,
        conveyor_level,
        ee_pos_scale,
        max_speed=2,
    ):
        super().__init__(
            grasp_time,
            post_grasp_dest,
            box_size,
            conveyor_level,
            ee_pos_scale,
            max_speed,
        )

    def _predict_grasp_pos(self, obj_pos, obj_vel, cur_t):
        return obj_pos + obj_vel * (self.grasp_time - cur_t)

    def _get_vel_pre_grasp(self, ee_pos, obj_pos, obj_vel, cur_t):
        grasp_pos = self._predict_grasp_pos(obj_pos, obj_vel, cur_t)
        cur_target = grasp_pos.copy()
        # to allow box to penetrate further
        # arm config 1
        cur_target[2] += -self.box_size[2] * (1 / 3)
        # arm config 2
        # cur_target[1] += 0.06
        if cur_t <= 0.8 * self.grasp_time:
            cur_target[2] += 0.04
        dirn = cur_target - ee_pos
        speed = min(self.max_speed, 10 * np.linalg.norm(dirn))
        vel = speed * (dirn / np.linalg.norm(dirn))

        logger.debug(grasp_pos=grasp_pos, target_pos=cur_target)
        logger.debug(pred_vel=vel, pred_speed=round(np.linalg.norm(vel), 3))
        return vel

    def get_action(self, ee_pos, obj_pos, obj_vel, cur_t):
        action = np.zeros(5)
        if cur_t <= self.grasp_time:
            action[4] = -1
            action[:3] = self._get_vel_pre_grasp(ee_pos, obj_pos, obj_vel, cur_t)

        else:
            action[4] = 1
            if cur_t <= self.grasp_time + 0.5:
                action[:3] = [0, 0, 0.5]
            else:
                action[:3] = self.post_grasp_dest - ee_pos
        return action


class VSController(Controller):
    def __init__(
        self,
        grasp_time: float,
        post_grasp_dest,
        box_size,
        conveyor_level,
        ee_pos_scale,
        ibvs_helper: IBVSHelper,
        cam_to_gt_R: R,
        max_speed=0.5,
    ):
        super().__init__(
            grasp_time,
            post_grasp_dest,
            box_size,
            conveyor_level,
            ee_pos_scale,
            max_speed,
        )
        self.ibvs_helper = ibvs_helper
        self.cam_to_gt_R = cam_to_gt_R

        self.ready_to_grasp = False
        self.real_grasp_time = None

    def _get_ee_val(self, rgb_img, depth_img):
        ee_vel_cam, err = self.ibvs_helper.get_velocity(rgb_img, depth_img)
        ee_vel_gt = self.cam_to_gt_R.apply(ee_vel_cam)
        speed = min(self.max_speed, np.linalg.norm(ee_vel_gt))
        vel = ee_vel_gt * (
            speed / np.linalg.norm(ee_vel_gt) if not np.isclose(speed, 0) else 1
        )
        if err < 0.05:
            self.ready_to_grasp = True

        logger.debug(pred_vel=vel, pred_speed=np.linalg.norm(vel), err=err)
        return vel

    def get_action(self, rgb_img, depth_img, cur_t, ee_pos):
        action = np.zeros(5)
        if cur_t <= self.grasp_time and not self.ready_to_grasp:
            action[4] = -1
            action[:3] = self._get_ee_val(rgb_img, depth_img)
            if cur_t <= 0.6 * self.grasp_time:
                tpos = self._action_vel_to_target_pos(action[:3], ee_pos)
                tpos[2] = max(tpos[2], self.conveyor_level + self.box_size[2] + 0.005)
                action[2] = self._target_pos_to_action_vel(tpos, ee_pos)[2]
        else:
            action[4] = 1
            if self.real_grasp_time is None:
                self.real_grasp_time = cur_t
            if cur_t <= self.real_grasp_time + 0.5:
                action[:3] = [0, 0, 0.5]
            else:
                action[:3] = self.post_grasp_dest - ee_pos
        return action
