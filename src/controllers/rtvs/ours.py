import numpy as np
import torch

from utils.img_saver import ImageSaver
from .utils.photo_error import mse_

from utils.logger import logger

from .base_rtvs import BaseRtvs
from .our_dcem_model import Model
from .utils.flow_utils import flow2img


class Ours(BaseRtvs):
    def __init__(
        self,
        img_goal: np.ndarray,
        cam_k: np.ndarray,
        ct=1,
        horizon=2,
        LR=0.005,
        iterations=10,
    ):
        self.vs_lstm = Model().to(device="cuda:0")
        super().__init__(img_goal, cam_k, ct, horizon, LR, iterations)

    def get_vel(self, img_src, obj_vel, pre_img_src=None, depth=None):
        """
        img_src = current RGB camera image
        prev_img_src = previous RGB camera image
                    (to be used for depth estimation using flowdepth)
        """
        img_goal = self.img_goal
        flow_utils = self.flow_utils
        vs_lstm = self.vs_lstm
        loss_fn = self.loss_fn
        optimiser = self.optimiser
        img_src = img_src
        if depth is not None:
            depth = depth
        if pre_img_src is not None:
            pre_img_src = pre_img_src
        ct = self.ct

        photo_error_val = mse_(img_src, img_goal)
        # if photo_error_val < 6000 and photo_error_val > 3600:
        #     self.horizon = 10 * (photo_error_val / 6000)
        # elif photo_error_val < 3000:
        #     self.horizon = 6

        iou_score = self.get_iou(img_src)
        self.cnt = 0 if not hasattr(self, "cnt") else self.cnt + 1
        # obj_mask = self.detect_mask(img_src, ((50, 100, 100), (70, 255, 255)))[
        #     ::ct, ::ct
        # ]
        f12 = flow_utils.flow_calculate(img_src, img_goal)[::ct, ::ct]
        # f12 = f12 * obj_mask
        # f12[:] = 0
        ImageSaver.save_flow_img(flow2img(f12), self.cnt)

        if depth is None:
            flow_depth_proxy = (
                flow_utils.flow_calculate(img_src, pre_img_src).astype("float64")
            )[::ct, ::ct]
            flow_depth = np.linalg.norm(flow_depth_proxy, axis=2).astype("float64")
            final_depth = 0.1 / ((1 + np.exp(-1 / flow_depth)) - 0.5)
        else:
            final_depth = (depth[::ct, ::ct] + 1) / 10

        vel, Lsx, Lsy, Ox, Oy = get_interaction_data(
            final_depth, ct, self.cam_k, obj_vel
        )
        # Lsx = Lsx * obj_mask
        # Lsy = Lsy * obj_mask
        # Ox = Ox * obj_mask
        # Oy = Oy * obj_mask

        Lsx = torch.tensor(Lsx, dtype=torch.float32).to(device="cuda:0")
        Lsy = torch.tensor(Lsy, dtype=torch.float32).to(device="cuda:0")
        Ox = torch.tensor(Ox, dtype=torch.float32).to(device="cuda:0")
        Oy = torch.tensor(Oy, dtype=torch.float32).to(device="cuda:0")
        f12 = torch.tensor(f12, dtype=torch.float32).to(device="cuda:0")
        f12 = vs_lstm.pooling(f12.permute(2, 0, 1).unsqueeze(dim=0))

        for itr in range(self.iterations):
            vs_lstm.v_interm = []
            vs_lstm.f_interm = []
            vs_lstm.mean_interm = []

            vs_lstm.zero_grad()
            f_hat = vs_lstm.forward(vel, Lsx, Lsy, Ox, Oy, self.horizon, f12)
            loss = loss_fn(f_hat, f12)

            logger.debug(ours_mse=loss.item() ** 0.5, ours_itr=itr)
            loss.backward(retain_graph=True)
            optimiser.step()

        # Do not accumulate flow and velocity at train time
        vs_lstm.v_interm = []
        vs_lstm.f_interm = []
        vs_lstm.mean_interm = []

        with torch.no_grad():
            f_hat = vs_lstm.forward(
                vel, Lsx, Lsy, Ox, Oy, -self.horizon, f12.to(torch.device("cuda:0"))
            )

        vel = vs_lstm.v_interm[0].detach().cpu().numpy()
        logger.info(RAW_OUR_VELOCITY=vel)
        return vel, iou_score, photo_error_val


def get_interaction_data(d1, ct, cam_k, obj_vel):
    kx = cam_k[0, 0]
    ky = cam_k[1, 1]
    Cx = cam_k[0, 2]
    Cy = cam_k[1, 2]
    print(kx, ky, Cx, Cy)

    xyz = np.zeros([d1.shape[0], d1.shape[1], 3])
    Lsx = np.zeros([d1.shape[0], d1.shape[1], 6])
    Lsy = np.zeros([d1.shape[0], d1.shape[1], 6])
    p = obj_vel[0]
    q = obj_vel[1]
    r = obj_vel[2]

    d1[d1 == 0] = np.median(d1)

    def xyz_func(i, j, k):
        i = i.astype(int)
        j = j.astype(int)
        return (
            ((0.5 * (k - 1) * (k - 2)) * (ct * j - Cx) / kx)
            + ((-k * (k - 2)) * (ct * i - Cy) / ky)
            + ((0.5 * k * (k - 1)) * (d1[i, j]))
        )

    def lsx_func(i, j, k):
        i = i.astype(int)
        j = j.astype(int)
        k = k.astype(int)
        return (
            ((k == 0).astype(int) * -1 / xyz[i, j, 2])
            + ((k == 2).astype(int) * xyz[i, j, 0] / xyz[i, j, 2])
            + ((k == 3).astype(int) * xyz[i, j, 0] * xyz[i, j, 1])
            + ((k == 4).astype(int) * (-(1 + xyz[i, j, 0] ** 2)))
            + ((k == 5).astype(int) * xyz[i, j, 1])
        )

    def lsy_func(i, j, k):
        i = i.astype(int)
        j = j.astype(int)
        k = k.astype(int)
        return (
            ((k == 1).astype(int) * -1 / xyz[i, j, 2])
            + ((k == 2).astype(int) * xyz[i, j, 1] / xyz[i, j, 2])
            + ((k == 3).astype(int) * (1 + xyz[i, j, 1] ** 2))
            + ((k == 4).astype(int) * -xyz[i, j, 0] * xyz[i, j, 1])
            + ((k == 5).astype(int) * -xyz[i, j, 0])
        )

    xyz = np.fromfunction(xyz_func, (d1.shape[0], d1.shape[1], 3), dtype=float)
    Lsx = np.fromfunction(lsx_func, (d1.shape[0], d1.shape[1], 6), dtype=float)
    Lsy = np.fromfunction(lsy_func, (d1.shape[0], d1.shape[1], 6), dtype=float)

    Ox = (1 / d1[:, :, None]) * (p - xyz[:, :, 0:1] * r)
    Oy = (1 / d1[:, :, None]) * (q - xyz[:, :, 1:2] * r)

    return None, Lsx, Lsy, Ox, Oy
