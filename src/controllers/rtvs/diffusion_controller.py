import warnings
import numpy as np
from .dcem_model import Model
from .calculate_flow import FlowNet2Utils
import torch
import os
from .utils.photo_error import mse_
from utils.img_saver import ImageSaver
from .utils.flow_utils import flow2img
from scipy.spatial.transform import Rotation as R

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"   # see issue #152
np.random.seed(0)
warnings.filterwarnings("ignore")
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
torch.manual_seed(0)
torch.autograd.set_detect_anomaly(True)


class DiffusionController:
    '''
    Code for RTVS: Real-Time Visual Servoing, IROS 2021
    '''
    def __init__(self,
                 img_goal: np.ndarray,
                 cam_to_gt_R: R,
                 cam_k: np.ndarray,
                 max_speed=0.9,
                 ct=1,
                 horizon=3,
                 LR=0.005,
                 iterations=10,
                 ):
        '''
        img_goal: RGB array for final pose
        ct = image downsampling parameter (high ct => faster but less accurate)
        LR = learning rate of NN
        iterations = iterations to train NN (high value => slower but more accurate)
        horizon = MPC horizon
        '''
        self.img_goal = img_goal
        self.horizon = horizon
        self.iterations = iterations
        self.ct = ct
        self.flow_utils = FlowNet2Utils()
        self.vs_lstm = Model().to(device="cuda:0")
        self.optimiser = torch.optim.Adam(self.vs_lstm.parameters(),
                                          lr=LR, betas=(0.93, 0.999))
        self.loss_fn = torch.nn.MSELoss(size_average=False)
        self.cam_to_gt_R = cam_to_gt_R
        self.cam_k = cam_k
        self.max_speed = max_speed
        
    def get_vel(self, img_src, pre_img_src, depth=None):
        '''
            img_src = current RGB camera image
            prev_img_src = previous RGB camera image 
                        (to be used for depth estimation using flowdepth)
        '''
        img_goal = self.img_goal
        flow_utils = self.flow_utils
        vs_lstm = self.vs_lstm
        loss_fn = self.loss_fn
        optimiser = self.optimiser
        ct = self.ct

        photo_error_val = mse_(img_src, img_goal)
        # if photo_error_val < 6000 and photo_error_val > 3600:
        #     self.horizon = 10*(photo_error_val/6000)
        # elif photo_error_val < 3000:
        #     self.horizon = 6

        self.cnt = 0 if not hasattr(self, "cnt") else self.cnt + 1
        f12 = flow_utils.flow_calculate(img_src, img_goal)[::ct, ::ct]
        ImageSaver.save_flow_img(flow2img(f12), self.cnt)
        
        if depth is None:
            flow_depth_proxy = flow_utils.flow_calculate(
                img_src, pre_img_src).astype('float64')
            Cy, Cx = flow_depth_proxy.shape[1]/2, flow_depth_proxy.shape[0]/2
            flow_depth = np.linalg.norm(flow_depth_proxy[::ct, ::ct], axis=2)
            flow_depth = flow_depth.astype('float64')
            # final_depth = 0.1*(1/(1+np.exp(-1/flow_depth)) - 0.5)
            final_depth = 0.1 / ((1 + np.exp(-1 / flow_depth)) - 0.5)
        else:
            final_depth = (depth[::ct, ::ct] + 1) / 10
        
        vel, Lsx, Lsy = get_interaction_data(final_depth, ct, self.cam_k)

        Lsx = torch.tensor(Lsx, dtype=torch.float32).to(device="cuda:0")
        Lsy = torch.tensor(Lsy, dtype=torch.float32).to(device="cuda:0")
        f12 = torch.tensor(f12, dtype=torch.float32).to(device="cuda:0")
        f12 = vs_lstm.pooling(f12.permute(2, 0, 1).unsqueeze(dim=0))

        for itr in range(self.iterations):
            vs_lstm.v_interm = []
            vs_lstm.f_interm = []
            vs_lstm.mean_interm = []

            vs_lstm.zero_grad()
            f_hat = vs_lstm.forward(vel, Lsx, Lsy, self.horizon, f12)
            loss = loss_fn(f_hat, f12)

            print("MSE:", str(np.sqrt(loss.item())))
            loss.backward(retain_graph=True)
            optimiser.step()

        #Do not accumulate flow and velocity at train time
        vs_lstm.v_interm = []
        vs_lstm.f_interm = []
        vs_lstm.mean_interm = []

        f_hat = vs_lstm.forward(vel, Lsx, Lsy, -self.horizon,
                                f12.to(torch.device('cuda:0')))
        
        vel = vs_lstm.v_interm[0].detach().cpu().numpy()

        ee_vel_cam = vel[:3]
        ee_vel_gt = self.cam_to_gt_R.apply(ee_vel_cam)
        ee_vel_gt_w = self.cam_to_gt_R.apply(vel[3:])

        speed = min(self.max_speed, np.linalg.norm(ee_vel_gt))
        vel[:3] = ee_vel_gt * (
            speed / np.linalg.norm(ee_vel_gt) if not np.isclose(speed, 0) else 1
        )
        vel[3:] = ee_vel_gt_w
        return vel

def get_interaction_data(d1, ct, cam_k):
    kx = cam_k[0, 0]
    ky = cam_k[1, 1]
    Cx = cam_k[0, 2]
    Cy = cam_k[1, 2]
    
    xyz = np.zeros([d1.shape[0], d1.shape[1], 3])
    Lsx = np.zeros([d1.shape[0], d1.shape[1], 6])
    Lsy = np.zeros([d1.shape[0], d1.shape[1], 6])

    med = np.median(d1)
    xyz = np.fromfunction(lambda i, j, k: 0.5*(k-1)*(k-2)*(ct*j-float(Cx))/float(kx)
     - k*(k-2)*(ct*i-float(Cy))/float(ky)
       + 0.5*k*(k-1)*((d1[i.astype(int), j.astype(int)] == 0)*med
         + d1[i.astype(int), j.astype(int)]), (d1.shape[0], d1.shape[1], 3), dtype=float)

    Lsx = np.fromfunction(lambda i, j, k: (k == 0).astype(int) * -1/xyz[i.astype(int), j.astype(int), 2]
     + (k == 2).astype(int) * xyz[i.astype(int), j.astype(int), 0]/xyz[i.astype(int), j.astype(int), 2]
      + (k == 3).astype(int) * xyz[i.astype(int), j.astype(int), 0]*xyz[i.astype(int), j.astype(int), 1]
         + (k == 4).astype(int)*(-(1+xyz[i.astype(int), j.astype(int), 0]**2))
           + (k == 5).astype(int)*xyz[i.astype(int), j.astype(int), 1], (d1.shape[0], d1.shape[1], 6), dtype=float)

    Lsy = np.fromfunction(lambda i, j, k: (k == 1).astype(int) * -1/xyz[i.astype(int), j.astype(int), 2]
     + (k == 2).astype(int) * xyz[i.astype(int), j.astype(int), 1]/xyz[i.astype(int), j.astype(int), 2]
      + (k == 3).astype(int) * (1+xyz[i.astype(int), j.astype(int), 1]**2)
       + (k == 4).astype(int)*-xyz[i.astype(int), j.astype(int), 0]*xyz[i.astype(int), j.astype(int), 1]
        + (k == 5).astype(int) * -xyz[i.astype(int), j.astype(int), 0], (d1.shape[0], d1.shape[1], 6), dtype=float)

    return None, Lsx, Lsy

# if __name__ == "__main__":
#     vel = np.zeros(6)
#     d1 = np.ones([256, 256])
#     _, Lsx, Lsy = get_interaction_data(d1, 1, 128, 128)
#     Lsx = np.expand_dims(Lsx, axis=2)
#     Lsy = np.expand_dims(Lsy, axis=2)
#     Ls = np.concatenate((Lsx, Lsy), axis=2)

#     flow = np.dot(Ls, vel)


