import torch
import numpy as np
from PIL import Image
import cv2

from .flownet2.models import FlowNet2
from .utils.frame_utils import read_gen, flow_to_image


class Args:
    fp16 = False
    rgb_max = 255.0


class FlowNet2Utils:
    net = FlowNet2(Args()).cuda()
    net.load_state_dict(torch.load("../data/FlowNet2_checkpoint.pth.tar")["state_dict"])

    def __init__(self):
        pass

    @staticmethod
    def resize_img(img, new_size):
        return cv2.resize(img, new_size, interpolation=cv2.INTER_LANCZOS4)
        # return np.asarray(
        #     Image.fromarray(img).resize(new_size, Image.LANCZOS)
        # )

    @torch.no_grad()
    def flow_calculate(self, img1, img2):
        assert img1.shape == img2.shape, f"shapes dont match {img1.shape}, {img2.shape}"
        org_shape = img1.shape[1::-1]
        req_shape = (512, 384)
        if org_shape != req_shape:
            img1 = self.resize_img(img1, req_shape)
            img2 = self.resize_img(img2, req_shape)

        images = [img1, img2]

        images = np.array(images).transpose(3, 0, 1, 2)
        im = torch.from_numpy(images.astype(np.float32)).unsqueeze(0).cuda()
        result = self.net(im)[0].squeeze()
        data = result.data.cpu().numpy().transpose(1, 2, 0)

        if org_shape != req_shape:
            data = self.resize_img(data, org_shape)

        return data

    def writeFlow(self, name, flow):
        f = open(name, "wb")
        f.write("PIEH".encode("utf-8"))
        np.array([flow.shape[1], flow.shape[0]], dtype=np.int32).tofile(f)
        flow = flow.astype(np.float32)
        flow.tofile(f)
        f.flush()
        f.close()

    def save_flow_with_image(self, folder):
        img_source_path = folder + "/results/" + "test.rgb.00000.00000.png"
        img_goal_path = folder + "/des.png"
        img_src = read_gen(img_source_path)
        img_goal = read_gen(img_goal_path)
        f12 = self.flow_calculate(img_src, img_goal)
        self.writeFlow(folder + "/flow.flo", f12)
        flow_image = flow_to_image(f12)
        im = Image.fromarray(flow_image)
        im.save(folder + "/flow.png")
