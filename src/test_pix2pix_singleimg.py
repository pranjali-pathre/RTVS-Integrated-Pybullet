from PIL import Image
import requests
import torch
from diffusers import StableDiffusionInstructPix2PixPipeline
import argparse
import os
import transformers
import random
import diffusion
import shutil
import numpy as np

def get_args():
    parser = argparse.ArgumentParser(description="Instructpix2pix test script options")
    parser.add_argument("--data_path", type=str, default="./test",
                        help="Path to the root data directory")
    parser.add_argument("--save_path", type=str, default="./test/",
                        help="Path to saved models")
    parser.add_argument(
        "--load_weights_folder",
        type=str,
        default="./instruct-pix2pix-model/instruct-pix2pix-model/",
        help="Path to a pretrained model used for initialization")
    parser.add_argument("--num_inference_steps", type=int, default=100)
    parser.add_argument("--image_guidance_scale", type=float, default=1.5)
    parser.add_argument("--guidance_scale", type=float, default=7.5)
    return parser.parse_args()

def mse_(image_1, image_2):
	imageA=np.asarray(image_1)
	imageB=np.asarray(image_2)  
	err = np.sum((imageA.astype("float") - imageB.astype("float"))**2)
	err /= float(imageA.shape[0] * imageA.shape[1])
	return err

def main():
    opt = get_args()
    if not os.path.exists(opt.save_path):
        os.makedirs(opt.save_path)

    model_id = opt.load_weights_folder  
    pipe = StableDiffusionInstructPix2PixPipeline.from_pretrained(
    model_id, torch_dtype=torch.float16, use_safetensors=True, safety_checker = None,
    ).to("cuda")
    generator = torch.Generator("cuda").manual_seed(0)

    filelist= [file for file in os.listdir(opt.data_path) if file.endswith('.png') or file.endswith('jpg')]
    filelist.sort()

    prompts = ['open the door',
            'grip the handle and push the door open',
            'use the handle to open the door',
            'unlatch the door',
            'push the door',
            'unbolt the door',
            'go in front of the door',
            'go to the door']

    num_inference_steps = opt.num_inference_steps
    image_guidance_scale = opt.image_guidance_scale
    guidance_scale = opt.guidance_scale

    ptr = 0
    file = filelist[-1]

    err = 1e5

    pos = [3., 0.053, 0.9425]
    curr_itr = 0
    data_path_servo = "./results_diffusion/"
    if not os.path.exists(data_path_servo):
        os.makedirs(data_path_servo)
    # else:
    #     shutil.rmtree(data_path_servo, ignore_errors=False)
    #     os.makedirs(data_path_servo)

    while 1:
        print("*******************Running: ", ptr, " *****************")
        ptr+=1
        
        url = os.path.join(opt.data_path, file)
        init_image = Image.open(url).convert("RGB")
        init_image = init_image.resize((256, 256))
        edited_image = pipe(
            prompts[0],
            image=init_image,
            num_inference_steps=num_inference_steps,
            image_guidance_scale=image_guidance_scale,
            guidance_scale=guidance_scale,
            generator=generator,
        ).images[0]
        new_file = file.replace(".png", "")
        new_file = str(int(new_file) + 1).zfill(6) + ".png"

        img_save = os.path.join(opt.data_path, new_file)
        edited_image.save(img_save)
        des_img = img_save
        
        gen_image = Image.open(img_save).convert("RGB")
        err = mse_(init_image, gen_image)
        if err < 200:
            print("Minimum error reached!")
            break

        print("*******************Servoing started for: ", ptr, "Error: ", err, " *****************")
        new_pos, curr_itr = diffusion.main("./door/8897/mobility.urdf", des_img, pos, curr_itr)
        pos = new_pos   
        print("*******************Servoing done for: ", ptr, " *****************")

        filelist= [file for file in os.listdir(data_path_servo) if file.endswith('.png') or file.endswith('jpg')]
        filelist.sort()
        shutil.copyfile(os.path.join(data_path_servo, filelist[-1]), os.path.join(opt.data_path, filelist[-1]))

        filelist= [file for file in os.listdir(opt.data_path) if file.endswith('.png') or file.endswith('jpg')]
        filelist.sort()
        file = filelist[-1]
        # break

if __name__ == "__main__":
    main()