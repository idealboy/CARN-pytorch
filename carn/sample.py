import os
import json
import time
import importlib
import argparse
import numpy as np
from collections import OrderedDict
import torch
import torch.nn as nn
from torch.autograd import Variable
from dataset import TestDataset
from PIL import Image

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str)
    parser.add_argument("--ckpt_path", type=str)
    parser.add_argument("--group", type=int, default=1)
    parser.add_argument("--sample_dir", type=str)
    parser.add_argument("--test_data_dir", type=str, default="dataset/Urban100")
    parser.add_argument("--cuda", action="store_true")
    parser.add_argument("--scale", type=int, default=4)
    parser.add_argument("--N", type = int, default=2)
    parser.add_argument("--shave", type=int, default=20)

    return parser.parse_args()


def save_image(tensor, filename):
    tensor = tensor.cpu()
    ndarr = tensor.mul(255).clamp(0, 255).byte().permute(1, 2, 0).numpy()
    im = Image.fromarray(ndarr)
    im.save(filename)

#N : the number to split into in vertical and horizon direction
def split_patch_pos(height, width, N, shave, h_chop, w_chop):
    h_step = int(height / N)
    w_step = int(width / N)
    
    pos_list = []

    for j in range(N):
        for i in range(N):

            half_shave = int(shave / 2)

            pos = [j * h_step - shave, j * h_step + h_chop, i * w_step - shave, i * w_step + w_chop]

            if pos[0] > 0 and pos[2] > 0 and pos[1] < height and pos[3] < width:
                pos[0] = pos[0] + half_shave
                pos[1] = pos[1] - half_shave
                pos[2] = pos[2] + half_shave
                pos[3] = pos[3] - half_shave
            else:
                if pos[0] > 0 and pos[1] < height:
                    pos[0] = pos[0] + half_shave
                    pos[1] = pos[1] - half_shave
                else:
                    if pos[0] <= 0:
                        pos[0] = 0
                        pos[1] = pos[0] + h_chop
                    if pos[1] >= height:
                        pos[1] = height
                        pos[0] = pos[1] - h_chop

                if pos[2] > 0 and pos[3] < width:
                    pos[2] = pos[2] + half_shave
                    pos[3] = pos[3] - half_shave
                else:
                    if pos[2] <= 0:
                        pos[2] = 0
                        pos[3] = pos[2] + w_chop
                    if pos[3] >= width:
                        pos[3] = width
                        pos[2] = pos[3] - w_chop

            pos_list.append(pos)

    return pos_list

# N: the number to split into in vertical and horizon direction
# scale : the resize scale
def merge_result_pos(split_pos, N, scale, shave, height, width, h_chop, w_chop):

    h_step = int(height / N) * scale
    w_step = int(width / N) * scale

    pos_list = []

    for j in range(N):
        for i in range(N):
            pos = split_pos[j * N + i]
            top = pos[0]
            bottom = pos[1]
            left = pos[2]
            right = pos[3]

            roi = [0, (bottom - top) * scale, 0, (right - left) * scale]

            if top > 0 and left > 0 and bottom < height and right < width:
                roi[0] = int(shave * scale / 2)
                roi[1] = roi[1] - roi[0]
                roi[2] = int(shave * scale / 2)
                roi[3] = roi[3] - roi[2]
            else:
                if top > 0 and bottom < height:
                    roi[0] = int(shave * scale / 2)
                    roi[1] = roi[1] - roi[0]
                else:
                    if top <= 0:
                        roi[1] = roi[1] - shave * scale
                    if bottom >= height:
                        roi[0] = shave * scale

                if left > 0 and right < width:
                    roi[2] = int(shave * scale / 2)
                    roi[3] = roi[3] - roi[2]
                else:
                    if left <= 0:
                        roi[3] = roi[3] - shave * scale
                    if right >= width:
                        roi[2] = shave * scale


            pos_list.append(roi)
    return pos_list


def sample(net, device, dataset, cfg):
    scale = cfg.scale
    for step, (hr, lr, name) in enumerate(dataset):
        if "DIV2K" in dataset.name:
            #t1 = time.time()
            #h, w = lr.size()[1:]
            #h_half, w_half = int(h/2), int(w/2)
            #h_chop, w_chop = h_half + cfg.shave, w_half + cfg.shave

            #pos_list = split_patch_pos(h, w, 2, cfg.shave)

            #lr_patch = torch.zeros((4, 3, h_chop, w_chop), dtype=torch.float)#1.0.1 torch
            #lr_patch[0].copy_(lr[:, 0:h_chop, 0:w_chop])
            #lr_patch[1].copy_(lr[:, 0:h_chop, w-w_chop:w])
            #lr_patch[2].copy_(lr[:, h-h_chop:h, 0:w_chop])
            #lr_patch[3].copy_(lr[:, h-h_chop:h, w-w_chop:w])


            #lr_patch = lr_patch.to(device)
            #
            #sr = net(lr_patch, cfg.scale).detach()

            #pos_list = merge_result_pos(pos_list, 2, 2, cfg.shave, h, w)
            #
            #h, h_half, h_chop = h*scale, h_half*scale, h_chop*scale
            #w, w_half, w_chop = w*scale, w_half*scale, w_chop*scale

            #result = torch.zeros((3, h, w), dtype=torch.float).to(device)#1.0.1 torch
            #result[:, 0:h_half, 0:w_half].copy_(sr[0, :, 0:h_half, 0:w_half])
            #result[:, 0:h_half, w_half:w].copy_(sr[1, :, 0:h_half, w_chop-w+w_half:w_chop])
            #result[:, h_half:h, 0:w_half].copy_(sr[2, :, h_chop-h+h_half:h_chop, 0:w_half])
            #result[:, h_half:h, w_half:w].copy_(sr[3, :, h_chop-h+h_half:h_chop, w_chop-w+w_half:w_chop])

            #sr = result
            #t2 = time.time()

            t1 = time.time()
            h, w = lr.size()[1:]
            N = cfg.N # we will split into N*N sub images from the whole imgae
            h_step, w_step = int(h/N), int(w/N)
            h_chop, w_chop = h_step + cfg.shave, w_step + cfg.shave

            pos_list = split_patch_pos(h, w, N, cfg.shave, h_chop, w_chop)

            lr_patch = torch.zeros((N*N, 3, h_chop, w_chop), dtype=torch.float)#1.0.1 torch

            for idx in range(N*N):
                lr_patch[idx].copy_(lr[:, pos_list[idx][0]:pos_list[idx][1], pos_list[idx][2]:pos_list[idx][3]])

            lr_patch = lr_patch.to(device)
            
            sr = net(lr_patch, cfg.scale).detach()

            pos_list = merge_result_pos(pos_list, N, scale, cfg.shave, h, w, h_chop * 2, w_chop * 2)
            
            h, h_step = h * scale, h_step * scale
            w, w_step = w * scale, w_step * scale

            result = torch.zeros((3, h, w), dtype=torch.float).to(device)#1.0.1 torch

            idx = 0
            for j in range(N):
                for i in range(N):
                    result[:, j * h_step : (j + 1) * h_step, i * w_step : (i + 1) * w_step].copy_(sr[idx, :, pos_list[idx][0] : pos_list[idx][1], pos_list[idx][2] : pos_list[idx][3]])
                    idx += 1

            sr = result
            t2 = time.time()
        else:
            t1 = time.time()
            lr = lr.unsqueeze(0).to(device)
            sr = net(lr, cfg.scale).detach().squeeze(0)
            lr = lr.squeeze(0)
            t2 = time.time()
        
        model_name = cfg.ckpt_path.split(".")[0].split("/")[-1]
        sr_dir = os.path.join(cfg.sample_dir,
                              model_name, 
                              cfg.test_data_dir.split("/")[-1],
                              "x{}".format(cfg.scale),
                              "SR")
        hr_dir = os.path.join(cfg.sample_dir,
                              model_name, 
                              cfg.test_data_dir.split("/")[-1],
                              "x{}".format(cfg.scale),
                              "HR")
        
        os.makedirs(sr_dir, exist_ok=True)
        os.makedirs(hr_dir, exist_ok=True)

        sr_im_path = os.path.join(sr_dir, "{}".format(name.replace("HR", "SR")))
        hr_im_path = os.path.join(hr_dir, "{}".format(name))

        save_image(sr, sr_im_path)
        save_image(hr, hr_im_path)
        print("Saved {} ({}x{} -> {}x{}, {:.3f}s)"
            .format(sr_im_path, lr.shape[1], lr.shape[2], sr.shape[1], sr.shape[2], t2-t1))


def main(cfg):
    module = importlib.import_module("model.{}".format(cfg.model))
    net = module.Net(multi_scale=True, 
                     group=cfg.group)
    print(json.dumps(vars(cfg), indent=4, sort_keys=True))

    state_dict = torch.load(cfg.ckpt_path)
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        name = k
        # name = k[7:] # remove "module."
        new_state_dict[name] = v

    net.load_state_dict(new_state_dict)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    net = net.to(device)
    
    dataset = TestDataset(cfg.test_data_dir, cfg.scale)
    sample(net, device, dataset, cfg)
 

if __name__ == "__main__":
    cfg = parse_args()
    main(cfg)
