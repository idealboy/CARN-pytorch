import os
import sys

import torch

import torchvision
import importlib
from collections import OrderedDict

os.environ["CUDA_VISIBLE_DEVICES"] = "4"

import os
import sys
import importlib
import argparse
import numpy as np
from collections import OrderedDict
import torch
import torch.nn as nn
import torchvision
from model.carn import Net


net = Net(multi_scale=True, scale=2, group=1)


state_dict = torch.load("carn_new.pth")
new_state_dict = OrderedDict()
for k, v in state_dict.items():
    name = k
    #print (name)
    #if "up2" in name:
    #    name = name.replace("upsample", "upsamplex{}".format(2))
    #elif "up3" in name:
    #    name = name.replace("upsample", "upsamplex{}".format(3))
    #elif "up4" in name:
    #    name = name.replace("upsample", "upsamplex{}".format(4))
    new_state_dict[name] = v

net.load_state_dict(new_state_dict)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
net = net.to(device)

sr = torch.jit.script(net)
sr.save("model.pt")

