import torch
import torch.nn as nn
import model.ops as ops

class Block(torch.nn.Module):
    def __init__(self, 
                 in_channels, out_channels,
                 group=1):
        super(Block, self).__init__()

        self.b1 = ops.ResidualBlock(64, 64)
        self.b2 = ops.ResidualBlock(64, 64)
        self.b3 = ops.ResidualBlock(64, 64)
        self.c1 = ops.BasicBlock(64*2, 64, 1, 1, 0)
        self.c2 = ops.BasicBlock(64*3, 64, 1, 1, 0)
        self.c3 = ops.BasicBlock(64*4, 64, 1, 1, 0)

    def forward(self, x):
        #c0 = o0 = x
        o0 = x
        c0 = o0

        b1 = self.b1(o0)
        c1 = torch.cat([c0, b1], dim=1)
        o1 = self.c1(c1)
        
        b2 = self.b2(o1)
        c2 = torch.cat([c1, b2], dim=1)
        o2 = self.c2(c2)
        
        b3 = self.b3(o2)
        c3 = torch.cat([c2, b3], dim=1)
        o3 = self.c3(c3)

        return o3
        

class Net(torch.nn.Module):
    def __init__(self, **kwargs):
        super(Net, self).__init__()
        
        self.scale = kwargs.get("scale")
        self.multi_scale = kwargs.get("multi_scale")
        self.group = kwargs.get("group", 1)

        self.sub_mean = ops.MeanShift((0.4488, 0.4371, 0.4040), sub=True)
        self.add_mean = ops.MeanShift((0.4488, 0.4371, 0.4040), sub=False)
        
        self.entry = nn.Conv2d(3, 64, 3, 1, 1)

        self.b1 = Block(64, 64)
        self.b2 = Block(64, 64)
        self.b3 = Block(64, 64)
        self.c1 = ops.BasicBlock(64*2, 64, 1, 1, 0)
        self.c2 = ops.BasicBlock(64*3, 64, 1, 1, 0)
        self.c3 = ops.BasicBlock(64*4, 64, 1, 1, 0)
        
        self.upsamplex2 = ops.UpsampleBlockX2(64)
        self.upsamplex3 = ops.UpsampleBlockX3(64)
        self.upsamplex4 = ops.UpsampleBlockX4(64)
        self.exit = nn.Conv2d(64, 3, 3, 1, 1)
                
    def forward(self, x, scale):
        x = self.sub_mean(x)
        x = self.entry(x)
        o0 = x
        c0 = o0

        b1 = self.b1(o0)
        c1 = torch.cat([c0, b1], dim=1)
        o1 = self.c1(c1)
        
        b2 = self.b2(o1)
        c2 = torch.cat([c1, b2], dim=1)
        o2 = self.c2(c2)
        
        b3 = self.b3(o2)
        c3 = torch.cat([c2, b3], dim=1)
        o3 = self.c3(c3)

        if scale==2:
            out = self.upsamplex2(o3)
            out = self.exit(out)
            out = self.add_mean(out)
            return out
        if scale==3:
            out = self.upsamplex3(o3)
            out = self.exit(out)
            out = self.add_mean(out)
            return out
        if scale==4:
            out = self.upsamplex4(o3)
            out = self.exit(out)
            out = self.add_mean(out)
            return out

