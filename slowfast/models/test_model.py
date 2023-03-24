import torch.nn as nn
import torch
from STFM import STFM
import time


inputs = torch.rand(1, 320, 8, 14, 14).cuda()
ori_img = torch.rand(1, 3, 16, 224, 224).cuda()
cf = STFM(img_size=224, patch_size=16, dim=768, num_tokens=32, input_channel=320,
            dense=True, pos_kernel=3, add_loss=True, T_multiple=2).cuda()
print(cf.patch_embed)
for i,j in cf.named_parameters():
    if 'proj' in i:
        j.requires_grad = False

for i,j in cf.named_parameters():
    print(i, j.requires_grad)
start = time.time()
out,pos = cf(inputs, ori_img)
end = time.time()
print(out.shape)
print(end-start)