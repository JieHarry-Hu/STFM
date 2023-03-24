import torch.nn as nn
import torch
from einops import rearrange
from timm.models.layers import DropPath
import numpy as np
from visualize import visualize_grid_attention_v2
#imput:[B, C, T, H, W]

def conv_3xnxn(inp, oup, kernel_size=3, stride=3, groups=1, T_multiple=2):
    return nn.Conv3d(inp, oup, (3, kernel_size, kernel_size), (T_multiple, stride, stride), (1, 0, 0), groups=groups)
def conv_3x3x3(inp, oup, groups=1):
    return nn.Conv3d(inp, oup, (3, 3, 3), (1, 1, 1), (1, 1, 1), groups=groups)
def conv_nxnxn(inp, oup, kernel_size=5, groups=1):
    padding = (kernel_size - 1) // 2
    return nn.Conv3d(inp, oup, (kernel_size, kernel_size, kernel_size), (1, 1, 1), (padding, padding, padding), groups=groups)
class Attention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        # NOTE scale factor was wrong in my original version, can set manually to be compat with prev weights
        self.scale = qk_scale or head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]   # make torchscript happy (cannot use tensor as tuple)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x

class adaptation(nn.Module):
    def __init__(self, in_channels=320, patch_size=16,act_layer=nn.ReLU6, drop=0., T_multiple=2):
        super(adaptation, self).__init__()
        out_channels = 3
        self.patch_size = patch_size
        self.T_multiple = T_multiple
        kernel_size = (T_multiple, patch_size, patch_size)
        stride = (T_multiple, patch_size, patch_size)
        self.transconv = nn.ConvTranspose3d(in_channels, out_channels, kernel_size, stride, bias=True)
        self.act = act_layer()
        self.drop = nn.Dropout(drop)
        self.norm = nn.BatchNorm3d(in_channels, affine=True)
        #self.norm = nn.GroupNorm(1, in_channels)
    def forward(self, x):
        in_shape = x.shape #[B, C=320, T=16, H=14, W]
        assert len(in_shape) == 5 and in_shape[1] == 320
        #print(self.norm(x))
        #print(self.transconv.weight)
        x = self.drop(self.act(self.transconv(self.norm(x)))) #[B, 3, T*T_multiple, H*patch_size, W*patch_size]
        return x

class DensePatchEmbedding(nn.Module):
    def __init__(self, img_size=224, dim=768, patch_size=16, stride=4, pos_kernel=3,dense=True, T_multiple=2):
        super(DensePatchEmbedding, self).__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.stride = stride
        self.dense = dense
        if dense:
            self.proj = conv_3xnxn(inp=3, oup=dim, kernel_size=patch_size, stride=stride, T_multiple=T_multiple)
        else:
            self.proj = conv_3xnxn(inp=3, oup=dim, kernel_size=patch_size, stride=patch_size, T_multiple=T_multiple)
        self.norm = nn.BatchNorm3d(3)
        if pos_kernel==3:
            self.pos_embed = conv_3x3x3(inp=dim, oup=dim, groups=dim)
        else:
            self.pos_embed = conv_nxnxn(inp=dim, oup=dim, kernel_size=pos_kernel, groups=dim)
    def forward(self, x):  #[B, C=3, T, H, W]
        assert x.shape[1] == 3 and x.shape[-1] == self.img_size
        x = self.norm(x)
        x = self.proj(x)   # dense: H,W: 224->53
        B, C, T, H, W = x.shape
        if self.dense:
            assert H == (self.img_size-self.patch_size)//self.stride + 1  
        x = x.flatten(2).transpose(1, 2)
        x = x.reshape(B, T, H, W, -1).permute(0, 4, 1, 2, 3).contiguous() #[B, C=768, T, H, W]
        p = self.pos_embed(x)
        assert p.shape==x.shape
        return x, p
def max_activation(x,dim=1): 
    assert len(x.shape)==3   
    mask = (x == x.max(dim=dim, keepdim=True)[0]).to(dtype=x.dtype)
    #result = torch.mul(mask, x)
    #result = torch.masked_select(x, mask)
    return mask

class tokenselector(nn.Module):
    def __init__(self, dim=768, num_tokens=32):
        super(tokenselector, self).__init__()
        self.dim = dim
        self.num_tokens = num_tokens
        #self.select = nn.Conv2d(in_channels=dim, out_channels=num_tokens, kernel_size=3, stride=1, padding=1)
        self.norm = nn.BatchNorm2d(dim)
        self.select = nn.Sequential(
                                nn.Conv2d(in_channels=dim, out_channels=dim, kernel_size=3, stride=1, padding=1,groups=dim),
                                nn.Hardswish(),
                                nn.Conv2d(in_channels=dim, out_channels=num_tokens, kernel_size=1, stride=1)
                                )
    def forward(self, x, pos):
        f = x[:, :, 0, :, :]  #f:[B C H W]
        B, C, H, W = f.shape
        f = self.norm(f)
        select_token = self.select(f)
        select_token = select_token.reshape(B, -1, H*W)  #[B, num_tokens, H*W]
        assert select_token.shape == (B, self.num_tokens, H*W)
        mask = max_activation(select_token, dim=2)  #[B, num_tokens, H*W]
        #assert (torch.sum(mask, dim=2).max() == 1), torch.sum(mask, dim=2).max()
        feat = rearrange(f, 'b c h w -> b (h w) c', c=self.dim)
        result = mask @ feat  #[B, num_tokens, C]
        p = rearrange(pos[:, :, 0, :, :], 'b c h w -> b (h w) c', c=self.dim)  #[B, H*W, C]
        p = mask @ p
        return result, p, mask

class chain(nn.Module):
    def __init__(self, dim=768, num_tokens=32, dense=True):
        super(chain, self).__init__()
        self.dim = dim
        self.num_tokens = num_tokens
        self.patch_tokens = 53**2 if dense else 14**2
        self.proj = Mlp(in_features=self.patch_tokens+num_tokens, out_features=self.patch_tokens)
        self.norm = nn.BatchNorm3d(dim)

    def forward(self, tem, tem_pos, x, pos):
        B, C, T, H, W = x.shape
        x = self.norm(x)
        assert tem.shape == (B, self.num_tokens, C) and C == self.dim and self.patch_tokens == H*W
        q = tem.unsqueeze(1).repeat(1, T-1, 1, 1) # B T-1 num_tokens dim
        q = q.reshape(B*(T-1), self.num_tokens, self.dim)
        f = x[:, :, 1:, :, :]  #f:[B C T-1 H W]
        p = pos[:, :, 1:, :, :]  #p:[B C T-1 H W]
        f = rearrange(f, 'b c t h w -> (b t) (h w) c', c=self.dim, t=T-1)
        p = rearrange(p, 'b c t h w -> (b t) (h w) c', c=self.dim, t=T-1)
        k = torch.cat([q, f], dim=1)  #[B*(T-1) num_tokens+H*W C]
        k = self.proj(k.transpose(1,2)) #+ f.transpose(1,2)  #[B*(T-1) C H*W]
        score = q @ k  #[B*(T-1) num_tokens, H*W]
        score = max_activation(score, dim=2)
        newf = score @ f  #[B*(T-1) num_tokens, C]
        newf = newf.reshape(B, T-1, self.num_tokens, self.dim)
        tem = tem.unsqueeze(1)
        out = torch.cat([tem, newf], dim=1)
        newp = score @ p  #[B*(T-1) num_tokens, C]
        newp = newp.reshape(B, T-1, self.num_tokens, self.dim)
        tem_pos = tem_pos.unsqueeze(1)
        out_pos = torch.cat([tem_pos, newp], dim=1)
        return out, out_pos #[B, T, num_tokens, dim]

def similar_loss(out):
    B, T, num_tokens, dim = out.shape
    out = out.transpose(1, 2)  #[B, num_tokens, T, dim]
    tem = out[:, :, 0, :].unsqueze(2).repeat(1,1,T,1)
    res = torch.abs(out-tem).flatten(start_dim=1) #[B, -1]
    loss = res.mean()
    return loss  #float32

class STFM(nn.Module):
    def __init__(self, img_size=224, patch_size=16, dim=768, num_tokens=32, input_channel=320,
                 dense=True, pos_kernel=3, add_loss=True, T_multiple=2):
        super(STFM, self).__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.dim = dim
        self.num_tokens = num_tokens
        self.input_channel = input_channel
        self.add_loss = add_loss
        self.adapt = adaptation(in_channels=input_channel, patch_size=patch_size, T_multiple=T_multiple)
        self.patch_embed = DensePatchEmbedding(img_size=img_size, dim=dim, patch_size=patch_size, stride=4,
                                                 pos_kernel=pos_kernel,dense=dense, T_multiple=T_multiple)
        self.tokenselector = tokenselector(dim=dim, num_tokens=num_tokens)
        self.chain = chain(dim=dim, num_tokens=num_tokens, dense=dense)
        #self.norm = nn.BatchNorm2d(dim)
    def sloss(self, out):
        if self.add_loss:
            B, T, num_tokens, dim = out.shape
            out = out.transpose(1, 2)  #[B, num_tokens, T, dim]
            tem = out[:, :, 0, :].unsqueeze(2).repeat(1,1,T,1)
            res = torch.abs(out-tem).flatten(start_dim=1) #[B, -1]
            loss = res.mean()
            return loss
        else:
            return torch.tensor(0.0)
    def forward(self, x, ori_img):
        assert x.shape[1] == self.input_channel
        x = self.adapt(x) + ori_img  #[B, C=3, T*T_multiple, H=224, W]
        x, pos = self.patch_embed(x)  #[B, dim, T, H, W]
        #x = self.norm(x)
        tem, tem_pos, mask = self.tokenselector(x, pos)  #[B, num_tokens, C]
        out, out_pos = self.chain(tem, tem_pos, x, pos)  #[B, T, num_tokens, C]
        return out, out_pos

class CFBlock(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(
            dim,
            num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale,
            attn_drop=attn_drop, proj_drop=drop)
        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

    def forward(self, x, pos):
        x = x + pos
        B, T, N, C = x.shape
        x = rearrange(x, 'b t n c -> b (t n) c', t=T)
        x = x + self.drop_path(self.attn(self.norm1(x)))
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        x = x.reshape(B, T, N, C)
        return x 










