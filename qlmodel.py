import torch
import math
import torch.nn as nn
# from spikingjelly.clock_driven.neuron import MultiStepLIFNode
from spikingjelly.activation_based import neuron
from spikingjelly.clock_driven import layer
from timm.models.layers import to_2tuple, trunc_normal_, DropPath
from timm.models.registry import register_model
from timm.models.vision_transformer import _cfg
from einops.layers.torch import Rearrange
import torch.nn.functional as F
from functools import partial
from timm.models import create_model

__all__ = ['Spikingformer']

# neuron.qlLIFNode(step_mode='m', tau=2.0, detach_reset=True, backend='cupy')

tau_thr = 2.0


class MLP(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.mlp1_lif = neuron.qlLIFNode(step_mode='m', tau=2.0, detach_reset=True, backend='cupy')
        self.mlp1_conv = nn.Conv2d(in_features, hidden_features, kernel_size=1, stride=1)
        self.mlp1_bn = nn.BatchNorm2d(hidden_features)

        self.mlp2_lif = neuron.qlLIFNode(step_mode='m', tau=2.0, detach_reset=True, backend='cupy')
        self.mlp2_conv = nn.Conv2d(hidden_features, out_features, kernel_size=1, stride=1)
        self.mlp2_bn = nn.BatchNorm2d(out_features)

        self.c_hidden = hidden_features
        self.c_output = out_features

    def forward(self, x, alpha):
        T, B, C, H, W = x.shape
        # T, B, C, N = x.shape

        x = self.mlp1_lif(x, alpha)
        x = self.mlp1_conv(x.flatten(0, 1))
        x = self.mlp1_bn(x).reshape(T, B, self.c_hidden, H, W)

        x = self.mlp2_lif(x, alpha)
        x = self.mlp2_conv(x.flatten(0, 1))
        x = self.mlp2_bn(x).reshape(T, B, C, H, W)
        return x


class SpikingHartleyAttention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0., sr_ratio=1):
        super().__init__()
        assert dim % num_heads == 0, f"dim {dim} should be divided by num_heads {num_heads}."
        self.hidden_size = dim
        self.num_blocks = num_heads
        self.block_size = self.hidden_size // self.num_blocks
        self.hidden_size_factor = 1
        self.scale = 0.002

        #定义可学习参数
        self.w1 = nn.Parameter(
            self.scale * torch.randn(1, self.num_blocks, self.block_size, self.block_size * self.hidden_size_factor))
        # self.w2 = nn.Parameter(
        #     self.scale * torch.randn(1, self.num_blocks, self.block_size, self.block_size * self.hidden_size_factor))
        self.act = neuron.qlLIFNode(step_mode='m', tau=tau_thr, v_threshold=0.5, detach_reset=True, backend='cupy')
        # self.act01_r = neuron.qlLIFNode(step_mode='m', tau=tau_thr, v_threshold=0.5, detach_reset=True, backend='cupy')
        # self.act01_i = neuron.qlLIFNode(step_mode='m', tau=tau_thr, v_threshold=0.5, detach_reset=True, backend='cupy')
        self.act11 = neuron.qlLIFNode(step_mode='m', tau=tau_thr, v_threshold=0.5, detach_reset=True, backend='cupy')
        # ------add---------
        # self.act02_r = neuron.qlLIFNode(step_mode='m', tau=tau_thr, v_threshold=0.5, detach_reset=True, backend='cupy')
        self.act12 = neuron.qlLIFNode(step_mode='m', tau=tau_thr, v_threshold=0.5, detach_reset=True, backend='cupy')
        # self.act31 = neuron.qlLIFNode(step_mode='m', tau=tau_thr, v_threshold=0.5, detach_reset=True, backend='cupy')
        # self.act32 = neuron.qlLIFNode(step_mode='m', tau=tau_thr, v_threshold=0.5, detach_reset=True, backend='cupy')
        # self.act41 = neuron.qlLIFNode(step_mode='m', tau=tau_thr, v_threshold=0.5, detach_reset=True, backend='cupy')
        # self.act42 = neuron.qlLIFNode(step_mode='m', tau=tau_thr, v_threshold=0.5, detach_reset=True, backend='cupy')
        # ------------------
        self.bn1_1 = nn.BatchNorm2d(dim * self.hidden_size_factor)
        self.bn1_2 = nn.BatchNorm2d(dim * self.hidden_size_factor)
        self.bn2_1 = nn.BatchNorm2d(dim)
        self.bn2_2 = nn.BatchNorm2d(dim)
        self.act21 = neuron.qlLIFNode(step_mode='m', tau=tau_thr, v_threshold=0.5, detach_reset=True, backend='cupy')
        self.act22 = neuron.qlLIFNode(step_mode='m', tau=tau_thr, v_threshold=0.5, detach_reset=True, backend='cupy')
        self.bn3 = nn.BatchNorm2d(dim)

    def hartley_transform(self, x):
        return  torch.fft.fft2((x), dim = (3, 4), norm="ortho")

    def inverse_hartley_transform(self, x, H, W):
        # 由于Hartley变换是自逆的，我们可以直接再次应用Hartley变换来得到逆变换
        return self.hartley_transform(x)

    def forward(self, x, alpha):
        T, B, C, H, W = x.shape
        dtype = x.dtype
        x = x.float()
        x = self.act(x, alpha)
        x = self.hartley_transform(x)

        # origin_ffted_fp = x.real - x.imag
        # origin_ffted_fn = x.imag + x.imag
        # origin_ffted_sp = self.act01_r(x.real - x.imag, alpha)
        # origin_ffted_sn = self.act01_i(x.imag + x.imag, alpha)

        # origin_ffted_fo = x.real
        # origin_ffted_fe = x.imag
        origin_ffted_so = self.act21(x.real, alpha)
        origin_ffted_se = self.act22(x.imag, alpha)

        x.real = origin_ffted_so
        x.imag = origin_ffted_se
        x = x.reshape(T * B, self.num_blocks, self.block_size, x.shape[3], x.shape[4])
        o1_real = self.act11(
            self.bn1_1((torch.einsum('bkihw,kio->bkohw', x.real, self.w1[0]) - \
                        torch.einsum('bkihw,kio->bkohw', x.imag, self.w1[0])).flatten(1, 2)).reshape(T, B,self.num_blocks,self.block_size * self.hidden_size_factor,x.shape[3],x.shape[4])
        , alpha).flatten(0, 1)
        o1_imag = self.act12(
            self.bn1_2((torch.einsum('bkihw,kio->bkohw', x.imag, self.w1[0]) + \
                        torch.einsum('bkihw,kio->bkohw', x.real, self.w1[0])).flatten(1, 2)).reshape(T, B,self.num_blocks,self.block_size * self.hidden_size_factor,
                                                                                                                      x.shape[3],x.shape[4]), alpha).flatten(0, 1)
        o2_real = (self.bn2_1((torch.einsum('bkihw,kio->bkohw', o1_real, self.w1[0]) - \
                               torch.einsum('bkihw,kio->bkohw', o1_imag, self.w1[0])).flatten(1, 2))
                             .reshape(T, B,self.num_blocks,self.block_size,x.shape[3],x.shape[4]))
        o2_imag = (self.bn2_2((torch.einsum('bkihw,kio->bkohw', o1_imag, self.w1[0]) + \
                               torch.einsum('bkihw,kio->bkohw', o1_real, self.w1[0])).flatten(1, 2))
                             .reshape(T, B,self.num_blocks,self.block_size,x.shape[3],x.shape[4]))
        # o1_real = self.act11(
        #     self.bn1_1((torch.einsum('bkihw,kio->bkohw', x.real, self.w1[0]) ).flatten(1, 2)).reshape(T, B,self.num_blocks,self.block_size * self.hidden_size_factor,x.shape[3],x.shape[4])
        # , alpha).flatten(0, 1)
        # o1_imag = self.act12(
        #     self.bn1_2((torch.einsum('bkihw,kio->bkohw', x.imag, self.w1[0])).flatten(1, 2)).reshape(T, B,self.num_blocks,self.block_size * self.hidden_size_factor,
        #                                                                                                               x.shape[3],x.shape[4]), alpha).flatten(0, 1)
        # o2_real = (self.bn2_1((torch.einsum('bkihw,kio->bkohw', o1_real, self.w1[0]) ).flatten(1, 2))
        #                      .reshape(T, B,self.num_blocks,self.block_size,x.shape[3],x.shape[4]))
        # o2_imag = (self.bn2_2((torch.einsum('bkihw,kio->bkohw', o1_imag, self.w1[0]) ).flatten(1, 2))
        #                      .reshape(T, B,self.num_blocks,self.block_size,x.shape[3],x.shape[4]))
        o2_real = o2_real.reshape(T, B, C, o2_real.shape[4], o2_real.shape[5])
        o2_imag = o2_imag.reshape(T, B, C, o2_imag.shape[4], o2_imag.shape[5])

        o2_fp = o2_real - o2_imag
        o2_fn = o2_real + o2_imag
        # o2_sp = self.act41(o2_real - o2_imag, alpha)
        # o2_sn = self.act42(o2_real + o2_imag, alpha)

        # o2_fo = o2_real
        # o2_fe = o2_imag
        # o2_so = self.act31(o2_real, alpha)
        # o2_se = self.act32(o2_imag, alpha)

        # x1 = o2_real * origin_ffted_p - (o2_imag * origin_ffted_n)
        # x = (((o2_so * origin_ffted_fp) - (o2_se * origin_ffted_fn)) + ((origin_ffted_so * o2_fp) - (origin_ffted_se * o2_fn))
        #      + ((o2_fo * origin_ffted_sp) - (o2_fe * origin_ffted_sn)) + ((origin_ffted_fo * o2_sp) - (origin_ffted_fe * o2_sn)))
        # x = (o2_fo * origin_ffted_sp) - (o2_fe * origin_ffted_sn)  #96.19  act 0.5
        x = ((origin_ffted_so * o2_fp) - (origin_ffted_se * o2_fn))  #  act 0.5

        x = F.softshrink(x, 0.02)

        x = self.hartley_transform(x)
        x = x.type(dtype)
        x = self.bn3(x.flatten(0, 1)).reshape(T, B, C, H, W)

        return x
class SpikingSelfAttention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0., sr_ratio=1):
        super().__init__()
        assert dim % num_heads == 0, f"dim {dim} should be divided by num_heads {num_heads}."

        self.dim = dim
        self.num_heads = num_heads

        self.proj_lif = neuron.qlLIFNode(step_mode='m', tau=2.0, detach_reset=True, backend='cupy')
        self.q_conv = nn.Conv1d(dim, dim, kernel_size=1, stride=1, bias=False)
        self.q_bn = nn.BatchNorm1d(dim)

        self.q_lif = neuron.qlLIFNode(step_mode='m', tau=2.0, detach_reset=True, backend='cupy')
        self.k_conv = nn.Conv1d(dim, dim, kernel_size=1, stride=1, bias=False)
        self.k_bn = nn.BatchNorm1d(dim)

        self.k_lif = neuron.qlLIFNode(step_mode='m', tau=2.0, detach_reset=True, backend='cupy')
        self.v_conv = nn.Conv1d(dim, dim, kernel_size=1, stride=1, bias=False)
        self.v_bn = nn.BatchNorm1d(dim)
        self.v_lif = neuron.qlLIFNode(step_mode='m', tau=2.0, detach_reset=True, backend='cupy')

        self.attn_lif = neuron.qlLIFNode(step_mode='m', tau=2.0, detach_reset=True, backend='cupy')
        self.proj_conv = nn.Conv1d(dim, dim, kernel_size=1, stride=1)
        self.proj_bn = nn.BatchNorm1d(dim)


    def forward(self, x, alpha):
        T, B, C, H, W = x.shape
        x = self.proj_lif(x, alpha)

        x = x.flatten(3)
        T, B, C, N = x.shape
        x_for_qkv = x.flatten(0, 1)

        q_conv_out = self.q_conv(x_for_qkv)
        q_conv_out = self.q_bn(q_conv_out).reshape(T, B, C, N)
        q_conv_out = self.q_lif(q_conv_out, alpha)
        q = q_conv_out.transpose(-1, -2).reshape(T, B, N, self.num_heads, C // self.num_heads).permute(0, 1, 3, 2, 4).contiguous()

        k_conv_out = self.k_conv(x_for_qkv)
        k_conv_out = self.k_bn(k_conv_out).reshape(T, B, C, N)
        k_conv_out = self.k_lif(k_conv_out, alpha)
        k = k_conv_out.transpose(-1, -2).reshape(T, B, N, self.num_heads, C // self.num_heads).permute(0, 1, 3, 2, 4).contiguous()

        v_conv_out = self.v_conv(x_for_qkv)
        v_conv_out = self.v_bn(v_conv_out).reshape(T, B, C, N)
        v_conv_out = self.v_lif(v_conv_out, alpha)
        v = v_conv_out.transpose(-1, -2).reshape(T, B, N, self.num_heads, C // self.num_heads).permute(0, 1, 3, 2, 4).contiguous()

        attn = (q @ k.transpose(-2, -1))
        x = (attn @ v) * 0.125

        x = x.transpose(3, 4).reshape(T, B, C, N)
        x = self.attn_lif(x, alpha)
        x = x.flatten(0, 1)
        x = self.proj_bn(self.proj_conv(x)).reshape(T, B, C, H, W)
        return x

class SpikingTransformer(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., norm_layer=nn.LayerNorm, sr_ratio=1):
        super().__init__()
        self.attn = SpikingHartleyAttention(dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale,
                                            attn_drop=attn_drop, proj_drop=drop, sr_ratio=sr_ratio)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = MLP(in_features=dim, hidden_features=mlp_hidden_dim, drop=drop)

    def forward(self, x, alpha):
        x = x + self.attn(x, alpha)
        x = x + self.mlp(x, alpha)

        return x

class Top_Down(nn.Module):
    def __init__(self, in_channel_list, out_channel):
        super().__init__()
        self.MSSF = nn.ModuleList([nn.Sequential(nn.Conv2d(in_ch, out_channel, kernel_size=1, bias=False),
                                                 nn.BatchNorm2d(out_channel)) for in_ch in in_channel_list])
        self.MSTF_lif = nn.ModuleList(
            [neuron.qlLIFNode(step_mode='m', tau=2.0, v_threshold=1.0, detach_reset=True, backend='cupy')
             for _ in in_channel_list])
        self.MSTF = nn.ModuleList([nn.Sequential(nn.Conv1d(out_channel, out_channel, kernel_size=1, bias=False),
                                                 nn.BatchNorm1d(out_channel)) for _ in in_channel_list])
        self.mp1 = torch.nn.MaxPool2d(kernel_size=3, stride=2, padding=1, dilation=1)
        self.mp2 = torch.nn.MaxPool2d(kernel_size=5, stride=4, padding=1, dilation=1)

    def forward(self, x, alpha):
        head_output = []
        T, B, C, H, W = x[-1].shape
        add_pre2corent = self.MSSF[-1](x[-1].flatten(0, 1))

        tempro_fusion = self.MSTF_lif[-1](add_pre2corent.reshape(T, B, C, H, W), alpha)
        tempro_fusion = tempro_fusion.permute(3, 4, 1, 2, 0).contiguous()
        tempro_fusion = self.MSTF[-1](tempro_fusion.flatten(0, 2))
        # tempro_fusion = self.MSTF[-1](tempro_fusion.permute(3, 4, 1, 2, 0).flatten(0, 2)).reshape(H, W, B, C, T).permute(4, 2, 3, 0, 1)
        head_output.append((tempro_fusion))
        for i in range(len(x) - 2, -1, -1):
            corent_inner = self.MSSF[i](x[i].flatten(0, 1))
            if corent_inner.shape[2:] != add_pre2corent.shape[2:]:
                add_pre2corent = F.interpolate(add_pre2corent, size=corent_inner.shape[2:])
            add_pre2corent = (add_pre2corent + corent_inner)
            tempro_fusion = add_pre2corent
            if i == 1:
                tempro_fusion = ((self.mp1(tempro_fusion)))
            elif i == 0:
                tempro_fusion = ((self.mp2(tempro_fusion)))
            tempro_fusion = self.MSTF_lif[i](tempro_fusion.reshape(T, B, C, H, W), alpha)
            tempro_fusion = tempro_fusion.permute(3, 4, 1, 2, 0).contiguous()
            tempro_fusion = self.MSTF[i](tempro_fusion.flatten(0, 2))
            head_output.append(tempro_fusion)
        x = torch.stack((list((head_output))), dim=0).reshape(-1, H, W, B, C, T)
        x = x.permute(0, 5, 3, 4, 1, 2).contiguous()
        x = torch.mean(x, dim=0)
        return x


class SpikingTokenizer(nn.Module):
    def __init__(self, img_size_h=128, img_size_w=128, patch_size=4, in_channels=2, embed_dims=256):
        super().__init__()
        self.image_size = [img_size_h, img_size_w]
        patch_size = to_2tuple(patch_size)
        self.patch_size = patch_size
        self.C = in_channels
        self.H, self.W = self.image_size[0] // patch_size[0], self.image_size[1] // patch_size[1]
        self.num_patches = self.H * self.W

        # self.Down_top = Down_top(in_channels, embed_dims // 8, pool=False)
        self.proj_conv = nn.Conv2d(in_channels, embed_dims // 8, kernel_size=3, stride=1, padding=1, bias=False)
        self.proj_bn = nn.BatchNorm2d(embed_dims // 8)
        # self.Down_top1 = Down1_top2_V2(inchannel=48, outchannel=96, layers=1)

        # self.Down1_top = Down_top(embed_dims // 8, embed_dims // 4, pool=False)
        self.proj1_lif = neuron.qlLIFNode(step_mode='m', tau=2.0, detach_reset=True, backend='cupy')
        self.proj1_conv = nn.Conv2d(embed_dims // 8, embed_dims // 4, kernel_size=3, stride=1, padding=1, bias=False)
        self.proj1_bn = nn.BatchNorm2d(embed_dims // 4)
        # self.Down_top2 = Down1_top2_V2(inchannel=96, outchannel=96, layers=2)

        # self.Down2_top = Down_top(embed_dims // 4, embed_dims // 2, pool=True)
        self.proj2_lif = neuron.qlLIFNode(step_mode='m', tau=2.0, detach_reset=True, backend='cupy')
        self.proj2_conv = nn.Conv2d(embed_dims // 4, embed_dims // 2, kernel_size=3, stride=1, padding=1, bias=False)
        self.proj2_bn = nn.BatchNorm2d(embed_dims // 2)
        self.proj2_mp = torch.nn.MaxPool2d(kernel_size=3, stride=2, padding=1, dilation=1, ceil_mode=False)

        # self.Down_top3 = Down1_top2_V2(inchannel=192, outchannel=192, layers=3)

        # self.Down3_top = Down_top(embed_dims // 2, embed_dims // 1, pool=True)
        self.proj3_lif = neuron.qlLIFNode(step_mode='m', tau=2.0, detach_reset=True, backend='cupy')
        self.proj3_conv = nn.Conv2d(embed_dims // 2, embed_dims // 1, kernel_size=3, stride=1, padding=1, bias=False)
        self.proj3_bn = nn.BatchNorm2d(embed_dims // 1)
        self.proj3_mp = torch.nn.MaxPool2d(kernel_size=3, stride=2, padding=1, dilation=1, ceil_mode=False)

        # self.Down4_top = Down_top(embed_dims , embed_dims , pool=True)
        self.proj4_lif = neuron.qlLIFNode(step_mode='m', tau=2.0, detach_reset=True, backend='cupy')
        self.proj4_conv = nn.Conv2d(embed_dims, embed_dims, kernel_size=3, stride=1, padding=1, bias=False)
        self.proj4_bn = nn.BatchNorm2d(embed_dims)
        self.proj4_mp = torch.nn.MaxPool2d(kernel_size=3, stride=2, padding=1, dilation=1, ceil_mode=False)
        self.proj5_lif = neuron.qlLIFNode(step_mode='m', tau=2.0, detach_reset=True, backend='cupy')

        self.Top_down = Top_Down([96, 192, 384, 384], 384)

    def forward(self, x, alpha):
        T, B, C, H, W = x.shape
        FPN_input = []
        x = self.proj_conv(x.flatten(0, 1))
        x = self.proj_bn(x).reshape(T, B, -1, H, W)

        x = self.proj1_lif(x, alpha).flatten(0, 1)
        x = self.proj1_conv(x)
        x = self.proj1_bn(x).reshape(T, B, -1, H, W)
        # FPN_input.append(x)

        x = self.proj2_lif(x, alpha)
        FPN_input.append(x)
        x = self.proj2_conv(x.flatten(0, 1))
        x = self.proj2_bn(x)
        x = (self.proj2_mp(x)).reshape(T, B, -1, int(H / 2), int(W / 2))
        # FPN_input.append(x)

        x = self.proj3_lif(x, alpha)
        FPN_input.append(x)
        x = self.proj3_conv(x.flatten(0, 1))
        x = self.proj3_bn(x)
        x = (self.proj3_mp(x)).reshape(T, B, -1, int(H / 4), int(W / 4))
        # FPN_input.append(x)

        x = self.proj4_lif(x, alpha)
        FPN_input.append(x)
        x = self.proj4_conv(x.flatten(0, 1))
        x = self.proj4_bn(x).reshape(T, B, -1, int(H / 4), int(W / 4))
        x = self.proj5_lif(x, alpha)
        FPN_input.append(x)

        x = self.Top_down(FPN_input, alpha)
        H, W = H // self.patch_size[0], W // self.patch_size[1]
        return x, (H, W)


class vit_snn(nn.Module):
    def __init__(self,
                 img_size_h=128, img_size_w=128, patch_size=16, in_channels=2, num_classes=11,
                 embed_dims=[64, 128, 256], num_heads=[1, 2, 4], mlp_ratios=[4, 4, 4], qkv_bias=False, qk_scale=None,
                 drop_rate=0., attn_drop_rate=0., drop_path_rate=0., norm_layer=nn.LayerNorm,
                 depths=[6, 8, 6], sr_ratios=[8, 4, 2], T=4, pretrained_cfg=None,
                 ):
        super().__init__()

        self.iter_num = 0

        self.step_size = 196
        self.STH_epoch = 101
        self.init_scale = 1.0
        self.alpha = self.init_scale


        self.num_classes = num_classes
        self.depths = depths
        self.T = T
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depths)]  # stochastic depth decay rule

        patch_embed = SpikingTokenizer(img_size_h=img_size_h,
                                       img_size_w=img_size_w,
                                       patch_size=patch_size,
                                       in_channels=in_channels,
                                       embed_dims=embed_dims)
        num_patches = patch_embed.num_patches

        block = nn.ModuleList([SpikingTransformer(
            dim=embed_dims, num_heads=num_heads, mlp_ratio=mlp_ratios, qkv_bias=qkv_bias,
            qk_scale=qk_scale, drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[j],
            norm_layer=norm_layer, sr_ratio=sr_ratios)
            for j in range(depths)])

        setattr(self, f"patch_embed", patch_embed)
        setattr(self, f"block", block)

        # classification head
        self.head = nn.Linear(embed_dims, num_classes) if num_classes > 0 else nn.Identity()
        self.apply(self._init_weights)

    def _init_weights(self, m):

        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward_features(self, x):
        block = getattr(self, f"block")

        patch_embed = getattr(self, f"patch_embed")

        x, (H, W) = patch_embed(x, self.alpha)
        # x, (H, W) = patch_embed(x, )
        for blk in block:
            x = blk(x, self.alpha)
            # x = blk(x,)
        return x.flatten(3).mean(3)
#80 / 130 81.01
#102 / 195 81.17
# 102 / 195 5e-4 79.75
# 102 / 195 5e-3 300epoch 81.07
#90 / 196 5e-3 400 epoch 81.15
#101 / 196 5e-3 400 epoch 81.46
#101 / 195 5e-3 400 epoch  nums-head=4  81.04
# 101 / 195 5e-3 400 epoch  nums-head=8  81.24
    def forward(self, x):
        if self.training:
            self.iter_num += 1
            if self.iter_num % self.step_size == 0:

                if (1.0 - self.iter_num / (self.STH_epoch * 390.0)) > 0.0:
                    # self.alpha = 0.0
                    self.alpha = math.pow((1.0 - self.iter_num / (self.STH_epoch * 390.0)), 2.0)
                    print(self.iter_num)
                    print(self.alpha)
                else:
                    self.alpha = 0.0
        x = (x.unsqueeze(0)).repeat(self.T, 1, 1, 1, 1)
        # x = (x.unsqueeze(0)).repeat(1, 1, 1, 1, 1)
        x = self.forward_features(x)
        x = self.head(x.mean(0))
        return x


@register_model
def Spikingformer(pretrained=False, **kwargs):
    model = vit_snn(
        **kwargs
    )
    model.default_cfg = _cfg()
    return model

# 50 96.37


