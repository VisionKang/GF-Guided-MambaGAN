import torch
from torch import nn
import numpy as np
from torch.nn import LayerNorm
from timm.models.layers import DropPath, to_2tuple, trunc_normal_
from mamba_simple import *
from mamba_module import *
from mamba_ssm.modules.mamba_simple import Mamba


class Basic1x1(nn.Module):
    def __init__(self, inplanes, planes, stride=1):
        super(Basic1x1, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(16, 1, kernel_size=1, padding=0, stride=stride),
            nn.Tanh())
        # nn.Conv2d(8, 1, kernel_size=1, padding=0, stride=stride))

    def forward(self, x):
        out = self.conv1(x)
        return out


class TransformerBlock(nn.Module):
    def __init__(self, dim, num_heads=8, mlp_ratio=4., qkv_bias=False, drop=0., attn_drop=0.):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(
            embed_dim=dim,
            num_heads=num_heads,
            dropout=attn_drop,
            bias=qkv_bias,
            batch_first=True
        )
        self.drop_path = DropPath(drop) if drop > 0. else nn.Identity()

        self.norm2 = nn.LayerNorm(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, drop=drop)

    def forward(self, x):
        # x: [B, N, C]
        residual = x
        x = self.norm1(x)
        x, _ = self.attn(x, x, x)  # self-attention
        x = residual + self.drop_path(x)

        residual = x
        x = self.norm2(x)
        x = self.mlp(x)
        x = residual + self.drop_path(x)
        return x
class HinResBlock(nn.Module):

    def __init__(self, in_size, out_size, relu_slope=0.2, use_HIN=True):
        super(HinResBlock, self).__init__()
        self.identity = nn.Conv2d(in_size, out_size, 1, 1, 0)

        self.conv_1 = nn.Conv2d(in_size, out_size, kernel_size=3, padding=1, bias=True)
        self.relu_1 = nn.LeakyReLU(relu_slope, inplace=False)
        self.conv_2 = nn.Conv2d(out_size, out_size, kernel_size=3, padding=1, bias=True)
        self.relu_2 = nn.LeakyReLU(relu_slope, inplace=False)
        if use_HIN:
            self.norm = nn.InstanceNorm2d(out_size // 2, affine=True)
        self.use_HIN = use_HIN

    def forward(self, x):
        resi = self.relu_1(self.conv_1(x))
        out_1, out_2 = torch.chunk(resi, 2, dim=1)
        resi = torch.cat([self.norm(out_1), out_2], dim=1)
        resi = self.relu_2(self.conv_2(resi))
        return x + resi


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


def window_partition(x, window_size):
    B, H, W, C = x.shape
    x = x.view(B, H // window_size, window_size, W // window_size, window_size, C)
    windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size, window_size, C)
    return windows


def window_reverse(windows, window_size, H, W):
    B = int(windows.shape[0] / (H * W / window_size / window_size))
    x = windows.view(B, H // window_size, W // window_size, window_size, window_size, -1)
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, H, W, -1)
    return x


class WindowAttention(nn.Module):
    def __init__(self, dim, window_size, num_heads, qkv_bias=True, qk_scale=None, attn_drop=0., proj_drop=0.):

        super().__init__()
        self.dim = dim
        self.window_size = window_size
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        self.relative_position_bias_table = nn.Parameter(
            torch.zeros((2 * window_size[0] - 1) * (2 * window_size[1] - 1), num_heads))

        coords_h = torch.arange(self.window_size[0])
        coords_w = torch.arange(self.window_size[1])
        coords = torch.stack(torch.meshgrid([coords_h, coords_w]))
        coords_flatten = torch.flatten(coords, 1)
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()
        relative_coords[:, :, 0] += self.window_size[0] - 1
        relative_coords[:, :, 1] += self.window_size[1] - 1
        relative_coords[:, :, 0] *= 2 * self.window_size[1] - 1
        relative_position_index = relative_coords.sum(-1)
        self.register_buffer("relative_position_index", relative_position_index)

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        trunc_normal_(self.relative_position_bias_table, std=.02)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x, mask=None):
        B_, N, C = x.shape
        qkv = self.qkv(x).reshape(B_, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        q = q * self.scale
        attn = (q @ k.transpose(-2, -1))

        relative_position_bias = self.relative_position_bias_table[self.relative_position_index.view(-1)].view(
            self.window_size[0] * self.window_size[1], self.window_size[0] * self.window_size[1], -1)
        relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()
        attn = attn + relative_position_bias.unsqueeze(0)

        if mask is not None:
            nW = mask.shape[0]
            attn = attn.view(B_ // nW, nW, self.num_heads, N, N) + mask.unsqueeze(1).unsqueeze(0)
            attn = attn.view(-1, self.num_heads, N, N)
            attn = self.softmax(attn)
        else:
            attn = self.softmax(attn)

        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B_, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


# class StageModule(nn.Module):
#     def __init__(self, dim, input_resolution, depth, num_heads=None, window_size=None,
#                  mlp_ratio=4., qkv_bias=None, qk_scale=None, drop=0., attn_drop=None,
#                  drop_path=0., norm_layer=nn.LayerNorm, use_checkpoint=False):
#         super().__init__()
#         self.dim = dim
#         self.input_resolution = input_resolution
#         self.depth = depth
#         self.use_checkpoint = use_checkpoint
#
#
#         self.blocks = nn.ModuleList([
#             MambaBlock(dim=dim,
#                        drop_path=drop_path[i] if isinstance(drop_path, list) else drop_path,
#                        norm_layer=norm_layer)
#             for i in range(depth)])
#
#     def forward(self, x):
#         for blk in self.blocks:
#             x = blk(x)
#         return x


class MambaBlock(nn.Module):

    def __init__(self, dim, drop_path=0., norm_layer=nn.LayerNorm):
        super().__init__()
        self.norm = norm_layer(dim)
        self.mamba = Mamba(
            d_model=dim,  # 杈撳叆缁村害
            d_state=8,  # 鐘舵€佺淮搴?
            d_conv=2,  # 鍗风Н鏍稿ぇ灏?
            expand=1,  # 鎵╁睍姣斾緥
        )
        self.drop_path = nn.Dropout(drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, x):
        x = x + self.drop_path(self.mamba(self.norm(x)))
        return x


class PatchUnEmbed(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x, x_size):
        B, N, C = x.shape
        H, W = x_size
        x = x.transpose(1, 2).view(B, C, H, W)
        return x


class PatchEmbed(nn.Module):
    def __init__(self, img_size=256, patch_size=4, stride=4, in_chans=96, embed_dim=96, norm_layer=None,
                 flatten=True):
        super().__init__()
        self.patch_size = patch_size
        self.stride = stride
        self.flatten = flatten
        self.embed_dim = embed_dim

        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=stride)
        self.norm = LayerNorm(embed_dim, 'BiasFree') if norm_layer else nn.Identity()

    def forward(self, x):
        B, C, H, W = x.shape
        # x = F.unfold(x, self.patch_size, stride=self.patch_size)
        x = self.proj(x)
        if self.flatten:
            x = x.flatten(2).transpose(1, 2)  # BCHW -> BNC
        # x = self.norm(x)
        return x


class SingleMambaBlock(nn.Module):
    def __init__(self, dim):
        super(SingleMambaBlock, self).__init__()
        self.encoder = Mamba(dim, bimamba_type=None)
        self.norm = LayerNorm(dim, 'with_bias')

    def forward(self, ipt):
        x, residual = ipt
        residual = x + residual
        x = self.norm(residual)
        return (self.encoder(x), residual)


class Base_MODEL(nn.Module):
    def __init__(self, in_chans=2, out_chans=16, patch_size=1, img_size=256,
                 embed_dim=96, depths=[4, 2, 2], drop_rate=0.):
        super().__init__()

        # 1. 输入预处理 (2 → 96)
        self.conv_begin = nn.Sequential(
            nn.Conv2d(in_chans, 24, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(24),
            nn.LeakyReLU(),
            nn.Conv2d(24, 48, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(48),
            nn.LeakyReLU(),
            nn.Conv2d(48, 96, kernel_size=3, stride=1, padding=1),
        )

        # 2. Patch Embedding & Unembedding
        self.ir_to_token = PatchEmbed(in_chans=96, embed_dim=96, patch_size=1, stride=1)
        self.patch_unembed = PatchUnEmbed()

        # ✅ 替换：使用 Transformer Blocks（3 层）
        self.transformer_blocks = nn.Sequential(
            TransformerBlock(dim=96, num_heads=6, drop=drop_rate),
            TransformerBlock(dim=96, num_heads=6, drop=drop_rate),
            TransformerBlock(dim=96, num_heads=6, drop=drop_rate)
        )

        # 3. 输出层 (96 → 16)
        self.conv_end = nn.Sequential(
            nn.Conv2d(96, 48, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(48),
            nn.LeakyReLU(),
            nn.Conv2d(48, 24, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(24),
            nn.LeakyReLU(),
            nn.Conv2d(24, 16, kernel_size=3, stride=1, padding=1)
        )

    def forward(self, x):
        # 输入: (B, 2, H, W)
        x = self.conv_begin(x)  # (B, 96, H, W)
        b, c, h, w = x.shape

        # 转为序列: (B, N, C)
        x_seq = self.ir_to_token(x)  # (B, H*W, 96)

        # ✅ Transformer 处理
        x_seq = self.transformer_blocks(x_seq)  # (B, H*W, 96)

        # 转回图像: (B, 96, H, W)
        x_img = self.patch_unembed(x_seq, (h, w))

        # 输出: (B, 16, H, W)
        out = self.conv_end(x_img)
        return out


class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3):
        super(ResidualBlock, self).__init__()
        padding = int(np.floor(kernel_size / 2))
        self.pad = nn.ReflectionPad2d(padding)
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride=1)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.LeakyReLU(0.2)

        # 如果输入输出通道不同，用 1x1 卷积投影
        if in_channels != out_channels:
            self.proj = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
        else:
            self.proj = None

    def forward(self, x):
        identity = x
        out = self.pad(x)
        out = self.conv(out)
        out = self.bn(out)
        out = self.relu(out)

        if self.proj is not None:
            identity = self.proj(identity)
        out = out + identity
        return out


class Detail_MODEL(nn.Module):
    def __init__(self, in_channel=2, out_channel=16, kernal_size=3):
        super(Detail_MODEL, self).__init__()
        reflection_padding = int(np.floor(kernal_size / 2))
        self.reflection_pad = nn.ReflectionPad2d(reflection_padding)

        # 初始特征提取
        self.Conv = nn.Conv2d(in_channels=in_channel, out_channels=16, kernel_size=kernal_size, stride=1)
        self.BatchNorm2d_1 = nn.BatchNorm2d(16)
        self.LeakyReLU = nn.LeakyReLU(0.2)
        self.up = nn.Upsample(scale_factor=8, mode='bilinear', align_corners=True)

        # 残差细节分支（5 blocks）
        self.res_block1 = ResidualBlock(16, 256)
        self.res_block2 = ResidualBlock(256, 128)
        self.res_block3 = ResidualBlock(128, 39)
        self.res_block4 = ResidualBlock(39, 32)
        self.res_block5 = ResidualBlock(32, 32)  # d_output: 32 channels

        # 语义分支（s_branch）—— 注意：这里 Conv_s4 输出仍为 32
        self.Conv_s1 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=kernal_size, stride=2)
        self.Conv_s2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=kernal_size, stride=2)
        self.Conv_s3 = nn.Conv2d(in_channels=64, out_channels=32, kernel_size=kernal_size, stride=2)
        self.Conv_s4 = nn.Conv2d(in_channels=32, out_channels=32, kernel_size=kernal_size, stride=1)  # ← 改为 32
        self.BatchNorm2d_3 = nn.BatchNorm2d(32)
        self.BatchNorm2d_4 = nn.BatchNorm2d(64)

        # 最终融合：d_output (32) + s_output (32) = 64 → 16
        self.final_conv = nn.Conv2d(64, out_channel, kernel_size=1)  # out_channel=16

    def forward(self, input):
        x = self.reflection_pad(input)
        x = self.Conv(x)
        x = self.BatchNorm2d_1(x)
        x_output = self.LeakyReLU(x)

        # 残差细节分支
        d1 = self.res_block1(x_output)
        d2 = self.res_block2(d1)
        d3 = self.res_block3(d2)
        d4 = self.res_block4(d3)
        d5 = self.res_block5(d4)
        d_output = d5  # (B, 32, H, W)

        # 语义分支
        s = self.reflection_pad(x_output)
        s = self.Conv_s1(s)
        s = self.BatchNorm2d_3(s)
        s = self.LeakyReLU(s)

        s = self.reflection_pad(s)
        s = self.Conv_s2(s)
        s = self.BatchNorm2d_4(s)
        s = self.LeakyReLU(s)

        s = self.reflection_pad(s)
        s = self.Conv_s3(s)
        s = self.BatchNorm2d_3(s)
        s = self.LeakyReLU(s)

        s = self.up(s)
        s = self.reflection_pad(s)
        s = self.Conv_s4(s)
        s = self.BatchNorm2d_3(s)
        s_output = self.LeakyReLU(s)  # (B, 32, H, W)

        # 拼接 + 最终输出 16 通道
        out = torch.cat([d_output, s_output], dim=1)  # (B, 64, H, W)
        out = self.final_conv(out)  # (B, 16, H, W)

        return out


class Fusion_MODEL(nn.Module):
    def __init__(self, in_channel=16, output_channel=1):
        super(Fusion_MODEL, self).__init__()
        self.convolutional_out = Basic1x1(in_channel, output_channel)

    def forward(self, input_1, input_2):
        # print("input_1 shape:", input_1.shape)
        # print("input_2 shape:", input_2.shape)
        add = input_1 + input_2
        out = self.convolutional_out(add)
        return out我现在要做消融实验，我要将基础层的mamba模块都用transform来代替