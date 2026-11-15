import torch
import torch.nn as nn
import math
import numpy as np
from timm.models.layers import DropPath, to_2tuple, trunc_normal_


class Basic1x1(nn.Module):
    """
    一个简单的1x1卷积模块，用于特征维度调整和非线性激活。
    """

    def __init__(self, inplanes, planes, stride=1):
        super(Basic1x1, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(inplanes, planes, kernel_size=1, padding=0, stride=stride),  # 1x1卷积，用于调整通道数
            nn.Tanh())  # Tanh激活函数

    def forward(self, x):
        out = self.conv1(x)
        return out


class Mlp(nn.Module):
    """
    多层感知机模块，用于前馈网络部分。
    """

    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features  # 如果未指定输出特征数，则与输入相同
        hidden_features = hidden_features or in_features  # 如果未指定隐藏层特征数，则与输入相同
        self.fc1 = nn.Linear(in_features, hidden_features)  # 第一个全连接层
        self.act = act_layer()  # 激活函数，默认GELU
        self.fc2 = nn.Linear(hidden_features, out_features)  # 第二个全连接层
        self.drop = nn.Dropout(drop)  # Dropout层

    def forward(self, x):
        x = self.fc1(x)  # 线性变换
        x = self.act(x)  # 激活
        x = self.drop(x)  # Dropout
        x = self.fc2(x)  # 再次线性变换
        x = self.drop(x)  # Dropout
        return x


def window_partition(x, window_size):
    """
    将特征图分割成不重叠的窗口。
    """
    B, H, W, C = x.shape  # 获取批次大小、高度、宽度、通道数
    # 重塑为 (B, H//window_size, window_size, W//window_size, window_size, C)
    x = x.view(B, H // window_size, window_size, W // window_size, window_size, C)
    # 重新排列维度并展平为 (-1, window_size, window_size, C)，其中-1是窗口总数
    windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size, window_size, C)
    return windows


def window_reverse(windows, window_size, H, W):
    """
    将窗口重新组合成完整的特征图。
    """
    # 计算批次大小B
    B = int(windows.shape[0] / (H * W / window_size / window_size))
    # 重塑为 (B, H//window_size, W//window_size, window_size, window_size, -1)
    x = windows.view(B, H // window_size, W // window_size, window_size, window_size, -1)
    # 重新排列维度并展平为 (B, H, W, -1)
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, H, W, -1)
    return x


# Mamba核心组件
class MambaBlock(nn.Module):
    """
    Mamba块实现，包含SSM、MLP和残差连接。
    """

    def __init__(
            self,
            d_model,
            d_state=16,
            d_conv=4,
            expand=2,
            dt_rank="auto",
            dt_min=0.001,
            dt_max=0.1,
            dt_init="random",
            dt_scale=1.0,
            dt_proj_bias=True,
            conv_bias=True,
            bias=False,
            use_fast_path=True,
    ):
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.d_conv = d_conv
        self.expand = expand
        self.d_inner = int(self.expand * self.d_model)

        # 计算dt_rank
        if dt_rank == "auto":
            self.dt_rank = math.ceil(self.d_model / 16)
        else:
            self.dt_rank = dt_rank

        # 输入投影
        self.in_proj = nn.Linear(self.d_model, self.d_inner * 2, bias=bias)
        self.conv1d = nn.Conv1d(
            in_channels=self.d_inner,
            out_channels=self.d_inner,
            kernel_size=d_conv,
            padding=d_conv - 1,
            groups=self.d_inner,
            bias=conv_bias,
        )

        # SSM参数
        self.x_proj = nn.Linear(self.d_inner, self.dt_rank + self.d_state * 2, bias=False)
        self.dt_proj = nn.Linear(self.dt_rank, self.d_inner, bias=dt_proj_bias)

        # 初始化dt_proj权重
        dt_init_std = self.dt_rank ** -0.5 * dt_scale
        if dt_init == "constant":
            nn.init.constant_(self.dt_proj.weight, dt_init_std)
        elif dt_init == "random":
            nn.init.uniform_(self.dt_proj.weight, -dt_init_std, dt_init_std)
        else:
            raise NotImplementedError

        # A和D参数
        A = torch.arange(1, self.d_state + 1, dtype=torch.float32).repeat(self.d_inner, 1)
        self.A_log = nn.Parameter(torch.log(A))
        self.D = nn.Parameter(torch.ones(self.d_inner))

        # 输出投影
        self.out_proj = nn.Linear(self.d_inner, self.d_model, bias=bias)
        self.use_fast_path = use_fast_path

    def forward(self, x):
        """
        x: (B, L, D)
        """
        (b, l, d) = x.shape

        # 输入投影和门控
        x_and_res = self.in_proj(x)  # (b, l, 2 * d_inner)
        (x, res) = x_and_res.split(split_size=[self.d_inner, self.d_inner], dim=-1)

        # 卷积
        x = x.transpose(1, 2)  # (b, d_inner, l)
        x = self.conv1d(x)[..., :l]  # (b, d_inner, l)
        x = x.transpose(1, 2)  # (b, l, d_inner)

        # SSM
        y = self.ssm(x)

        # 残差连接
        y = y * torch.nn.functional.silu(res)

        # 输出投影
        output = self.out_proj(y)
        return output

    def ssm(self, x):
        """
        x: (B, L, D)
        """
        (d_in, n) = self.A_log.shape
        A = -torch.exp(self.A_log.float())  # (d_inner, d_state)
        D = self.D.float()
        x_dbl = self.x_proj(x)  # (b, l, dt_rank + 2 * n)
        (dt, B, C) = x_dbl.split([self.dt_rank, n, n], dim=-1)
        dt = self.dt_proj(dt)  # (b, l, d_inner)

        # SSM的离散化计算
        # 使用选择性扫描
        dt = torch.nn.functional.softplus(dt)
        y = self.selective_scan(x, dt, A, B, C, D)
        return y

    def selective_scan(self, u, delta, A, B, C, D):
        """
        实现选择性扫描，这是Mamba的核心
        这里简化实现，实际中可能需要更复杂的实现或使用专门的CUDA库
        """
        # 为了简化，这里使用一个近似的实现
        # 在实际应用中，应使用专门的Mamba CUDA实现
        batch, l, d_inner = u.shape
        _, _, n = B.shape

        # 初始化状态
        s = torch.zeros(batch, d_inner, n, device=u.device, dtype=u.dtype)
        y = torch.zeros_like(u)

        # 逐时间步计算
        for i in range(l):
            x = u[:, i, :]  # (batch, d_inner)
            delta_i = delta[:, i, :]  # (batch, d_inner)
            A_i = A  # (d_inner, n)
            B_i = B[:, i, :]  # (batch, n)
            C_i = C[:, i, :]  # (batch, n)

            # 计算离散化A和B
            deltaA = torch.exp(A_i * delta_i.unsqueeze(-1))  # (batch, d_inner, n)
            deltaB = B_i.unsqueeze(1) * delta_i.unsqueeze(-1)  # (batch, d_inner, n)

            # 状态更新
            s = deltaA * s + deltaB * x.unsqueeze(-1)

            # 输出计算
            y[:, i, :] = torch.einsum('bin,bn->bi', s, C_i) + D * x

        return y


class MambaTransformerBlock(nn.Module):
    """
    替换SwinTransformerBlock的Mamba块，结合窗口划分和Mamba。
    """

    def __init__(self, dim, input_resolution, window_size=7, shift_size=0,
                 mlp_ratio=4., drop=0., drop_path=0.,
                 norm_layer=nn.LayerNorm, use_checkpoint=False):
        super().__init__()
        self.dim = dim  # 特征维度
        self.input_resolution = input_resolution  # 输入分辨率 (H, W)
        self.window_size = window_size  # 窗口大小
        self.shift_size = shift_size  # 窗口偏移大小
        self.mlp_ratio = mlp_ratio  # MLP隐藏层维度比例
        self.use_checkpoint = use_checkpoint

        # 如果输入分辨率小于窗口大小，则不偏移且窗口大小等于输入分辨率
        if min(self.input_resolution) <= self.window_size:
            self.shift_size = 0
            self.window_size = min(self.input_resolution)
        assert 0 <= self.shift_size < self.window_size, "shift_size must in 0-window_size"

        # Layer Normalization
        self.norm1 = norm_layer(dim)
        # Mamba模块
        self.mamba = MambaBlock(d_model=dim)

        # 随机深度 (Stochastic Depth)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        # MLP模块
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, drop=drop)

        # 生成窗口的掩码 (仅在shift_size > 0时)
        if self.shift_size > 0:
            H, W = self.input_resolution
            img_mask = torch.zeros((1, H, W, 1))
            h_slices = (slice(0, -self.window_size),
                        slice(-self.window_size, -self.shift_size),
                        slice(-self.shift_size, None))
            w_slices = (slice(0, -self.window_size),
                        slice(-self.window_size, -self.shift_size),
                        slice(-self.shift_size, None))
            cnt = 0
            for h in h_slices:
                for w in w_slices:
                    img_mask[:, h, w, :] = cnt
                    cnt += 1

            mask_windows = window_partition(img_mask, self.window_size)
            mask_windows = mask_windows.view(-1, self.window_size * self.window_size)
            attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)
            attn_mask = attn_mask.masked_fill(attn_mask != 0, float(-100.0)).masked_fill(attn_mask == 0, float(0.0))
        else:
            attn_mask = None

        self.register_buffer("attn_mask", attn_mask)  # 注册为buffer，不参与梯度更新

    def forward(self, x):
        H, W = self.input_resolution
        B, L, C = x.shape
        assert L == H * W, "input feature has wrong size"

        shortcut = x  # 残差连接
        x = self.norm1(x)
        x = x.view(B, H, W, C)

        # 如果shift_size > 0，则进行循环移位
        if self.shift_size > 0:
            shifted_x = torch.roll(x, shifts=(-self.shift_size, -self.shift_size), dims=(1, 2))
        else:
            shifted_x = x

        # 将特征图分割成窗口
        x_windows = window_partition(shifted_x, self.window_size)
        x_windows = x_windows.view(-1, self.window_size * self.window_size, C)

        # 在窗口内执行Mamba
        mamba_windows = self.mamba(x_windows)

        # 将窗口重新拼接回特征图
        mamba_windows = mamba_windows.view(-1, self.window_size, self.window_size, C)
        shifted_x = window_reverse(mamba_windows, self.window_size, H, W)

        # 如果之前进行了移位，现在需要移回
        if self.shift_size > 0:
            x = torch.roll(shifted_x, shifts=(self.shift_size, self.shift_size), dims=(1, 2))
        else:
            x = shifted_x
        x = x.view(B, H * W, C)

        # 残差连接 + DropPath
        x = shortcut + self.drop_path(x)
        # MLP + 残差连接 + DropPath
        x = x + self.drop_path(self.mlp(self.norm2(x)))

        return x


class MambaStageModule(nn.Module):
    """
    包含多个MambaTransformerBlock的阶段模块。
    """

    def __init__(self, dim, input_resolution, depth, window_size,
                 mlp_ratio=4., drop=0., drop_path=0., norm_layer=nn.LayerNorm, use_checkpoint=False):
        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.depth = depth  # 块的深度
        self.use_checkpoint = use_checkpoint

        # 创建depth个MambaTransformerBlock
        # 注意：奇数块使用shifted window，偶数块不使用
        self.blocks = nn.ModuleList([
            MambaTransformerBlock(dim=dim, input_resolution=input_resolution,
                                  window_size=window_size,
                                  shift_size=0 if (i % 2 == 0) else window_size // 2,
                                  mlp_ratio=mlp_ratio,
                                  drop=drop,
                                  drop_path=drop_path[i] if isinstance(drop_path, list) else drop_path,
                                  norm_layer=norm_layer)
            for i in range(depth)])

    def forward(self, x):
        for blk in self.blocks:
            x = blk(x)  # 顺序通过每个Mamba块
        return x


class PatchEmbed(nn.Module):
    """
    图像块嵌入层，将图像分割成块并转换为序列。
    """

    def __init__(self, img_size=224, patch_size=1, in_chans=1, embed_dim=96, norm_layer=None):
        super().__init__()
        img_size = to_2tuple(img_size)  # 转换为元组
        patch_size = to_2tuple(patch_size)
        patches_resolution = [img_size[0] // patch_size[0], img_size[1] // patch_size[1]]  # [H/P, W/P]
        self.img_size = img_size
        self.patch_size = patch_size
        self.patches_resolution = patches_resolution
        self.num_patches = patches_resolution[0] * patches_resolution[1]

        self.in_chans = in_chans
        self.embed_dim = embed_dim

        # 使用卷积层代替传统的线性投影
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)
        if norm_layer is not None:
            self.norm = norm_layer(embed_dim)
        else:
            self.norm = None

    def forward(self, x):
        B, C, H, W = x.shape
        # 检查输入尺寸是否与模型定义的尺寸匹配
        assert H == self.img_size[0] and W == self.img_size[1], \
            f"Input image size ({H}*{W}) doesn't match model ({self.img_size[0]}*{self.img_size[1]})."
        x = self.proj(x).flatten(2).transpose(1,
                                              2)  # [B, C, H, W] -> [B, embed_dim, H/P, W/P] -> [B, embed_dim, N] -> [B, N, embed_dim]
        if self.norm is not None:
            x = self.norm(x)
        return x


class Base_MODEL(nn.Module):
    """
    主要的Mamba模型，使用Mamba块代替Swin Transformer块。
    """

    def __init__(self, img_size=224, patch_size=1, in_chans=2, out_chans=64,
                 embed_dim=96, depths=[4, 4, 4],
                 window_size=7, mlp_ratio=4., drop_rate=0., drop_path_rate=0.2,
                 norm_layer=nn.LayerNorm, patch_norm=True,
                 use_checkpoint=False):
        super().__init__()

        print(
            "MambaModel expand initial----depths:{};drop_path_rate:{};out_chans:{}".format(
                depths,
                drop_path_rate, out_chans))

        self.out_chans = out_chans  # 这是patch_embed的输出通道数，也是后续Mamba模块的输入通道数
        self.embed_dim = embed_dim  # 这是另一个通道数参数，通常与out_chans一致或相关
        self.patch_norm = patch_norm
        self.num_features = int(embed_dim)  # 最终输出特征维度
        self.mlp_ratio = mlp_ratio

        # Patch Embedding层，将输入从 in_chans 转换到 embed_dim
        self.patch_embed = PatchEmbed(
            img_size=img_size, patch_size=patch_size, in_chans=in_chans, embed_dim=embed_dim,
            norm_layer=norm_layer if self.patch_norm else None)
        patches_resolution = self.patch_embed.patches_resolution
        self.patches_resolution = patches_resolution

        self.pos_drop = nn.Dropout(p=drop_rate)
        reflection_padding = int(np.floor(3 / 2))
        self.reflection_pad = nn.ReflectionPad2d(reflection_padding)

        # 输入预处理层，将 in_chans (2) -> out_chans (64)
        self.conv_begin = nn.Conv2d(in_chans, out_chans, 3, 1)
        # 输出后处理层，将 embed_dim (96) -> out_chans (64)
        self.conv_end = nn.Conv2d(embed_dim, out_chans, 3, 1)
        self.BatchNorm2d = torch.nn.BatchNorm2d(out_chans)  # 使用out_chans作为BN层的通道数
        self.LeakyReLU = torch.nn.LeakyReLU(0.2)

        # 随机深度率
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]

        # 定义三个Mamba阶段模块 (EN1_0, EN2_0, EN3_0)
        # 每个阶段的输入通道数都是 embed_dim (96)
        self.EN1_0 = MambaStageModule(dim=int(embed_dim),  # Mamba模块输入通道数
                                      input_resolution=(patches_resolution[0],
                                                        patches_resolution[1]),
                                      depth=depths[0],
                                      window_size=window_size,
                                      mlp_ratio=self.mlp_ratio,
                                      drop=drop_rate,
                                      drop_path=dpr[sum(depths[:0]):sum(depths[:0 + 1])],
                                      norm_layer=norm_layer,
                                      use_checkpoint=use_checkpoint)
        self.EN2_0 = MambaStageModule(dim=int(embed_dim),  # Mamba模块输入通道数
                                      input_resolution=(patches_resolution[0],
                                                        patches_resolution[1]),
                                      depth=depths[1],
                                      window_size=window_size,
                                      mlp_ratio=self.mlp_ratio,
                                      drop=drop_rate,
                                      drop_path=dpr[sum(depths[:1]):sum(depths[:1 + 1])],
                                      norm_layer=norm_layer,
                                      use_checkpoint=use_checkpoint)
        self.EN3_0 = MambaStageModule(dim=int(embed_dim),  # Mamba模块输入通道数
                                      input_resolution=(patches_resolution[0],
                                                        patches_resolution[1]),
                                      depth=depths[2],
                                      window_size=window_size,
                                      mlp_ratio=self.mlp_ratio,
                                      drop=drop_rate,
                                      drop_path=dpr[sum(depths[:2]):sum(depths[:2 + 1])],
                                      norm_layer=norm_layer,
                                      use_checkpoint=use_checkpoint)

        self.norm = norm_layer(self.num_features)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def transformer(self, x):
        B, L, C = x.shape
        H = W = int(L ** 0.5)
        x = x.view(B, H, W, C)
        x = x.permute(0, 3, 1, 2)
        return x

    def forward(self, x):
        x = self.reflection_pad(x)
        x = self.conv_begin(x)
        x = self.BatchNorm2d(x)
        x = self.LeakyReLU(x)

        x = self.patch_embed(x)

        x = self.pos_drop(x)
        x1 = self.EN1_0(x)
        x1 = x1 + x
        x2 = self.EN2_0(x1)
        x2 = x2 + x1
        x3 = self.EN3_0(x2)
        x3 = x3 + x2

        x3 = self.norm(x3)

        x3 = self.transformer(x3)

        x4 = self.reflection_pad(x3)
        x4 = self.conv_end(x4)
        x4 = self.BatchNorm2d(x4)
        x4 = self.LeakyReLU(x4)

        return x4


class Detail_MODEL(nn.Module):
    def __init__(self, in_channel=2, out_channel=32, kernal_size=3):
        super(Detail_MODEL, self).__init__()
        reflection_padding = int(np.floor(kernal_size / 2))
        self.reflection_pad = nn.ReflectionPad2d(reflection_padding)

        self.Conv = nn.Conv2d(in_channels=in_channel, out_channels=16, kernel_size=kernal_size, stride=1)
        self.BatchNorm2d_1 = torch.nn.BatchNorm2d(16)
        self.LeakyReLU = torch.nn.LeakyReLU(0.2)
        self.up = nn.Upsample(scale_factor=8, mode='bilinear', align_corners=True)

        self.Conv_d1 = nn.Conv2d(in_channels=16, out_channels=8, kernel_size=kernal_size, stride=1)
        self.Conv_d2 = nn.Conv2d(in_channels=8, out_channels=8, kernel_size=kernal_size, stride=1)
        self.Conv_d3 = nn.Conv2d(in_channels=16, out_channels=8, kernel_size=kernal_size, stride=1)
        self.Conv_d4 = nn.Conv2d(in_channels=24, out_channels=8, kernel_size=kernal_size, stride=1)
        self.BatchNorm2d_2 = torch.nn.BatchNorm2d(8)

        self.Conv_s1 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=kernal_size, stride=2)
        self.Conv_s2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=kernal_size, stride=2)
        self.Conv_s3 = nn.Conv2d(in_channels=64, out_channels=32, kernel_size=kernal_size, stride=2)
        self.Conv_s4 = nn.Conv2d(in_channels=32, out_channels=out_channel, kernel_size=kernal_size, stride=1)
        self.BatchNorm2d_3 = torch.nn.BatchNorm2d(32)
        self.BatchNorm2d_4 = torch.nn.BatchNorm2d(64)

    def forward(self, input):
        x = self.reflection_pad(input)
        x = self.Conv(x)
        x = self.BatchNorm2d_1(x)
        x_output = self.LeakyReLU(x)

        d_1 = self.reflection_pad(x_output)
        d_1 = self.Conv_d1(d_1)
        d_1 = self.BatchNorm2d_2(d_1)
        d_1_output = self.LeakyReLU(d_1)

        d_2 = self.reflection_pad(d_1_output)
        d_2 = self.Conv_d2(d_2)
        d_2 = self.BatchNorm2d_2(d_2)
        d_2_output = self.LeakyReLU(d_2)

        d_3 = torch.cat([d_1_output, d_2_output], 1)
        d_3 = self.reflection_pad(d_3)
        d_3 = self.Conv_d3(d_3)
        d_3 = self.BatchNorm2d_2(d_3)
        d_3_output = self.LeakyReLU(d_3)

        d_4 = torch.cat([d_1_output, d_2_output, d_3_output], 1)
        d_4 = self.reflection_pad(d_4)
        d_4 = self.Conv_d4(d_4)
        d_4 = self.BatchNorm2d_2(d_4)
        d_4_output = self.LeakyReLU(d_4)

        d_output = torch.cat([d_1_output, d_2_output, d_3_output, d_4_output], 1)

        s_1 = self.reflection_pad(x)
        s_1 = self.Conv_s1(s_1)
        s_1 = self.BatchNorm2d_3(s_1)
        s_1_output = self.LeakyReLU(s_1)

        s_2 = self.reflection_pad(s_1_output)
        s_2 = self.Conv_s2(s_2)
        s_2 = self.BatchNorm2d_4(s_2)
        s_2_output = self.LeakyReLU(s_2)

        s_3 = self.reflection_pad(s_2_output)
        s_3 = self.Conv_s3(s_3)
        s_3 = self.BatchNorm2d_3(s_3)
        s_3_output = self.LeakyReLU(s_3)

        s_4 = self.up(s_3_output)
        s_4 = self.reflection_pad(s_4)
        s_4 = self.Conv_s4(s_4)
        s_4 = self.BatchNorm2d_3(s_4)
        s_output = self.LeakyReLU(s_4)

        out = torch.cat([d_output, s_output], 1)

        return out


class Fusion_MODEL(nn.Module):
    def __init__(self, in_channel=64, output_channel=1):
        super(Fusion_MODEL, self).__init__()
        self.convolutional_out = Basic1x1(in_channel, output_channel)

    def forward(self, input_1, input_2):
        add = input_1 + input_2
        out = self.convolutional_out(add)
        return out



