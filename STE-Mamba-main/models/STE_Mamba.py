import torch
import torch.nn as nn
import torch.nn.functional as F
from functools import partial
import torch.fft
from mamba_ssm import Mamba
from timm.models.layers import DropPath, trunc_normal_
from timm.models.registry import register_model
from timm.models.vision_transformer import _cfg
import math
import numpy as np
from einops import rearrange, repeat, einsum


class EinFFT(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.hidden_size = dim
        self.num_blocks = 4
        self.block_size = self.hidden_size // self.num_blocks
        self.dim_attention = MambaLayer(dim=self.block_size)
        self.dropout = nn.Dropout(0.1)
        assert self.hidden_size % self.num_blocks == 0
        self.sparsity_threshold = 0.01
        self.scale = 0.02
        d_model = self.block_size
        d_ff = dim * 4
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.norm4 = nn.LayerNorm(d_model)

        self.MLP1 = nn.Sequential(nn.Linear(d_model, d_ff),
                                  nn.ReLU(),
                                  nn.Linear(d_ff, d_model))
        self.MLP2 = nn.Sequential(nn.Linear(d_model, d_ff),
                                  nn.ReLU(),
                                  nn.Linear(d_ff, d_model))

        self.complex_weight_1 = nn.Parameter(
            torch.randn(2, self.num_blocks, self.block_size, self.block_size, dtype=torch.float32) * self.scale)
        self.complex_weight_2 = nn.Parameter(
            torch.randn(2, self.num_blocks, self.block_size, self.block_size, dtype=torch.float32) * self.scale)
        self.complex_bias_1 = nn.Parameter(
            torch.randn(2, self.num_blocks, self.block_size, dtype=torch.float32) * self.scale)
        self.complex_bias_2 = nn.Parameter(
            torch.randn(2, self.num_blocks, self.block_size, dtype=torch.float32) * self.scale)

    def multiply(self, input, weights):
        return torch.einsum('...bd,bdk->...bk', input, weights)

    def forward(self, x, H, W):
        B, N, C = x.shape
        x = x.view(B, N, self.num_blocks, self.block_size)
        dim_in = x.reshape(B * N, self.num_blocks, self.block_size)
        dim_enc = self.dim_attention(dim_in)
        dim_enc = dim_in + self.dropout(dim_enc)
        dim_enc = self.norm3(dim_enc)
        dim_enc = dim_enc + self.dropout(self.MLP2(dim_enc))
        dim_enc = self.norm4(dim_enc)
        x = x.to(torch.float32)
        x = x.reshape(B, N, C)
        return x


class MambaBlock(nn.Module):
    def __init__(self, d_model, d_state=64, expand=2, d_conv=4, conv_bias=True, bias=False):
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.d_conv = d_conv
        self.expand = expand
        self.conv_bias = conv_bias
        self.bias = bias
        self.d_inner = int(self.expand * self.d_model)
        self.dt_rank = math.ceil(self.d_model / 16)

        self.in_proj = nn.Linear(self.d_model, self.d_inner * 2, bias=self.bias)

        self.conv1d = nn.Conv1d(
            in_channels=self.d_inner,
            out_channels=self.d_inner,
            bias=self.conv_bias,
            kernel_size=self.d_conv,
            groups=self.d_inner,
            padding=self.d_conv - 1,
        )

        self.x_proj = nn.Linear(self.d_inner, self.dt_rank + self.d_state * 2, bias=False)

        self.dt_proj = nn.Linear(self.dt_rank, self.d_inner, bias=True)

        A = repeat(torch.arange(1, self.d_state + 1), 'n -> d n', d=self.d_inner)
        self.A_log = nn.Parameter(torch.log(A))
        self.D = nn.Parameter(torch.ones(self.d_inner))
        self.out_proj = nn.Linear(self.d_inner, self.d_model, bias=self.bias)

    def forward(self, x):
        (b, l, d) = x.shape

        x_and_res = self.in_proj(x)
        (x, res) = x_and_res.split(split_size=[self.d_inner, self.d_inner], dim=-1)

        x = rearrange(x, 'b l d_in -> b d_in l')
        x = self.conv1d(x)[:, :, :l]
        x = rearrange(x, 'b d_in l -> b l d_in')

        x = F.silu(x)

        y = self.ssm(x)

        y = y * F.silu(res)

        output = self.out_proj(y)

        return output

    def ssm(self, x):
        (d_in, n) = self.A_log.shape

        A = -torch.exp(self.A_log.float())
        D = self.D.float()

        x_dbl = self.x_proj(x)

        (delta, B, C) = x_dbl.split(split_size=[self.dt_rank, n, n], dim=-1)
        delta = F.softplus(self.dt_proj(delta))

        y = self.selective_scan(x, delta, A, B, C, D)

        return y

    def selective_scan(self, u, delta, A, B, C, D):
        (b, l, d_in) = u.shape
        n = A.shape[1]

        deltaA = torch.exp(einsum(delta, A, 'b l d_in, d_in n -> b l d_in n'))
        deltaB_u = einsum(delta, B, u, 'b l d_in, b l n, b l d_in -> b l d_in n')

        x = torch.zeros((b, d_in, n), device=deltaA.device)
        ys = []
        for i in range(l):
            x = deltaA[:, i] * x + deltaB_u[:, i]
            y = einsum(x, C[:, i, :], 'b d_in n, b n -> b d_in')
            ys.append(y)
        y = torch.stack(ys, dim=1)

        y = y + u * D

        return y


class MambaLayer(nn.Module):
    def __init__(self, dim, d_state=64, d_conv=4, expand=2):
        super().__init__()
        self.dim = dim
        self.norm = nn.LayerNorm(dim)
        self.mamba_ssm = Mamba(
            d_model=dim,
            d_state=d_state,
            d_conv=d_conv,
            expand=expand
        )

    def forward(self, x):
        B, L, C = x.shape
        x_norm = self.norm(x)
        x_mamba = self.mamba_ssm(x_norm)
        return x_mamba


def rand_bbox(size, lam, scale=1):
    W = size[1] // scale
    H = size[2] // scale
    cut_rat = np.sqrt(1. - lam)
    cut_w = int(W * cut_rat)
    cut_h = int(H * cut_rat)

    cx = np.random.randint(W)
    cy = np.random.randint(H)

    bbx1 = np.clip(cx - cut_w // 2, 0, W)
    bby1 = np.clip(cy - cut_h // 2, 0, H)
    bbx2 = np.clip(cx + cut_w // 2, 0, W)
    bby2 = np.clip(cy + cut_h // 2, 0, H)

    return bbx1, bby1, bbx2, bby2


class FFN(nn.Module):
    def __init__(self, in_features, hidden_features, args=None):
        super().__init__()
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden_features, in_features)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def forward(self, x, H, W):
        x = self.fc1(x)
        x = self.act(x)
        x = self.fc2(x)
        return x


class ClassBlock(nn.Module):
    def __init__(self, dim, mlp_ratio, norm_layer=nn.LayerNorm, cm_type='mlp', args=None):
        super().__init__()
        self.norm2 = norm_layer(dim)
        self.attn = MambaLayer(dim)
        self.mlp = FFN(dim, int(dim * mlp_ratio))
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def forward(self, x, H, W):
        cls_embed = x[:, :1]
        cls_embed = cls_embed + self.attn(x[:, :1])
        cls_embed = cls_embed + self.mlp(self.norm2(cls_embed), H, W)
        return torch.cat([cls_embed, x[:, 1:]], dim=1)


class Block_mamba(nn.Module):
    def __init__(self,
                 dim,
                 mlp_ratio,
                 drop_path=0.,
                 norm_layer=nn.LayerNorm,
                 sr_ratio=1,
                 cm_type='mlp',
                 args=None
                 ):
        super().__init__()
        self.norm2 = norm_layer(dim)
        self.attn = MambaLayer(dim)
        self.mlp = FFN(dim, int(dim * mlp_ratio), args)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def forward(self, x, H, W):
        x = x + self.drop_path(self.attn(x))
        x = x + self.drop_path(self.mlp(self.norm2(x), H, W))
        return x


class Stem(nn.Module):
    def __init__(self, in_channels, stem_hidden_dim, out_channels):
        super().__init__()
        hidden_dim = stem_hidden_dim
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, hidden_dim, kernel_size=7, stride=2,
                      padding=3, bias=False),
            nn.BatchNorm2d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_dim, hidden_dim, kernel_size=3, stride=1,
                      padding=1, bias=False),
            nn.BatchNorm2d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_dim, hidden_dim, kernel_size=3, stride=1,
                      padding=1, bias=False),
            nn.BatchNorm2d(hidden_dim),
            nn.ReLU(inplace=True),
        )
        self.proj = nn.Conv2d(hidden_dim,
                              out_channels,
                              kernel_size=3,
                              stride=2,
                              padding=1)
        self.norm = nn.LayerNorm(out_channels)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def forward(self, x):
        x = self.conv(x)

        x = self.proj(x)
        _, _, H, W = x.shape
        x = x.flatten(2).transpose(1, 2)
        x = self.norm(x)
        return x, H, W


def calculate_stem_output_shape(bs, dim, out_channels):
    new_dim_1 = math.floor((dim + 2 * 3 - 7) / 2) + 1

    new_dim_2 = new_dim_1

    new_dim_3 = math.floor((new_dim_2 + 2 * 1 - 3) / 2) + 1

    return (bs, new_dim_3, out_channels)


class DownSamples(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.proj = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=2,
                              padding=1)
        self.norm = nn.LayerNorm(out_channels)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def forward(self, x):
        x = self.proj(x)
        _, _, H, W = x.shape
        x = x.flatten(2).transpose(1, 2)
        x = self.norm(x)
        return x, H, W


def calculate_downsamples_output_shape(bs, dim, out_channels):
    new_dim = math.floor((dim + 2 * 1 - 3) / 2) + 1

    return (bs, new_dim, out_channels)


def get_embed_dim(args, embed_dims, num_stages):
    stem_out = calculate_stem_output_shape(args.batch_size, args.input_dims, embed_dims[0])
    enc_in_channel = [stem_out[1]]

    current_channel = stem_out[1]
    for i in range(num_stages - 1):
        downsample_out = calculate_downsamples_output_shape(args.batch_size, current_channel, embed_dims[i + 1])
        enc_in_channel.append(downsample_out[1])
        current_channel = downsample_out[1]

    stem_out = calculate_stem_output_shape(args.batch_size, args.in_len, embed_dims[0])
    enc_in_time = [stem_out[1]]

    current_time = stem_out[1]
    for i in range(num_stages - 1):
        downsample_out = calculate_downsamples_output_shape(args.batch_size, current_time, embed_dims[i + 1])
        enc_in_time.append(downsample_out[1])
        current_time = downsample_out[1]
    return enc_in_time, enc_in_channel


class STE_Mamba(nn.Module):
    def __init__(self,
                 num_classes=1024,
                 stem_hidden_dim=32,
                 embed_dims=[64, 128, 320, 448],
                 mlp_ratios=[8, 8, 4, 4],
                 drop_path_rate=0.,
                 norm_layer=nn.LayerNorm,
                 depths=[3, 4, 6, 3],
                 sr_ratios=[4, 2, 1, 1],
                 num_stages=4,
                 cm_type='mlp',
                 args=None,
                 **kwargs
                 ):
        super().__init__()
        self.in_chans_time = args.in_len
        self.in_chans_channel = args.input_dims
        self.num_classes = num_classes
        self.args = args
        self.depths = depths
        self.num_stages = num_stages
        self.cm_type = cm_type

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]
        cur = 0
        self.first_embed_channel = Stem(self.in_chans_channel, stem_hidden_dim, embed_dims[0])
        self.first_embed_time = Stem(self.in_chans_time, stem_hidden_dim, embed_dims[0])
        for i in range(1, num_stages):
            patch_embed = DownSamples(embed_dims[i - 1], embed_dims[i])
            setattr(self, f"patch_embed{i - 1}", patch_embed)
        for i in range(num_stages):
            channel_mix_mamba = nn.ModuleList(
                [Block_mamba(
                    dim=embed_dims[i],
                    mlp_ratio=mlp_ratios[i],
                    drop_path=dpr[cur + j],
                    norm_layer=norm_layer,
                    sr_ratio=sr_ratios[i],
                    cm_type=cm_type,
                    args=args)
                    for j in range(depths[i])]

            )
            time_mix_mamba = nn.ModuleList([Block_mamba(
                dim=embed_dims[i],
                mlp_ratio=mlp_ratios[i],
                drop_path=dpr[cur + j],
                norm_layer=norm_layer,
                sr_ratio=sr_ratios[i],
                cm_type=cm_type,
                args=args)
                for j in range(depths[i])])

            norm = norm_layer(embed_dims[i])
            cur += depths[i]

            setattr(self, f"channel_mix_mamba{i + 1}", channel_mix_mamba)
            setattr(self, f"time_mix_mamba{i + 1}", time_mix_mamba)
            setattr(self, f"norm{i + 1}", norm)

        post_layers = ['ca']
        self.post_network = nn.ModuleList([
            ClassBlock(
                dim=embed_dims[-1],
                args=args,
                mlp_ratio=mlp_ratios[-1],
                norm_layer=norm_layer,
                cm_type=self.cm_type,
            )
            for _ in range(len(post_layers))
        ])
        self.head = nn.Linear(embed_dims[-1], num_classes) if num_classes > 0 else nn.Identity()

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def forward_cls(self, x, H, W):
        B, N, C = x.shape
        cls_tokens = x.mean(dim=1, keepdim=True)
        x = torch.cat((cls_tokens, x), dim=1)
        for block in self.post_network:
            x = block(x, H, W)
        return x

    def forward_features(self, x):
        B = x.shape[0]
        x_channel, H_channel, W_channel = self.first_embed_channel(x.permute(0, 2, 1, 3))
        x_time, H_time, W_time = self.first_embed_time(x)
        for i, j in zip(range(self.num_stages), range(self.num_stages)):
            if (i != 0):
                patch_embed = getattr(self, f"patch_embed{j - 1}")
                if (self.args.use_channel_mixer):
                    x_channel, H_channel, W_channel = patch_embed(x_channel)
                if (self.args.use_time_mixer):
                    x_time, H_time, W_time = patch_embed(x_time)
            channel_mix = getattr(self, f"channel_mix_mamba{i + 1}")
            time_mix = getattr(self, f"time_mix_mamba{i + 1}")
            for t_mix, c_mix in zip(time_mix, channel_mix):
                if (self.args.use_time_mixer):
                    x_time = t_mix(x_time, H_time, W_time)
                if (self.args.use_channel_mixer):
                    x_channel = c_mix(x_channel, H_channel, W_channel)

            if i != self.num_stages - 1:
                norm = getattr(self, f"norm{i + 1}")
                if (self.args.use_channel_mixer):
                    x_channel = norm(x_channel)
                    x_channel = x_channel.reshape(B, H_channel, W_channel, -1).permute(0, 3, 1,
                                                                                       2).contiguous()
                if (self.args.use_time_mixer):
                    x_time = norm(x_time)
                    x_time = x_time.reshape(B, H_time, W_time, -1).permute(0, 3, 1, 2).contiguous()
        norm = getattr(self, f"norm{self.num_stages}")
        if (self.args.use_time_mixer):
            x_time = self.forward_cls(x_time, H_time, W_time)[:, 0]
            x_time = norm(x_time)
        if (self.args.use_channel_mixer):
            x_channel = self.forward_cls(x_channel, H_channel, W_channel)[:, 0]
            x_channel = norm(x_channel)

        return x_time, x_channel

    def forward(self, x):
        x = x.unsqueeze(-1)
        x_time, x_channel = self.forward_features(x)
        if (self.args.use_time_mixer):
            x_time = self.head(x_time)
        if (self.args.use_channel_mixer):
            x_channel = self.head(x_channel)
        return x_time, x_channel


class DWConv(nn.Module):
    def __init__(self, dim=768):
        super(DWConv, self).__init__()
        self.dwconv = nn.Conv2d(dim, dim, 3, 1, 1, bias=True, groups=dim)

    def forward(self, x, H, W):
        B, N, C = x.shape
        x = x.transpose(1, 2).view(B, C, H, W)
        x = self.dwconv(x)
        x = x.flatten(2).transpose(1, 2)
        return x


@register_model
def STE_Mamba_s(pretrained=False, **kwargs):
    model = STE_Mamba(
        stem_hidden_dim=32,
        embed_dims=[64, 128, 320, 448],
        mlp_ratios=[8, 8, 4, 4],
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        depths=[3, 4, 6, 3],
        sr_ratios=[4, 2, 1, 1],
        **kwargs)
    model.default_cfg = _cfg()
    return model


@register_model
def STE_Mamba_b(pretrained=False, **kwargs):
    model = STE_Mamba(
        stem_hidden_dim=64,
        embed_dims=[64, 128, 320, 512],
        mlp_ratios=[8, 8, 4, 4],
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        depths=[3, 4, 12, 3],
        sr_ratios=[4, 2, 1, 1],
        **kwargs)
    model.default_cfg = _cfg()
    return model


@register_model
def STE_Mamba_l(pretrained=False, **kwargs):
    model = STE_Mamba(
        stem_hidden_dim=64,
        embed_dims=[96, 192, 384, 512],
        mlp_ratios=[8, 8, 4, 4],
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        depths=[3, 6, 18, 3],
        sr_ratios=[4, 2, 1, 1],
        **kwargs)
    model.default_cfg = _cfg()
    return model