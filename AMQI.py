import torch
import torch.nn as nn
import math
from functools import partial
from typing import Callable, Any
from einops import rearrange, repeat
from timm.models.layers import DropPath, trunc_normal_

import selective_scan_cuda  # from mamba-ssm v1.2.0.post1
from ResNet50 import ResNet50


class SFE(nn.Module):
    """
    Shallow Feature Extractor (SFE)
    """

    def __init__(self, in_channels, out_channels, kernel_size=3, stride=2):
        super(SFE, self).__init__()

        # SFE
        self.sfe = nn.Sequential(
            # Conv + IN + ReLU
            nn.Conv2d(in_channels, out_channels, kernel_size, stride=stride, padding=(kernel_size // 2), bias=True),
            nn.InstanceNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):  # B 3 H W
        # SFE
        x = self.sfe(x)  # B C H W
        return x


class CA(nn.Module):
    """
    Channel Attention (CA)
    """

    def __init__(self, in_channels, ratio=8):
        super(CA, self).__init__()

        # Avg Pool & Max pool
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        # MLP
        self.mlp = nn.Sequential(
            # Conv + ReLU + Conv (Replace Linear + ReLU + Linear)
            nn.Conv2d(in_channels, in_channels // ratio, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels // ratio, in_channels, 1, bias=False)
        )
        # Sigmoid
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):  # B C H W
        # Weight
        w = self.mlp(self.avg_pool(x)) + self.mlp(self.max_pool(x))  # B C 1 1
        w = self.sigmoid(w)  # B C 1 1
        # Weight * X
        x = w * x  # B C H W
        return x


class LKS(nn.Module):
    """
    Large Kernel-guided Spatial Attention (LKS)
    """

    def __init__(self, in_channels, dw_kernel_size=5, dwd_kernel_size=7, dwd_dilation_size=3):
        super(LKS, self).__init__()

        # DWConv + DWDConv + PWConv
        self.dwconv = nn.Conv2d(in_channels, in_channels, kernel_size=dw_kernel_size, padding=dw_kernel_size // 2,
                                groups=in_channels, bias=False)
        self.dwdconv = nn.Conv2d(in_channels, in_channels, kernel_size=dwd_kernel_size,
                                 padding=(dwd_dilation_size * (dwd_kernel_size - 1)) // 2,
                                 dilation=dwd_dilation_size, groups=in_channels, bias=False)
        self.pwconv = nn.Conv2d(in_channels, in_channels, kernel_size=1, padding=0, bias=False)
        # Sigmoid
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):  # B C H W
        # Weight
        w = self.pwconv(self.dwdconv(self.dwconv(x)))  # B C H W
        w = self.sigmoid(w)  # B C H W
        # Weight * X
        x = w * x  # B C H W
        return x


class CALKS(nn.Module):
    """
    Channel Attention + Large Kernel-guided Spatial Attention (CALKS)
    """

    def __init__(self, in_channels):
        super(CALKS, self).__init__()

        # Fusion
        self.conv1 = nn.Conv2d(in_channels, in_channels, 3, padding=1, bias=False)
        # CA & LKS
        self.ca = CA(in_channels)
        self.lks = LKS(in_channels)
        # Fusion
        self.conv2 = nn.Conv2d(in_channels, in_channels, 3, padding=1, bias=False)

    def forward(self, x):  # B C H W
        residual = x
        # Fusion
        x = self.conv1(x)  # B C H W
        # CA
        x = self.ca(x)  # B C H W
        # LKS
        x = self.lks(x)  # B C H W
        # Fusion
        x = self.conv2(x)  # B C H W
        # Residual
        x = x + residual  # B C H W
        return x


class FFN(nn.Module):
    """
    Feed Forward Network (FFN)
    """

    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super(FFN, self).__init__()

        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Conv2d(in_features, hidden_features, 1)
        self.dwconv = nn.Conv2d(hidden_features, hidden_features, 3, 1, 1, bias=True, groups=hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Conv2d(hidden_features, out_features, 1)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.dwconv(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class LIRNet(nn.Module):
    """
    Local Information Representation Network (LIRNet)
    """

    def __init__(self, in_channels, mlp_ratio=4., drop=0., drop_path=0., act_layer=nn.GELU):
        super(LIRNet, self).__init__()

        self.norm1 = nn.BatchNorm2d(in_channels)
        self.calks = CALKS(in_channels)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

        self.norm2 = nn.BatchNorm2d(in_channels)
        mlp_hidden_dim = int(in_channels * mlp_ratio)
        self.ffn = FFN(in_features=in_channels, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

    def forward(self, x):
        x = x + self.drop_path(self.calks(self.norm1(x)))
        x = x + self.drop_path(self.ffn(self.norm2(x)))
        return x


class LIRNetGroup(nn.Module):
    """
    Local Information Representation Network Group (LIRNetGroup)
    """

    def __init__(self, in_channels, depths=6):
        super(LIRNetGroup, self).__init__()

        # LIRNetGroup
        modules = [LIRNet(in_channels=in_channels) for _ in range(depths)]
        self.group = nn.Sequential(*modules)

    def forward(self, x):  # B C H W
        # LIRNetGroup
        out = self.group(x)  # B C H W
        return out


class SelectiveScanFn(torch.autograd.Function):
    """
    SelectiveScanFn
    """

    @staticmethod
    def forward(ctx, u, delta, A, B, C, D=None, z=None, delta_bias=None, delta_softplus=False, return_last_state=False):
        if u.stride(-1) != 1:
            u = u.contiguous()
        if delta.stride(-1) != 1:
            delta = delta.contiguous()
        if D is not None:
            D = D.contiguous()
        if B.stride(-1) != 1:
            B = B.contiguous()
        if C.stride(-1) != 1:
            C = C.contiguous()
        if z is not None and z.stride(-1) != 1:
            z = z.contiguous()
        if B.dim() == 3:
            B = rearrange(B, "b dstate l -> b 1 dstate l")
            ctx.squeeze_B = True
        if C.dim() == 3:
            C = rearrange(C, "b dstate l -> b 1 dstate l")
            ctx.squeeze_C = True
        out, x, *rest = selective_scan_cuda.fwd(u, delta, A, B, C, D, z, delta_bias, delta_softplus)
        ctx.delta_softplus = delta_softplus
        ctx.has_z = z is not None
        last_state = x[:, :, -1, 1::2]  # (batch, dim, dstate)
        if not ctx.has_z:
            ctx.save_for_backward(u, delta, A, B, C, D, delta_bias, x)
            return out if not return_last_state else (out, last_state)
        else:
            ctx.save_for_backward(u, delta, A, B, C, D, z, delta_bias, x, out)
            out_z = rest[0]
            return out_z if not return_last_state else (out_z, last_state)

    @staticmethod
    def backward(ctx, dout, *args):
        if not ctx.has_z:
            u, delta, A, B, C, D, delta_bias, x = ctx.saved_tensors
            z = None
            out = None
        else:
            u, delta, A, B, C, D, z, delta_bias, x, out = ctx.saved_tensors
        if dout.stride(-1) != 1:
            dout = dout.contiguous()
        # The kernel supports passing in a pre-allocated dz (e.g., in case we want to fuse the
        # backward of selective_scan_cuda with the backward of chunk).
        # Here we just pass in None and dz will be allocated in the C++ code.
        du, ddelta, dA, dB, dC, dD, ddelta_bias, *rest = selective_scan_cuda.bwd(
            u, delta, A, B, C, D, z, delta_bias, dout, x, out, None, ctx.delta_softplus,
            False  # option to recompute out_z, not used here
        )
        dz = rest[0] if ctx.has_z else None
        dB = dB.squeeze(1) if getattr(ctx, "squeeze_B", False) else dB
        dC = dC.squeeze(1) if getattr(ctx, "squeeze_C", False) else dC
        return (du, ddelta, dA, dB, dC, dD if D is not None else None,
                dz, ddelta_bias if delta_bias is not None else None, None, None)


def selective_scan_fn(u, delta, A, B, C, D=None, z=None, delta_bias=None, delta_softplus=False, return_last_state=False):
    """
    if return_last_state is True, returns (out, last_state)
    last_state has shape (batch, dim, dstate). Note that the gradient of the last state is
    not considered in the backward pass.
    """
    return SelectiveScanFn.apply(u, delta, A, B, C, D, z, delta_bias, delta_softplus, return_last_state)


class CrossScan(torch.autograd.Function):
    """
    CrossScan
    """

    @staticmethod
    def forward(ctx, x: torch.Tensor):
        B, C, H, W = x.shape
        ctx.shape = (B, C, H, W)
        xs = x.new_empty((B, 4, C, H * W))
        xs[:, 0] = x.flatten(2, 3)
        xs[:, 1] = x.transpose(dim0=2, dim1=3).flatten(2, 3)
        xs[:, 2:4] = torch.flip(xs[:, 0:2], dims=[-1])
        return xs

    @staticmethod
    def backward(ctx, ys: torch.Tensor):
        # out: (b, k, d, l)
        B, C, H, W = ctx.shape
        L = H * W
        ys = ys[:, 0:2] + ys[:, 2:4].flip(dims=[-1]).view(B, 2, -1, L)
        y = ys[:, 0] + ys[:, 1].view(B, -1, W, H).transpose(dim0=2, dim1=3).contiguous().view(B, -1, L)
        return y.view(B, -1, H, W)


class CrossMerge(torch.autograd.Function):
    """
    CrossMerge
    """

    @staticmethod
    def forward(ctx, ys: torch.Tensor):
        B, K, D, H, W = ys.shape
        ctx.shape = (H, W)
        ys = ys.view(B, K, D, -1)
        ys = ys[:, 0:2] + ys[:, 2:4].flip(dims=[-1]).view(B, 2, D, -1)
        y = ys[:, 0] + ys[:, 1].view(B, -1, W, H).transpose(dim0=2, dim1=3).contiguous().view(B, D, -1)
        return y

    @staticmethod
    def backward(ctx, x: torch.Tensor):
        # B, D, L = x.shape
        # out: (b, k, d, l)
        H, W = ctx.shape
        B, C, L = x.shape
        xs = x.new_empty((B, 4, C, L))
        xs[:, 0] = x
        xs[:, 1] = x.view(B, C, H, W).transpose(dim0=2, dim1=3).flatten(2, 3)
        xs[:, 2:4] = torch.flip(xs[:, 0:2], dims=[-1])
        xs = xs.view(B, 4, C, H, W)
        return xs


def cross_selective_scan(
        x: torch.Tensor = None,
        x_proj_weight: torch.Tensor = None,
        x_proj_bias: torch.Tensor = None,
        dt_projs_weight: torch.Tensor = None,
        dt_projs_bias: torch.Tensor = None,
        A_logs: torch.Tensor = None,
        Ds: torch.Tensor = None,
        delta_softplus=True,
        out_norm: torch.nn.Module = None,
        # ==============================
        to_dtype=True,  # True: final out to dtype
        force_fp32=False,  # True: input fp32
        CrossScan=CrossScan,
        CrossMerge=CrossMerge):

    B, D, H, W = x.shape
    D, N = A_logs.shape
    K, D, R = dt_projs_weight.shape
    L = H * W

    xs = CrossScan.apply(x)
    x_dbl = torch.einsum("b k d l, k c d -> b k c l", xs, x_proj_weight)
    if x_proj_bias is not None:
        x_dbl = x_dbl + x_proj_bias.view(1, K, -1, 1)
    dts, Bs, Cs = torch.split(x_dbl, [R, N, N], dim=2)
    dts = torch.einsum("b k r l, k d r -> b k d l", dts, dt_projs_weight)

    xs = xs.view(B, -1, L)
    dts = dts.contiguous().view(B, -1, L)
    As = -torch.exp(A_logs.to(torch.float))  # (k * c, d_state)
    Bs = Bs.contiguous().view(B, K, N, L)
    Cs = Cs.contiguous().view(B, K, N, L)
    Ds = Ds.to(torch.float)  # (K * c)
    delta_bias = dt_projs_bias.view(-1).to(torch.float)

    if force_fp32:
        xs = xs.to(torch.float)
        dts = dts.to(torch.float)
        Bs = Bs.to(torch.float)
        Cs = Cs.to(torch.float)

    # VMamba -> SelectiveScanCore
    # Swin-UMamba -> selective_scan_fn
    ys: torch.Tensor = selective_scan_fn(
        xs, dts, As, Bs, Cs, Ds, z=None, delta_bias=delta_bias, delta_softplus=delta_softplus, return_last_state=False
    ).view(B, K, -1, H, W)

    y: torch.Tensor = CrossMerge.apply(ys)
    y = y.transpose(dim0=1, dim1=2).contiguous()  # (B, L, C)
    y = out_norm(y).view(B, H, W, -1)
    return (y.to(x.dtype) if to_dtype else y)


class Permute(nn.Module):
    """
    Permute
    """

    def __init__(self, *args):
        super(Permute, self).__init__()

        self.args = args

    def forward(self, x: torch.Tensor):
        return x.permute(*self.args)


class Mlp(nn.Module):
    """
    Mlp
    """

    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0., channels_first=False):
        super(Mlp, self).__init__()

        out_features = out_features or in_features
        hidden_features = hidden_features or in_features

        Linear = partial(nn.Conv2d, kernel_size=1, padding=0) if channels_first else nn.Linear
        self.fc1 = Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class VSSM(nn.Module):
    """
    Visual State Space Model (VSSM)
    """

    def __init__(
            self,
            # basic dims ===========
            d_model=96,
            d_state=16,
            ssm_ratio=2.0,
            dt_rank="auto",
            act_layer=nn.SiLU,
            # dwconv ===============
            d_conv=3,  # < 2 means no conv
            conv_bias=True,
            # ======================
            dropout=0.0,
            bias=False,
            # dt init ==============
            dt_min=0.001,
            dt_max=0.1,
            dt_init="random",
            dt_scale=1.0,
            dt_init_floor=1e-4):

        super(VSSM, self).__init__()

        d_inner = int(ssm_ratio * d_model)
        dt_rank = math.ceil(d_model / 16) if dt_rank == "auto" else dt_rank
        self.d_conv = d_conv
        self.out_norm = nn.LayerNorm(d_inner)

        # forward_type debug =======================================
        self.forward_core = partial(self.forward_corev2, force_fp32=True)
        k_group = 4

        # in proj =======================================
        d_proj = d_inner * 2
        self.in_proj = nn.Linear(d_model, d_proj, bias=bias)
        self.act: nn.Module = act_layer()

        # conv =======================================
        self.conv2d = nn.Conv2d(
            in_channels=d_inner,
            out_channels=d_inner,
            groups=d_inner,
            bias=conv_bias,
            kernel_size=d_conv,
            padding=(d_conv - 1) // 2,
        )

        # x proj ============================
        self.x_proj = [
            nn.Linear(d_inner, (dt_rank + d_state * 2), bias=False)
            for _ in range(k_group)
        ]
        self.x_proj_weight = nn.Parameter(torch.stack([t.weight for t in self.x_proj], dim=0))  # (K, N, inner)
        del self.x_proj

        # out proj =======================================
        self.out_proj = nn.Linear(d_inner, d_model, bias=bias)
        self.dropout = nn.Dropout(dropout) if dropout > 0. else nn.Identity()

        # dt proj ============================
        self.dt_projs = [
            self.dt_init(dt_rank, d_inner, dt_scale, dt_init, dt_min, dt_max, dt_init_floor)
            for _ in range(k_group)
        ]
        self.dt_projs_weight = nn.Parameter(
            torch.stack([t.weight for t in self.dt_projs], dim=0))  # (K, inner, rank)
        self.dt_projs_bias = nn.Parameter(torch.stack([t.bias for t in self.dt_projs], dim=0))  # (K, inner)
        del self.dt_projs

        # A, D =======================================
        self.A_logs = self.A_log_init(d_state, d_inner, copies=k_group, merge=True)  # (K * D, N)
        self.Ds = self.D_init(d_inner, copies=k_group, merge=True)  # (K * D)

    @staticmethod
    def dt_init(dt_rank, d_inner, dt_scale=1.0, dt_init="random", dt_min=0.001, dt_max=0.1, dt_init_floor=1e-4):
        dt_proj = nn.Linear(dt_rank, d_inner, bias=True)

        # Initialize special dt projection to preserve variance at initialization
        dt_init_std = dt_rank ** -0.5 * dt_scale
        if dt_init == "constant":
            nn.init.constant_(dt_proj.weight, dt_init_std)
        elif dt_init == "random":
            nn.init.uniform_(dt_proj.weight, -dt_init_std, dt_init_std)
        else:
            raise NotImplementedError

        # Initialize dt bias so that F.softplus(dt_bias) is between dt_min and dt_max
        dt = torch.exp(
            torch.rand(d_inner) * (math.log(dt_max) - math.log(dt_min))
            + math.log(dt_min)
        ).clamp(min=dt_init_floor)
        # Inverse of softplus: https://github.com/pytorch/pytorch/issues/72759
        inv_dt = dt + torch.log(-torch.expm1(-dt))
        with torch.no_grad():
            dt_proj.bias.copy_(inv_dt)
        # Our initialization would set all Linear.bias to zero, need to mark this one as _no_reinit
        # dt_proj.bias._no_reinit = True
        return dt_proj

    @staticmethod
    def A_log_init(d_state, d_inner, copies=-1, device=None, merge=True):
        # S4D real initialization
        A = repeat(
            torch.arange(1, d_state + 1, dtype=torch.float32, device=device),
            "n -> d n",
            d=d_inner,
        ).contiguous()
        A_log = torch.log(A)  # Keep A_log in fp32
        if copies > 0:
            A_log = repeat(A_log, "d n -> r d n", r=copies)
            if merge:
                A_log = A_log.flatten(0, 1)
        A_log = nn.Parameter(A_log)
        A_log._no_weight_decay = True
        return A_log

    @staticmethod
    def D_init(d_inner, copies=-1, device=None, merge=True):
        # D "skip" parameter
        D = torch.ones(d_inner, device=device)
        if copies > 0:
            D = repeat(D, "n1 -> r n1", r=copies)
            if merge:
                D = D.flatten(0, 1)
        D = nn.Parameter(D)  # Keep in fp32
        D._no_weight_decay = True
        return D

    def forward_corev2(self, x: torch.Tensor, cross_selective_scan=cross_selective_scan, **kwargs):
        x_proj_weight = self.x_proj_weight
        dt_projs_weight = self.dt_projs_weight
        dt_projs_bias = self.dt_projs_bias
        A_logs = self.A_logs
        Ds = self.Ds
        out_norm = getattr(self, "out_norm", None)

        return cross_selective_scan(
            x, x_proj_weight, None, dt_projs_weight, dt_projs_bias,
            A_logs, Ds, delta_softplus=True,
            out_norm=out_norm,
            **kwargs,
        )

    def forward(self, x: torch.Tensor, **kwargs):
        # Linear
        x = self.in_proj(x)
        x, z = x.chunk(2, dim=-1)  # (b, h, w, d)
        # SiLU
        z = self.act(z)
        x = x.permute(0, 3, 1, 2).contiguous()
        # DWConv
        x = self.conv2d(x)  # (b, d, h, w)
        # SiLU
        x = self.act(x)
        # SS2D
        y = self.forward_core(x)
        y = y * z
        # Linear
        out = self.dropout(self.out_proj(y))
        return out


class VSSMEncoder(nn.Module):
    """
    Visual State Space Model Encoder (VSSMEncoder)
    """

    def __init__(
            self,
            hidden_dim: int = 0,
            drop_path: float = 0,
            norm_layer: Callable[..., torch.nn.Module] = partial(nn.LayerNorm, eps=1e-6),
            # =============================
            ssm_d_state: int = 16,
            ssm_ratio=2.0,
            ssm_dt_rank: Any = "auto",
            ssm_act_layer=nn.SiLU,
            ssm_conv: int = 3,
            ssm_conv_bias=True,
            ssm_drop_rate: float = 0,
            # =============================
            mlp_ratio=4.0,
            mlp_act_layer=nn.GELU,
            mlp_drop_rate: float = 0.0):

        super(VSSMEncoder, self).__init__()

        # LayerNorm
        self.norm1 = norm_layer(hidden_dim)
        # VSSM
        self.vssm = VSSM(
            d_model=hidden_dim,
            d_state=ssm_d_state,
            ssm_ratio=ssm_ratio,
            dt_rank=ssm_dt_rank,
            act_layer=ssm_act_layer,
            # ==========================
            d_conv=ssm_conv,
            conv_bias=ssm_conv_bias,
            # ==========================
            dropout=ssm_drop_rate,
            bias=False,
        )
        self.drop_path = DropPath(drop_path)
        # LayerNorm
        self.norm2 = norm_layer(hidden_dim)
        # MLP
        mlp_hidden_dim = int(hidden_dim * mlp_ratio)
        self.mlp = Mlp(in_features=hidden_dim, hidden_features=mlp_hidden_dim, act_layer=mlp_act_layer,
                       drop=mlp_drop_rate)

    def forward(self, x: torch.Tensor):
        x = x + self.drop_path(self.vssm(self.norm1(x)))  # VSSM
        x = x + self.drop_path(self.mlp(self.norm2(x)))  # MLP
        return x


class GIRNet(nn.Module):
    """
    Global Information Representation Network (GIRNet)
    """

    def __init__(
            self,
            in_channels=48,
            patch_size=2,
            dims=96,
            depths=6,
            # =========================
            ssm_d_state=16,
            ssm_ratio=2.0,
            ssm_dt_rank="auto",
            ssm_act_layer=nn.SiLU,
            ssm_conv=3,
            ssm_conv_bias=True,
            ssm_drop_rate=0.0,
            # =========================
            mlp_ratio=4.0,
            mlp_act_layer=nn.GELU,
            mlp_drop_rate=0.0,
            # =========================
            drop_path_rate=0.1,
            norm_layer=nn.LayerNorm):
        
        super(GIRNet, self).__init__()

        self.patch_merge = nn.Conv2d(in_channels, dims, kernel_size=patch_size, stride=patch_size, bias=True)
        self.norm = norm_layer(dims)

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depths)]  # stochastic depth decay rule
        # VSSMEncoders
        self.vssmencoders = nn.ModuleList()
        for d in range(depths):
            self.vssmencoders.append(VSSMEncoder(
                hidden_dim=dims,
                drop_path=dpr[d],
                norm_layer=norm_layer,
                ssm_d_state=ssm_d_state,
                ssm_ratio=ssm_ratio,
                ssm_dt_rank=ssm_dt_rank,
                ssm_act_layer=ssm_act_layer,
                ssm_conv=ssm_conv,
                ssm_conv_bias=ssm_conv_bias,
                ssm_drop_rate=ssm_drop_rate,
                mlp_ratio=mlp_ratio,
                mlp_act_layer=mlp_act_layer,
                mlp_drop_rate=mlp_drop_rate,
            ))

    def forward(self, x: torch.Tensor, f: torch.Tensor=None):
        # Patch Merge (Down Sampling)
        x = self.patch_merge(x)
        if f != None:
            x = x * f
        x = x.permute(0, 2, 3, 1)
        x = self.norm(x)
        # VSSMEncoders
        for vssmencoder in self.vssmencoders:
            x = vssmencoder(x)
        return x


class Predictor(nn.Module):
    """
    Predictor
    """

    def __init__(self, dim=384):
        super(Predictor, self).__init__()

        self.predictor = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),  # B C 1 1
            nn.Flatten(1),  # B C
            nn.Linear(dim, 1),  # B 1
        )

    def forward(self, x):  # B C H W
        x = self.predictor(x)  # B 1
        return x


class AMQI(nn.Module):
    """
    Attention and Mamba-driven Quality Index (AMQI)
    """

    def __init__(self,
                 in_channels=3,
                 lirnet_channels=32,
                 lirnet_depths=6,
                 girnet_depths=[3, 3, 3],
                 girnet_dims=[96, 192, 384],
                 pretrained_model_path='',
                 has_resnet50=True):

        super(AMQI, self).__init__()

        # SFE
        self.sfe = SFE(in_channels, lirnet_channels)
        # LIRNets
        self.lirnets = LIRNetGroup(in_channels=lirnet_channels, depths=lirnet_depths)
        # GIRNet Stage1
        self.girnetstage1 = GIRNet(in_channels=lirnet_channels, dims=girnet_dims[0], depths=girnet_depths[0])
        # GIRNet Stage2
        self.girnetstage2 = GIRNet(in_channels=girnet_dims[0], dims=girnet_dims[1], depths=girnet_depths[1])
        # GIRNet Stage3
        self.girnetstage3 = GIRNet(in_channels=girnet_dims[1], dims=girnet_dims[2], depths=girnet_depths[2])

        # Weight init
        # self.apply(self._init_weights)

        self.has_resnet50 = has_resnet50
        if has_resnet50:
            # ResNet50
            # self.resnet50 = ResNet50(pretrained_model_path=pretrained_model_path, pretrained=True)  # Train Need
            self.resnet50 = ResNet50(pretrained_model_path=pretrained_model_path, pretrained=False)  # Temp Test
            self.conv1 = nn.Sequential(
                nn.Conv2d(256, 64, 1),
                nn.ReLU(inplace=True),
                nn.Conv2d(64, 1, 1),
                nn.ReLU(inplace=True)
            )
            self.conv2 = nn.Sequential(
                nn.Conv2d(512, 128, 1),
                nn.ReLU(inplace=True),
                nn.Conv2d(128, 1, 1),
                nn.ReLU(inplace=True)
            )
            self.conv3 = nn.Sequential(
                nn.Conv2d(1024, 256, 1),
                nn.ReLU(inplace=True),
                nn.Conv2d(256, 64, 1),
                nn.ReLU(inplace=True),
                nn.Conv2d(64, 1, 1),
                nn.ReLU(inplace=True)
            )
            self.conv4 = nn.Sequential(
                nn.Conv2d(2048, 512, 1),
                nn.ReLU(inplace=True),
                nn.Conv2d(512, 128, 1),
                nn.ReLU(inplace=True),
                nn.Conv2d(128, 1, 1),
                nn.ReLU(inplace=True)
            )
            nn.init.kaiming_normal_(self.conv1._modules['0'].weight.data)
            nn.init.kaiming_normal_(self.conv1._modules['2'].weight.data)
            nn.init.kaiming_normal_(self.conv2._modules['0'].weight.data)
            nn.init.kaiming_normal_(self.conv2._modules['2'].weight.data)
            nn.init.kaiming_normal_(self.conv3._modules['0'].weight.data)
            nn.init.kaiming_normal_(self.conv3._modules['2'].weight.data)
            nn.init.kaiming_normal_(self.conv3._modules['4'].weight.data)
            nn.init.kaiming_normal_(self.conv4._modules['0'].weight.data)
            nn.init.kaiming_normal_(self.conv4._modules['2'].weight.data)
            nn.init.kaiming_normal_(self.conv4._modules['4'].weight.data)

        # Feature-Quality Mapping (FQM)
        self.patch_merge = nn.Conv2d(girnet_dims[-1], girnet_dims[-1] * 2, kernel_size=2, stride=2, bias=True)
        nn.init.kaiming_normal_(self.patch_merge.weight.data)
        self.predictor = Predictor(dim=girnet_dims[-1] * 2)

    def _init_weights(self, m: nn.Module):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0.0)
        elif isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight, mode="fan_out")
            if m.bias is not None:
                nn.init.constant_(m.bias, 0.0)
        elif isinstance(m, (nn.LayerNorm, nn.InstanceNorm2d, nn.BatchNorm2d)):
            if m.weight is not None:
                nn.init.constant_(m.weight, 1.0)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0.0)

    def forward(self, x):  # B 3 H W
        if self.has_resnet50:
            # ResNet50
            f = self.resnet50(x)
            f1, f2, f3, f4 = f['f1'], f['f2'], f['f3'], f['f4']
            f1 = self.conv1(f1)
            f2 = self.conv2(f2)
            f3 = self.conv3(f3)
            f4 = self.conv4(f4)
            # SFE
            x = self.sfe(x)  # B C H//2 W//2 C1
            # LIRNets
            x = self.lirnets(x)  # B C H//2 W//2 C1
            # GIRNet Stage1
            x = self.girnetstage1(x, f1)  # B H//4 W//4 C2
            x = x.permute(0, 3, 1, 2)
            # GIRNet Stage2
            x = self.girnetstage2(x, f2)  # B H//8 W//8 C3
            x = x.permute(0, 3, 1, 2)
            # GIRNet Stage3
            x = self.girnetstage3(x, f3)  # B H//16 W//16 C4
            x = x.permute(0, 3, 1, 2)
            # FQM
            x = self.patch_merge(x)  # B 2C4 H//32 W//32
            x = x * f4
            x = self.predictor(x)  # B 1
        else:
            # SFE
            x = self.sfe(x)  # B C H//2 W//2 C1
            # LIRNets
            x = self.lirnets(x)  # B C H//2 W//2 C1
            # GIRNet Stage1
            x = self.girnetstage1(x, None)  # B H//4 W//4 C2
            x = x.permute(0, 3, 1, 2)
            # GIRNet Stage2
            x = self.girnetstage2(x, None)  # B H//8 W//8 C3
            x = x.permute(0, 3, 1, 2)
            # GIRNet Stage3
            x = self.girnetstage3(x, None)  # B H//16 W//16 C4
            x = x.permute(0, 3, 1, 2)
            # FQM
            x = self.patch_merge(x)  # B 2C4 H//32 W//32
            x = self.predictor(x)  # B 1
        return x


if __name__ == "__main__":
    pass
    # Test
    input = torch.rand([2, 3, 224, 224]).cuda()
    net = AMQI(pretrained_model_path=f"...../resnet50-0676ba61.pth").cuda().eval()
    output = net(input)
    print(f"output: {output}")
