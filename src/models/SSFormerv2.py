import torch
import torch.nn as nn
import torch.nn.functional as F
from functools import partial
import logging

from timm.models.layers import DropPath
import math
import numpy as np
from mamba_ssm import Mamba, Mamba2

from utils import calculate_output_length
from .layers import EinFFT, FlattenHead


import mne
import matplotlib.pyplot as plt

log = logging.getLogger(__name__)

# Shape throughout the model is B, C, D, N

class EinFFTInd(nn.Module):
    def __init__(self, in_channels, d_model):
        super().__init__()
        self.einfft2 = EinFFT(dim=d_model*in_channels, num_blocks=d_model)

    def forward(self, x):
        B, C, D, L = x.shape

        # Channel mixing
        x = x.permute(0, 3, 2, 1) 
        x = x.reshape(B, L, D*C)
        x = self.einfft2(x)
        x = x.reshape(B, L, D, C)
        x = x.permute(0, 3, 2, 1)

        return x

class PWInd(nn.Module):
    def __init__(self, in_channels, d_model):
        super().__init__()

        self.pw2_1 = nn.Conv1d(in_channels=in_channels*d_model, out_channels=2*in_channels*d_model, kernel_size=1, stride=1, padding=0, groups=d_model)
        self.pw2_act1 = nn.ReLU()
        self.pw2_2 = nn.Conv1d(in_channels=2*in_channels*d_model, out_channels=in_channels*d_model, kernel_size=1, stride=1, padding=0, groups=d_model)

    def forward(self, x):

        B, M, D, N = x.shape

        x = x.permute(0, 2, 1, 3)
        x = x.reshape(B, D*M, N)
        x = self.pw2_1(x)
        x = self.pw2_act1(x)
        x = self.pw2_2(x)
        x = x.reshape(B, D, M, N)
        x = x.permute(0, 2, 1, 3)

        return x

class PW(nn.Module):
    def __init__(self, in_channels, d_model):
        super().__init__()

        self.pw1_1 = nn.Conv1d(in_channels=in_channels*d_model, out_channels=2*in_channels*d_model, kernel_size=1, stride=1, padding=0)
        self.pw1_act1 = nn.ReLU()
        self.pw1_2 = nn.Conv1d(in_channels=2*in_channels*d_model, out_channels=in_channels*d_model, kernel_size=1, stride=1, padding=0)

    def forward(self, x):

        B, M, D, N = x.shape

        x = x.reshape(B, M*D, N)
        x = self.pw1_1(x)
        x = self.pw1_act1(x)
        x = self.pw1_2(x)
        x = x.reshape(B, M, D, N)

        return x

class Patcher(nn.Module):
    def __init__(
        self,
        d_in, 
        d_model,
        patch_size,
        patch_stride,
        separable=False,
        input_len=2000,
        norm_along_tokens=True,
        norm_type='layernorm',
        use_alternative=False

    ):
        super().__init__()

        self.d_in = d_in
        self.d_model = d_model
        self.patch_size = patch_size
        self.patch_stride = patch_stride
        self.use_alternative = use_alternative


        if use_alternative:
            self.patcher = nn.Conv1d( 
                in_channels=self.d_in,
                out_channels=self.d_model,
                kernel_size=self.patch_size,
                stride=self.patch_stride
            )
            self.norm1 = NormWrapper(
                d_model=self.d_model, 
                num_tokens=input_len//self.patch_stride,
                along_tokens=norm_along_tokens,
                norm_type=norm_type,
                bc_combined=True
            )
            self.act1 = nn.GELU()

            self.conv2 = nn.Conv1d(
                in_channels=self.d_model,
                out_channels=self.d_model,
                kernel_size=7,
                stride=1,
                padding="same"
            )
            self.norm2 = NormWrapper(
                d_model=self.d_model, 
                num_tokens=input_len//self.patch_stride,
                along_tokens=norm_along_tokens,
                norm_type=norm_type,
                bc_combined=True
            )
            self.act2 = nn.GELU()

            self.conv3 = nn.Conv1d(
                in_channels=self.d_model,
                out_channels=self.d_model,
                kernel_size=7,
                stride=1,
                padding="same"
            )
        else:
            self.patcher = nn.Conv1d( 
                in_channels=self.d_in,
                out_channels=self.d_model,
                kernel_size=self.patch_size,
                stride=self.patch_stride
            )

    def forward(self, x): # x in B*C, D, L
        
        if not self.use_alternative:
            x = self.patcher(x)
        else:
            x = self.act1(self.norm1(self.patcher(x)))
            x = self.act2(self.norm2(self.conv2(x)))
            x = self.conv3(x)

        return x
        

class EmbedDS(nn.Module):
    """
    Projects the dimension of the model, and optionally downsamples.
    """
    def __init__(
        self,
        in_channels,
        out_channels,
        stride,
        kernel_size=1
    ):
        super().__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.stride = stride

        self.embed_ds = nn.Conv1d(
            in_channels=self.in_channels,
            out_channels=self.out_channels,
            kernel_size=kernel_size, 
            stride=self.stride
        )

    def forward(self, x):
        return self.embed_ds(x)


class Block(nn.Module): # Applied SSM (Mamba) followed by 2 way EinFFT or Pointwise convs
    def __init__(
        self,
        in_channels, # The number of variates
        ssm_type, 
        cm_type,
        num_tokens,
        embed_dim,
        norm_along_tokens,
        drop_path,
        prenorm=True, 
        ssm_dropout=0.0,
        norm_type='layernorm',
        d_state=16
    ):
        super().__init__()

        self.prenorm = prenorm
        self.ssm_dropout = ssm_dropout

        self.norm1 = NormWrapper(d_model=embed_dim, num_tokens=num_tokens, along_tokens=norm_along_tokens, norm_type=norm_type)
        if ssm_type == "mamba":
            self.ssm = Mamba(
                d_model=embed_dim,
                d_state=d_state,
                d_conv=4,
                expand=2
            )
        elif ssm_type == "mamba2":
            self.ssm = Mamba2(
                d_model=embed_dim,
                d_state=d_state,
                d_conv=4,
                expand=2
            )
        else:
            raise ValueError(f"{ssm_type} not yet implemented.")
        

        self.norm2 = NormWrapper(d_model=embed_dim, num_tokens=num_tokens, along_tokens=norm_along_tokens, norm_type=norm_type)
        if cm_type == "EinFFTInd":
            self.cm = EinFFTInd(in_channels=in_channels, d_model=embed_dim)
        elif cm_type == "PWInd":
            self.cm = PWInd(in_channels=in_channels, d_model=embed_dim)
        elif cm_type == "PW":
            self.cm = PW(in_channels=in_channels, d_model=embed_dim)

        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, x):
        
        # x shape [B, C, D, L]
        B, C, D, L = x.shape

        # Mamba (norm -> mamba)
        residual = x
        if self.prenorm:
            x = self.norm1(x)

        x = x.reshape(B * C, D, L).permute(0, 2, 1)
        x = self.ssm(x)
        x = F.dropout(x, p=self.ssm_dropout, training=self.training)
        x = x.permute(0, 2, 1).reshape(B, C, D, L)

        x = residual + self.drop_path(x)
        if not self.prenorm:
            x = self.norm1(x)

        # Channel and feature mixing
        residual = x
        if self.prenorm:
            x = self.norm2(x)
        x = self.cm(x)
        x = residual + self.drop_path(x)
        if not self.prenorm:
            x = self.norm2(x)

        return x


class NormWrapper(nn.Module): 
    def __init__(
        self,
        d_model, # Either the number of tokens, or the dim of the model
        num_tokens,
        along_tokens,
        norm_type,
        bc_combined=False
    ):
        super().__init__()

        assert norm_type in ['layernorm', 'batchnorm']

        self.norm_type = norm_type
        self.along_tokens = along_tokens
        self.bc_combined = bc_combined

        if self.along_tokens and norm_type == 'layernorm':
            self.norm_dim = num_tokens
        else:
            self.norm_dim = d_model
        
        if norm_type == 'layernorm':
            self.norm = nn.LayerNorm(self.norm_dim)
        elif norm_type == 'batchnorm':
            self.norm = nn.BatchNorm1d(self.norm_dim)

    def forward(self, x):
        
        # X is shape B, C, D, L
        if not self.bc_combined:
            B, C, D, L = x.shape

        if self.norm_type == 'layernorm' and self.along_tokens:
            nd = x.shape[-1]
            assert nd == self.norm_dim
            return self.norm(x)
        
        elif self.norm_type == 'layernorm' and not self.along_tokens:
            if not self.bc_combined:
                x = x.reshape(B*C, D, L)
            x = x.permute(0, 2, 1) # [BC, L, D]
            x = self.norm(x)
            x = x.permute(0, 2, 1)
            if not self.bc_combined:
                x = x.reshape(B, C, D, L)
            return x
        
        elif self.norm_type == 'batchnorm':
            if not self.bc_combined:
                x = x.reshape(B*C, D, L)
            x = self.norm(x)
            if not self.bc_combined:
                x = x.reshape(B, C, D, L)
            return x

class SSFormerv2(nn.Module):
    def __init__(
            self,
            in_channels, # Number of electrodes
            d_in, # 1
            out_dims=3,
            patch_dim=8,
            patch_size=8,
            patch_stride=8,
            seq_length=2000, 
            norm_type='layernorm',
            norm_along_tokens=True, # setting this to true normalizes along the token dimension (each channel/feature has its mean independently set to 0 and scaled by std)
            ssm_type='mamba',
            cm_type='EinFFTInd',
            embed_dims=[64, 128],
            depths=[1, 2],
            prenorm=True,
            ds_ratio=2,
            drop_path_rate=0.05,
            dropout_1d_rate=0.05,
            pool_type='mean',
            separable_patcher=False,
            ssm_dropout=0.2,
            fc_layers=2,
            fc_dropout=0.3,
            mamba_d_state=16,
            use_alternative=False,
            ds_kernel_size=1,
            use_age='no',
            **kwargs
        ):

        super().__init__()

        self.in_channels = in_channels
        self.d_in = d_in
        self.patch_dim = patch_dim
        self.patch_size = patch_size
        self.patch_stride = patch_stride
        self.seq_len = seq_length
        self.embed_dims = embed_dims
        self.dropout_1d_rate = dropout_1d_rate
        self.prenorm = prenorm
        self.norm_along_tokens = norm_along_tokens
        self.pool_type = pool_type
        self.depths = depths
        self.use_age = use_age

        # Converts input from B, C, d_in, L to B, C, D, N
        self.patcher = Patcher(
            d_in=self.d_in,
            d_model=self.patch_dim,
            patch_size=self.patch_size,
            patch_stride=self.patch_stride,
            separable=separable_patcher,
            input_len=seq_length,
            use_alternative=use_alternative,
            norm_along_tokens=norm_along_tokens
        ) 
        
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))] 
        self.embed_ds_layers = nn.ModuleList()
        self.blocks = nn.ModuleList()
        self.norms = nn.ModuleList()
        cur_depth = 0
        for i in range(len(self.embed_dims) + 1): # The +1 is to account for the patcher
            if i == 0:
                self.embed_ds_layers.append(self.patcher)
                num_tokens = calculate_output_length(self.seq_len, self.patch_size, self.patch_stride, padding=0, dilation=1)
                self.norms.append(NormWrapper(self.patch_dim, num_tokens, along_tokens=norm_along_tokens, norm_type=norm_type)) 
                out_dim = self.patch_dim
            else:
                if i == 1:
                    in_dims = self.patch_dim
                else:
                    in_dims = embed_dims[i-2]
                self.embed_ds_layers.append(
                    EmbedDS(
                        in_channels=in_dims,
                        out_channels=self.embed_dims[i-1],
                        stride=ds_ratio,
                        kernel_size=ds_kernel_size
                    )
                )
                num_tokens = calculate_output_length(num_tokens, ds_kernel_size, ds_ratio, padding=0, dilation=1)
                self.norms.append(NormWrapper(self.embed_dims[i-1], num_tokens=num_tokens, along_tokens=norm_along_tokens, norm_type=norm_type))
                out_dim = self.embed_dims[i-1]

            blks = nn.ModuleList()
            for d in range(depths[i]):
                
                b = Block(
                    in_channels=self.in_channels,
                    ssm_type=ssm_type,
                    cm_type=cm_type,
                    num_tokens=num_tokens,
                    embed_dim=out_dim,
                    norm_along_tokens=norm_along_tokens,
                    drop_path=dpr[cur_depth],
                    prenorm=prenorm,
                    ssm_dropout=ssm_dropout,
                    norm_type=norm_type,
                    d_state=mamba_d_state
                )
                blks.append(b)
                cur_depth += 1 # To keep track of dpr level
            self.blocks.append(blks)

        if self.pool_type == "mean":
            self.final_pool = nn.AdaptiveAvgPool1d(1)
        elif self.pool_type == "max":
            self.final_pool = nn.AdaptiveMaxPool1d(1)


        head_modules = []
        pooled_dims = self.in_channels*self.embed_dims[-1]

        if self.use_age == 'fc':
            pooled_dims = pooled_dims + 1

        for i in range(fc_layers - 1):
            layer = nn.Sequential(
                nn.Linear(pooled_dims, pooled_dims // 2, bias=False),
                nn.LayerNorm(pooled_dims // 2), 
                nn.GELU(),
                nn.Dropout(p=fc_dropout),
            )
            pooled_dims = pooled_dims // 2
            head_modules.append(layer)
        head_modules.append(nn.Linear(pooled_dims, out_dims))
        self.head = nn.Sequential(*head_modules)



    def forward(self, x, age=None):

        if self.d_in == 5:
            B, C, D, L = x.shape
        elif self.d_in == 1:
            x = x.unsqueeze(-2)
        else:
            raise ValueError(f"{self.d_in} is not a valid d_in value.")
        
        for i in range(len(self.embed_dims) + 1):

            B, C, D, L = x.shape
            x = x.reshape(B*C, D, L)

            # Patch/Downsample
            if i == 0: # Dont dropout1d after patcher layer
                x = self.embed_ds_layers[i](x)
            else:
                x = F.dropout1d(x, p=self.dropout_1d_rate, training=self.training)
                x = self.embed_ds_layers[i](x)

            _, D, N = x.shape
            x = x.reshape(B, C, D, N)

            if not self.prenorm:
                x = self.norms[i](x)

            # Block
            for d in range(self.depths[i]):
                x = self.blocks[i][d](x) # 

        B, C, D, N = x.shape

        # Pooling and norm
        x = x.reshape(B, C*D, N)

        # Classification
        x = self.final_pool(x)
        x = x.reshape(B, -1)

        if self.use_age == 'fc':
            x = torch.cat((x, age.reshape(-1, 1)), dim=1)

        x = self.head(x)

        return x