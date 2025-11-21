import torch
import torch.nn as nn
import torch.utils.checkpoint as checkpoint
from timm.models.layers import DropPath, to_2tuple, trunc_normal_
import torch.nn.functional as F
from einops import rearrange, repeat
import math
from utils import grid_sample

#########################################

class SepConv2d(torch.nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 stride=1,
                 padding=0,
                 dilation=1,act_layer=nn.ReLU):
        super(SepConv2d, self).__init__()
        self.depthwise = torch.nn.Conv2d(in_channels,
                                         in_channels,
                                         kernel_size=kernel_size,
                                         stride=stride,
                                         padding=padding,
                                         dilation=dilation,
                                         groups=in_channels)
        self.pointwise = torch.nn.Conv2d(in_channels, out_channels, kernel_size=1)
        self.act_layer = act_layer() if act_layer is not None else nn.Identity()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride

    def forward(self, x):
        x = self.depthwise(x)
        x = self.act_layer(x)
        x = self.pointwise(x)
        return x

    def flops(self, H, W): 
        flops = 0
        flops += H*W*self.in_channels*self.kernel_size**2/self.stride**2
        flops += H*W*self.in_channels*self.out_channels
        return flops

##########################################################################
## Channel Attention Layer
class CALayer(nn.Module):
    def __init__(self, channel, reduction=16, bias=False):
        super(CALayer, self).__init__()
        # global average pooling: feature --> point
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        # feature channel downscale and upscale --> channel weight
        self.conv_du = nn.Sequential(
                nn.Conv2d(channel, channel // reduction, 1, padding=0, bias=bias),
                nn.ReLU(inplace=True),
                nn.Conv2d(channel // reduction, channel, 1, padding=0, bias=bias),
                nn.Sigmoid()
        )

    def forward(self, x):
        y = self.avg_pool(x)
        y = self.conv_du(y)
        return x * y

def conv(in_channels, out_channels, kernel_size, bias=False, stride = 1):
    return nn.Conv2d(
        in_channels, out_channels, kernel_size,
        padding=(kernel_size//2), bias=bias, stride = stride)

##########################################################################
## Channel Attention Block (CAB)
class CAB(nn.Module):
    def __init__(self, n_feat, kernel_size, reduction, bias, act):
        super(CAB, self).__init__()
        modules_body = []
        modules_body.append(conv(n_feat, n_feat, kernel_size, bias=bias))
        modules_body.append(act)
        modules_body.append(conv(n_feat, n_feat, kernel_size, bias=bias))

        self.CA = CALayer(n_feat, reduction, bias=bias)
        self.body = nn.Sequential(*modules_body)

    def forward(self, x):
        res = self.body(x)
        res = self.CA(res)
        res += x
        return res

#########################################
######## Embedding for q,k,v ########
class ConvProjection(nn.Module):
    def __init__(self, dim, heads = 8, dim_head = 64, kernel_size=3, q_stride=1, k_stride=1, v_stride=1, dropout = 0.,
                 last_stage=False,bias=True):

        super().__init__()

        inner_dim = dim_head *  heads
        self.heads = heads
        pad = (kernel_size - q_stride)//2
        self.to_q = SepConv2d(dim, inner_dim, kernel_size, q_stride, pad, bias)
        self.to_k = SepConv2d(dim, inner_dim, kernel_size, k_stride, pad, bias)
        self.to_v = SepConv2d(dim, inner_dim, kernel_size, v_stride, pad, bias)

    def forward(self, x, attn_kv=None):
        b, n, c, h = *x.shape, self.heads
        l = int(math.sqrt(n))
        w = int(math.sqrt(n))

        attn_kv = x if attn_kv is None else attn_kv
        x = rearrange(x, 'b (l w) c -> b c l w', l=l, w=w)
        attn_kv = rearrange(attn_kv, 'b (l w) c -> b c l w', l=l, w=w)
        # print(attn_kv)
        q = self.to_q(x)
        q = rearrange(q, 'b (h d) l w -> b h (l w) d', h=h)
        
        k = self.to_k(attn_kv)
        v = self.to_v(attn_kv)
        k = rearrange(k, 'b (h d) l w -> b h (l w) d', h=h)
        v = rearrange(v, 'b (h d) l w -> b h (l w) d', h=h)
        return q,k,v    
    
    def flops(self, H, W): 
        flops = 0
        flops += self.to_q.flops(H, W)
        flops += self.to_k.flops(H, W)
        flops += self.to_v.flops(H, W)
        return flops


def lambda_init_fn(depth):
    return 0.8 - 0.6 * math.exp(-0.3 * depth)

class DifferentialLinearProjection(nn.Module):
    """模態特定的 Differential Linear Projection"""
    def __init__(self, dim, heads=8, dim_head=64, dropout=0., bias=True):
        super().__init__()
        self.head_dim = dim_head
        inner_dim = dim_head * heads
        self.heads = heads
        
        # Q 投影保持不變
        self.to_q = nn.Linear(dim, inner_dim, bias=bias)
        
        # 分別為兩個模態創建 KV 投影
        self.to_kv_geometric = nn.Linear(dim, inner_dim * 2, bias=bias)  # 幾何分支的 KV
        self.to_kv_semantic = nn.Linear(dim, inner_dim * 2, bias=bias)   # 語義分支的 KV
        
        # 模態特徵投影層 - 將不同維度特徵投影到統一維度
        self.geo_proj = nn.Linear(3, dim, bias=bias)  # 假設幾何特徵是3維
        self.dino_proj = nn.Linear(1024, dim, bias=bias)  # 假設DINO特徵是1024維
        
        # 可學習的融合權重
        self.geo_weight = nn.Parameter(torch.tensor(0.1))
        self.sem_weight = nn.Parameter(torch.tensor(0.1))
        
        self.dim = dim
        self.inner_dim = inner_dim
        print("Modal-specific differential transformer initialized!")
    
    def forward(self, x, geo_feat, dino_feat, attn_kv=None):
        B_, N, C = x.shape
        attn_kv = x if attn_kv is None else attn_kv
        
        # Q 保持原樣
        q = self.to_q(x).reshape(B_, N, 1, self.heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q = q[0]  # [B_, heads, N, head_dim]
        
        # 投影模態特徵到統一維度
        geo_feat_proj = self.geo_proj(geo_feat)  # [B_, N, dim]
        dino_feat_proj = self.dino_proj(dino_feat)  # [B_, N, dim]
        
        # 簡單的特徵融合 - 加權相加
        geo_enhanced = attn_kv + self.geo_weight * geo_feat_proj
        semantic_enhanced = attn_kv + self.sem_weight * dino_feat_proj
        
        # 分別計算兩組 KV
        kv_geo = self.to_kv_geometric(geo_enhanced).reshape(B_, N, 2, self.heads, self.head_dim).permute(2, 0, 3, 1, 4)
        kv_sem = self.to_kv_semantic(semantic_enhanced).reshape(B_, N, 2, self.heads, self.head_dim).permute(2, 0, 3, 1, 4)
        
        # 組合成最終的 KV
        # kv[0] = 幾何增強的 KV, kv[1] = 語義增強的 KV
        k = torch.stack([kv_geo[0], kv_sem[0]], dim=0)  # [2, B_, heads, N, head_dim]
        v = torch.stack([kv_geo[1], kv_sem[1]], dim=0)  # [2, B_, heads, N, head_dim]
        
        return q, k, v


class DifferentialLinearProjection_Concat_kv(nn.Module):
    """Concat 版本的模態特定 Differential Linear Projection"""
    def __init__(self, dim, heads=8, dim_head=64, dropout=0., bias=True):
        super().__init__()
        self.head_dim = dim_head
        inner_dim = dim_head * heads
        self.heads = heads
        
        # 基礎 QKV 投影
        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=bias)
        
        # 兩個模態的額外 KV 投影
        self.to_kv_geometric = nn.Linear(dim, inner_dim * 2, bias=bias)
        self.to_kv_semantic = nn.Linear(dim, inner_dim * 2, bias=bias)
        
        # 模態特徵投影層
        self.geo_proj = nn.Linear(3, dim, bias=bias)
        self.dino_proj = nn.Linear(1024, dim, bias=bias)
        
        # 可學習的融合權重
        self.geo_weight = nn.Parameter(torch.tensor(0.1))
        self.sem_weight = nn.Parameter(torch.tensor(0.1))
        
        self.dim = dim
        self.inner_dim = inner_dim

    def forward(self, x, geo_feat, dino_feat, attn_kv=None):
        B_, N, C = x.shape
        attn_kv = x if attn_kv is None else attn_kv
        
        # 基礎的 QKV
        qkv_dec = self.to_qkv(x).reshape(B_, N, 3, self.heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k_d, v_d = qkv_dec[0], qkv_dec[1], qkv_dec[2]
        
        # 投影模態特徵到統一維度
        geo_feat_proj = self.geo_proj(geo_feat)
        dino_feat_proj = self.dino_proj(dino_feat)
        
        # 簡單的特徵融合
        geo_enhanced = attn_kv + self.geo_weight * geo_feat_proj
        semantic_enhanced = attn_kv + self.sem_weight * dino_feat_proj
        
        # 分別計算兩組額外的 KV
        kv_geo = self.to_kv_geometric(geo_enhanced).reshape(B_, N, 2, self.heads, self.head_dim).permute(2, 0, 3, 1, 4)
        kv_sem = self.to_kv_semantic(semantic_enhanced).reshape(B_, N, 2, self.heads, self.head_dim).permute(2, 0, 3, 1, 4)
        
        k_geo, v_geo = kv_geo[0], kv_geo[1]
        k_sem, v_sem = kv_sem[0], kv_sem[1]
        
        # 拼接：[基礎KV, 幾何KV, 語義KV]
        k = torch.cat((k_d, k_geo, k_sem), dim=2)
        v = torch.cat((v_d, v_geo, v_sem), dim=2)
        
        return q, k, v

class DifferentialWindowAttention(nn.Module):
    def __init__(self, dim, win_size, num_heads, depth=1, token_projection='linear', 
                 qkv_bias=True, qk_scale=None, attn_drop=0., proj_drop=0., se_layer=False,
                 geo_dim=3, dino_dim=1024):  # 新增參數
        super().__init__()
        self.dim = dim
        self.win_size = win_size
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = qk_scale or self.head_dim ** -0.5
        
        # 預先定義特徵投影層
        self.geo_dim = geo_dim
        self.dino_dim = dino_dim
        self.geo_adaptive_proj = nn.Linear(geo_dim, 3) if geo_dim != 3 else nn.Identity()
        self.dino_adaptive_proj = nn.Linear(dino_dim, 1024) if dino_dim != 1024 else nn.Identity()

        # Differential parameters
        self.lambda_init = lambda_init_fn(depth)
        self.lambda_q1 = nn.Parameter(torch.ones(1) * 0.5)
        self.lambda_k1 = nn.Parameter(torch.ones(1) * 0.5)
        self.lambda_q2 = nn.Parameter(torch.ones(1) * 0.5)
        self.lambda_k2 = nn.Parameter(torch.ones(1) * 0.5)

        self.subln = nn.LayerNorm(dim)

        # 使用模態特定的投影層
        if token_projection == 'linear_concat':
            self.qkv = DifferentialLinearProjection_Concat_kv(dim, num_heads, dim // num_heads, bias=qkv_bias)
        else:
            self.qkv = DifferentialLinearProjection(dim, num_heads, dim // num_heads, bias=qkv_bias)

        # 相對位置編碼
        self.relative_position_bias_table = nn.Parameter(
            torch.zeros((2 * win_size[0] - 1) * (2 * win_size[1] - 1), num_heads))

        coords_h = torch.arange(self.win_size[0])
        coords_w = torch.arange(self.win_size[1])
        coords = torch.stack(torch.meshgrid([coords_h, coords_w], indexing='ij'))
        coords_flatten = torch.flatten(coords, 1)
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()
        relative_coords[:, :, 0] += self.win_size[0] - 1
        relative_coords[:, :, 1] += self.win_size[1] - 1
        relative_coords[:, :, 0] *= 2 * self.win_size[1] - 1
        relative_position_index = relative_coords.sum(-1)
        self.register_buffer("relative_position_index", relative_position_index)

        self.token_projection = token_projection
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        nn.init.trunc_normal_(self.relative_position_bias_table, std=.02)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x, dino_mat, point_feature, normal, attn_kv=None, mask=None):
        B_, N, C = x.shape
        
        # 準備模態特徵
        dino_mat = self.dino_adaptive_proj(dino_mat)
        point_feature = self.geo_adaptive_proj(point_feature)
        
        geo_feat = point_feature
        dino_feat = dino_mat
        
        # QKV 投影
        q, k, v = self.qkv(x, geo_feat, dino_feat, attn_kv)
        q = q * self.scale
        
        # k, v 格式: [2, B_, heads, N, head_dim]
        k_geo, k_sem = k[0], k[1]  # 幾何和語義的 K
        v_geo, v_sem = v[0], v[1]  # 幾何和語義的 V
        
        # 所有 heads 都計算兩種注意力
        attn_geo = torch.matmul(q, k_geo.transpose(-2, -1))  # [B_, heads, N, N]
        attn_sem = torch.matmul(q, k_sem.transpose(-2, -1))  # [B_, heads, N, N]
        
        # 添加相對位置偏置
        relative_position_bias = self.relative_position_bias_table[
            self.relative_position_index.view(-1)
        ].view(self.win_size[0] * self.win_size[1], self.win_size[0] * self.win_size[1], -1)
        relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()
        
        ratio = attn_geo.size(-1) // relative_position_bias.size(-1)
        if ratio > 1:
            relative_position_bias = repeat(relative_position_bias, 'nH l c -> nH l (c d)', d=ratio)
        
        attn_geo = attn_geo + relative_position_bias.unsqueeze(0)
        attn_sem = attn_sem + relative_position_bias.unsqueeze(0)
        
        # 處理 mask
        if mask is not None:
            nW = mask.shape[0]
            mask = repeat(mask, 'nW m n -> nW m (n d)', d=ratio)
            attn_geo = attn_geo.view(B_ // nW, nW, self.num_heads, N, N * ratio) + mask.unsqueeze(1).unsqueeze(0)
            attn_sem = attn_sem.view(B_ // nW, nW, self.num_heads, N, N * ratio) + mask.unsqueeze(1).unsqueeze(0)
            attn_geo = attn_geo.view(-1, self.num_heads, N, N * ratio)
            attn_sem = attn_sem.view(-1, self.num_heads, N, N * ratio)
        
        # Softmax
        attn_geo = self.softmax(attn_geo)
        attn_sem = self.softmax(attn_sem)
        
        # Differential Attention: 用語義減去幾何
        lambda_val = torch.sigmoid(self.lambda_q1 * self.lambda_k1) + self.lambda_init
        attn_diff = attn_sem - lambda_val * attn_geo
        
        # 應用注意力
        attn_geo = self.attn_drop(attn_geo)
        attn_diff = self.attn_drop(attn_diff)
        
        # 兩個分支的輸出
        x_geo = torch.matmul(attn_geo, v_geo)   # 幾何分支
        x_diff = torch.matmul(attn_diff, v_sem)  # 差異分支
        
        # 加權融合
        x = x_geo + x_diff  # 或者用可學習權重
        x = x.transpose(1, 2).contiguous().view(B_, N, C)
        x = self.subln(x)
        x = x * (1 - self.lambda_init)
        x = self.proj(x)
        x = self.proj_drop(x)
        
        return x

    def extra_repr(self) -> str:
        return f'dim={self.dim}, win_size={self.win_size}, num_heads={self.num_heads}, ' \
               f'head_dim={self.head_dim}, lambda_init={self.lambda_init:.3f}'



#########################################
########### feed-forward network #############
class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)
        self.in_features = in_features
        self.hidden_features = hidden_features
        self.out_features = out_features

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x

    def flops(self, H, W):
        flops = 0
        # fc1
        flops += H*W*self.in_features*self.hidden_features 
        # fc2
        flops += H*W*self.hidden_features*self.out_features
        print("MLP:{%.2f}"%(flops/1e9))
        return flops

class LeFF(nn.Module):
    def __init__(self, dim=32, hidden_dim=128, act_layer=nn.GELU,drop = 0.):
        super().__init__()
        self.linear1 = nn.Sequential(nn.Linear(dim, hidden_dim),
                                act_layer())
        self.dwconv = nn.Sequential(nn.Conv2d(hidden_dim,hidden_dim,groups=hidden_dim,kernel_size=3,stride=1,padding=1),
                        act_layer())
        self.linear2 = nn.Sequential(nn.Linear(hidden_dim, dim))
        self.dim = dim
        self.hidden_dim = hidden_dim

    def forward(self, x, img_size=(128,128)):
        # bs x hw x c
        bs, hw, c = x.size()
        # hh = int(math.sqrt(hw))
        hh = img_size[0]
        ww = img_size[1]

        x = self.linear1(x)

        # spatial restore
        x = rearrange(x, ' b (h w) (c) -> b c h w ', h = hh, w = ww)
        # bs,hidden_dim,32x32

        x = self.dwconv(x)

        # flaten
        x = rearrange(x, ' b c h w -> b (h w) c', h = hh, w = ww)

        x = self.linear2(x)

        return x

    def flops(self, H, W):
        flops = 0
        # fc1
        flops += H*W*self.dim*self.hidden_dim 
        # dwconv
        flops += H*W*self.hidden_dim*3*3
        # fc2
        flops += H*W*self.hidden_dim*self.dim
        print("LeFF:{%.2f}"%(flops/1e9))
        return flops

#########################################
########### window operation#############
def window_partition(x, win_size, dilation_rate=1):
    B, H, W, C = x.shape
    if dilation_rate !=1:
        x = x.permute(0,3,1,2).contiguous() # B, C, H, W
        assert type(dilation_rate) is int, 'dilation_rate should be a int'
        x = F.unfold(x, kernel_size=win_size,dilation=dilation_rate,padding=4*(dilation_rate-1),stride=win_size) # B, C*Wh*Ww, H/Wh*W/Ww
        windows = x.permute(0,2,1).contiguous().view(-1, C, win_size, win_size) # B' ,C ,Wh ,Ww
        windows = windows.permute(0,2,3,1).contiguous() # B' ,Wh ,Ww ,C
    else:
        x = x.view(B, H // win_size, win_size, W // win_size, win_size, C)
        windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, win_size, win_size, C) # B' ,Wh ,Ww ,C
    return windows

def window_reverse(windows, win_size, H, W, dilation_rate=1):
    # B' ,Wh ,Ww ,C
    B = int(windows.shape[0] / (H * W / win_size / win_size))
    x = windows.view(B, H // win_size, W // win_size, win_size, win_size, -1)
    if dilation_rate !=1:
        x = windows.permute(0,5,3,4,1,2).contiguous() # B, C*Wh*Ww, H/Wh*W/Ww
        x = F.fold(x, (H, W), kernel_size=win_size, dilation=dilation_rate, padding=4*(dilation_rate-1),stride=win_size)
    else:
        x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, H, W, -1)
    return x

#########################################
# Downsample Block
class Downsample(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(Downsample, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channel, out_channel, kernel_size=4, stride=2, padding=1)
            # nn.Conv2d(in_channel * 4, out_channel, kernel_size=3, padding=1)
        )
        self.in_channel = in_channel
        self.out_channel = out_channel

    def forward(self, x, img_size=(128,128)):
        B, L, C = x.shape
        H = img_size[0]
        W = img_size[1]
        x = x.transpose(1, 2).contiguous().view(B, C, H, W)

        out = self.conv(x).flatten(2).transpose(1,2).contiguous()  # B H*W C
        return out

    def flops(self, H, W):
        flops = 0
        # conv
        flops += H/2*W/2*self.in_channel*self.out_channel*4*4
        print("Downsample:{%.2f}"%(flops/1e9))
        return flops

# Upsample Block
class Upsample(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(Upsample, self).__init__()
        self.deconv = nn.Sequential(
            nn.ConvTranspose2d(in_channel, out_channel, kernel_size=2, stride=2)
        )

        self.in_channel = in_channel
        self.out_channel = out_channel

    def forward(self, x, img_size=(128,128)):
        B, L, C = x.shape
        H = img_size[0]
        W = img_size[1]
        x = x.transpose(1, 2).contiguous().view(B, C, H, W)
        out = self.deconv(x)

        out = out.flatten(2).transpose(1,2).contiguous() # B H*W C
        return out

    def flops(self, H, W):
        flops = 0
        # conv
        flops += H*2*W*2*self.in_channel*self.out_channel*2*2 
        print("Upsample:{%.2f}"%(flops/1e9))
        return flops

# Input Projection
class InputProj(nn.Module):
    def __init__(self, in_channel=3, out_channel=64, kernel_size=3, stride=1, norm_layer=None,act_layer=nn.LeakyReLU):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Conv2d(in_channel, out_channel, kernel_size=3, stride=stride, padding=kernel_size//2),
            act_layer(inplace=True)
        )
        if norm_layer is not None:
            self.norm = norm_layer(out_channel)
        else:
            self.norm = None
        self.in_channel = in_channel
        self.out_channel = out_channel

    def forward(self, x):
        B, C, H, W = x.shape
        x = self.proj(x).flatten(2).transpose(1, 2).contiguous()  # B H*W C
        if self.norm is not None:
            x = self.norm(x)
        return x

    def flops(self, H, W):
        flops = 0
        # conv
        flops += H*W*self.in_channel*self.out_channel*3*3

        if self.norm is not None:
            flops += H*W*self.out_channel 
        print("Input_proj:{%.2f}"%(flops/1e9))
        return flops

# Output Projection
class OutputProj(nn.Module):
    def __init__(self, in_channel=64, out_channel=3, kernel_size=3, stride=1, norm_layer=None,act_layer=None):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Conv2d(in_channel, out_channel, kernel_size=3, stride=stride, padding=kernel_size//2),
        )
        if act_layer is not None:
            self.proj.add_module(act_layer(inplace=True))
        if norm_layer is not None:
            self.norm = norm_layer(out_channel)
        else:
            self.norm = None
        self.in_channel = in_channel
        self.out_channel = out_channel

    def forward(self, x, img_size=(128,128)):
        B, L, C = x.shape
        H = img_size[0]
        W = img_size[1]
        # H = int(math.sqrt(L))
        # W = int(math.sqrt(L))
        x = x.transpose(1, 2).contiguous().view(B, C, H, W)
        x = self.proj(x)
        if self.norm is not None:
            x = self.norm(x)
        return x

    def flops(self, H, W):
        flops = 0
        # conv
        flops += H*W*self.in_channel*self.out_channel*3*3

        if self.norm is not None:
            flops += H*W*self.out_channel 
        print("Output_proj:{%.2f}"%(flops/1e9))
        return flops


class SepConv(nn.Module):
    def __init__(
        self,
        in_channel,
        out_channel,
        kernel_size,
        stride=1,
        bias=True,
        padding_mode="zeros",
    ):
        super().__init__()
        self.conv1 = nn.Conv2d(
            in_channel,
            in_channel,
            kernel_size,
            stride=stride,
            padding=kernel_size // 2,
            groups=in_channel,
            bias=bias,
            padding_mode=padding_mode,
        )
        self.conv2 = nn.Conv2d(
            in_channel, out_channel, kernel_size=1, stride=1, padding=0, bias=bias
        )

    def forward(self, x):
        out = self.conv1(x)
        out = self.conv2(out)
        return out
    
class WaveletEnhanceBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.channels = channels
        ll = torch.tensor([[0.5, 0.5], [0.5, 0.5]])
        lh = torch.tensor([[-0.5, -0.5], [0.5, 0.5]])
        hl = torch.tensor([[-0.5, 0.5], [-0.5, 0.5]])
        hh = torch.tensor([[0.5, -0.5], [-0.5, 0.5]])
        kernel = torch.stack([ll, lh, hl, hh], dim=0).unsqueeze(1)
        kernel = kernel.repeat(channels, 1, 1, 1)
        self.register_buffer("haar_kernel", kernel)

        self.fuse = nn.Conv2d(4 * channels, channels, kernel_size=1, bias=False)
        self.post = SepConv(channels, channels, kernel_size=3, bias=False)

    def forward(self, x):
        B, C, H, W = x.shape
        dwt = F.conv2d(x, self.haar_kernel, stride=2, groups=C)

        fea = self.fuse(dwt)  # -> [B, C, H//2, W//2]
        fea = self.post(fea)  # -> [B, C, H//2, W//2]

        out = F.interpolate(fea, size=(H, W), mode="bilinear", align_corners=False)

        return out
    
#########################################
########### CA Transformer #############
class CA_DWT_TransformerBlock(nn.Module):
    def __init__(self, dim, input_resolution, num_heads, win_size=10, shift_size=0,
                 mlp_ratio=4., qkv_bias=True, qk_scale=None, drop=0., attn_drop=0., drop_path=0.,
                 act_layer=nn.GELU, norm_layer=nn.LayerNorm, token_projection='linear', token_mlp='leff',
                 se_layer=False):
        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.num_heads = num_heads
        self.win_size = win_size
        self.shift_size = shift_size
        self.mlp_ratio = mlp_ratio
        self.token_mlp = token_mlp
        if min(self.input_resolution) <= self.win_size:
            self.shift_size = 0
            self.win_size = min(self.input_resolution)
        assert 0 <= self.shift_size < self.win_size, "shift_size must in 0-win_size"

        self.norm1 = norm_layer(dim)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer,
                       drop=drop) if token_mlp == 'ffn' else LeFF(dim, mlp_hidden_dim, act_layer=act_layer, drop=drop)
        self.CAB = CAB(dim//2, kernel_size=3, reduction=4, bias=False, act=nn.PReLU())
        self.DWT= WaveletEnhanceBlock(dim//2)

    def extra_repr(self) -> str:
        return f"dim={self.dim}, input_resolution={self.input_resolution}, num_heads={self.num_heads}, " \
               f"win_size={self.win_size}, shift_size={self.shift_size}, mlp_ratio={self.mlp_ratio}"

    def forward(self, x, dino_mat, point, normal, mask=None, img_size=(128, 128)):
        B, L, C = x.shape
        H = img_size[0]
        W = img_size[1]
        assert L == W * H, \
            f"Input image size ({H}*{W} doesn't match model ({L})."

        shortcut = x
        x = self.norm1(x)

        # spatial restore
        x = rearrange(x, ' b (h w) (c) -> b c h w ', h=H, w=W)
        # bs,hidden_dim,32x32

        x1, x2 = torch.chunk(x, 2, dim=1)

        x1 = self.CAB(x1)
        x2 = self.DWT(x2)

        
        x = torch.cat([x1, x2], dim=1)

        # flaten
        x = rearrange(x, ' b c h w -> b (h w) c', h=H, w=W)
        x = x.view(B, H * W, C)

        # FFN
        x = shortcut + self.drop_path(x)
        x = x + self.drop_path(self.mlp(self.norm2(x), img_size=img_size))
        
        return x

    def flops(self):
        flops = 0
        H, W = self.input_resolution
        # norm1
        flops += self.dim * H * W
        # W-MSA/SW-MSA
        flops += self.attn.flops(H, W)
        # norm2
        flops += self.dim * H * W
        # mlp
        flops += self.mlp.flops(H, W)
        print("LeWin:{%.2f}" % (flops / 1e9))
        return flops



#########################################
########### SIM Transformer #############

#########################################
########### SIM Transformer #############
class CA_DWT_SIMTransformerBlock(nn.Module):
    def __init__(self, dim, input_resolution, num_heads, win_size=10, shift_size=0,
                 mlp_ratio=4., qkv_bias=True, qk_scale=None, drop=0., attn_drop=0., drop_path=0.,
                 act_layer=nn.GELU, norm_layer=nn.LayerNorm,token_projection='linear',token_mlp='leff',se_layer=False):
        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.num_heads = num_heads
        self.win_size = win_size
        self.shift_size = shift_size
        self.mlp_ratio = mlp_ratio
        self.token_mlp = token_mlp
        if min(self.input_resolution) <= self.win_size:
            self.shift_size = 0
            self.win_size = min(self.input_resolution)
        assert 0 <= self.shift_size < self.win_size, "shift_size must in 0-win_size"

        self.norm1 = norm_layer(dim)
        self.attn = DifferentialWindowAttention(
            dim, win_size=to_2tuple(self.win_size), num_heads=num_heads,
            qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop,
            token_projection=token_projection,se_layer=se_layer)

        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim,act_layer=act_layer, drop=drop) if token_mlp=='ffn' else LeFF(dim,mlp_hidden_dim,act_layer=act_layer, drop=drop)
        #self.CAB = CAB(dim, kernel_size=3, reduction=4, bias=False, act=nn.PReLU())

        self.CAB = CAB(dim//2, kernel_size=3, reduction=4, bias=False, act=nn.PReLU())
        self.DWT= WaveletEnhanceBlock(dim//2)

    def extra_repr(self) -> str:
        return f"dim={self.dim}, input_resolution={self.input_resolution}, num_heads={self.num_heads}, " \
               f"win_size={self.win_size}, shift_size={self.shift_size}, mlp_ratio={self.mlp_ratio}"

    def forward(self, x, dino_mat, point, normal, mask=None, img_size = (128, 128)):
        B, L, C = x.shape
        H = img_size[0]
        W = img_size[1]
        assert L == W * H, \
            f"Input image size ({H}*{W} doesn't match model ({L})."

        C_dino_mat = dino_mat.shape[1]
        C_point = point.shape[1]
        C_normal = normal.shape[1]

        if mask != None:
            input_mask = F.interpolate(mask, size=(H,W)).permute(0,2,3,1).contiguous()
            input_mask_windows = window_partition(input_mask, self.win_size) # nW, win_size, win_size, 1
            attn_mask = input_mask_windows.view(-1, self.win_size * self.win_size) # nW, win_size*win_size
            attn_mask = attn_mask.unsqueeze(2)*attn_mask.unsqueeze(1) # nW, win_size*win_size, win_size*win_size
            attn_mask = attn_mask.masked_fill(attn_mask!=0, float(-100.0)).masked_fill(attn_mask == 0, float(0.0))
        else:
            attn_mask = None

        ## shift mask
        if self.shift_size > 0:
            # calculate attention mask for SW-MSA
            shift_mask = torch.zeros((1, H, W, 1)).type_as(x)
            h_slices = (slice(0, -self.win_size),
                        slice(-self.win_size, -self.shift_size),
                        slice(-self.shift_size, None))
            w_slices = (slice(0, -self.win_size),
                        slice(-self.win_size, -self.shift_size),
                        slice(-self.shift_size, None))
            cnt = 0
            for h in h_slices:
                for w in w_slices:
                    shift_mask[:, h, w, :] = cnt
                    cnt += 1
            shift_mask_windows = window_partition(shift_mask, self.win_size)  # nW, win_size, win_size, 1
            shift_mask_windows = shift_mask_windows.view(-1, self.win_size * self.win_size) # nW, win_size*win_size
            shift_attn_mask = shift_mask_windows.unsqueeze(1) - shift_mask_windows.unsqueeze(2) # nW, win_size*win_size, win_size*win_size
            shift_attn_mask = shift_attn_mask.masked_fill(shift_attn_mask != 0, float(-100.0)).masked_fill(shift_attn_mask == 0, float(0.0))
            attn_mask = attn_mask + shift_attn_mask if attn_mask is not None else shift_attn_mask

        shortcut = x
        x = self.norm1(x)


        x = x.view(B, H, W, C)
        dino_mat = dino_mat.permute(0, 2, 3, 1).contiguous()
        point = point.permute(0, 2, 3, 1).contiguous()
        normal = normal.permute(0, 2, 3, 1).contiguous()

        # cyclic shift
        if self.shift_size > 0:
            shifted_x = torch.roll(x, shifts=(-self.shift_size, -self.shift_size), dims=(1, 2))
            shifted_dino_mat = torch.roll(dino_mat, shifts=(-self.shift_size, -self.shift_size), dims=(1, 2))
            shifted_point = torch.roll(point, shifts=(-self.shift_size, -self.shift_size), dims=(1, 2))
            shifted_normal = torch.roll(normal, shifts=(-self.shift_size, -self.shift_size), dims=(1, 2))
        else:
            shifted_x = x
            shifted_dino_mat = dino_mat
            shifted_point = point
            shifted_normal = normal

        x_windows = window_partition(shifted_x, self.win_size)  # nW*B, win_size, win_size, C  N*C->C
        x_windows = x_windows.view(-1, self.win_size * self.win_size, C)  # nW*B, win_size*win_size, C


        dino_mat_windows = window_partition(shifted_dino_mat, self.win_size)  # nW*B, win_size, win_size, C  N*C->C
        dino_mat_windows = dino_mat_windows.view(-1, self.win_size * self.win_size, C_dino_mat)  # nW*B, win_size*win_size, C

        point_windows = window_partition(shifted_point, self.win_size)  # nW*B, win_size, win_size, C  N*C->C
        point_windows = point_windows.view(-1, self.win_size * self.win_size, C_point)  # nW*B, win_size*win_size, C

        normal_windows = window_partition(shifted_normal, self.win_size)  # nW*B, win_size, win_size, C  N*C->C
        normal_windows = normal_windows.view(-1, self.win_size * self.win_size, C_normal)  # nW*B, win_size*win_size, C

        # W-MSA/SW-MSA
        attn_windows = self.attn(x_windows, dino_mat_windows, point_windows, normal_windows, mask=attn_mask)  # nW*B, win_size*win_size, C


        # merge windows
        attn_windows = attn_windows.view(-1, self.win_size, self.win_size, C)


        shifted_x = window_reverse(attn_windows, self.win_size, H, W)  # B H' W' C

        # reverse cyclic shift
        if self.shift_size > 0:
            x = torch.roll(shifted_x, shifts=(self.shift_size, self.shift_size), dims=(1, 2))
        else:
            x = shifted_x
        x = x.view(B, H * W, C)

        x = rearrange(x, ' b (h w) (c) -> b c h w ', h=H, w=W)
        # bs,hidden_dim,32x32

        x1, x2 = torch.chunk(x, 2, dim=1)

        x1 = self.CAB(x1)
        x2 = self.DWT(x2)

        
        x = torch.cat([x1, x2], dim=1)


        x = rearrange(x, ' b c h w -> b (h w) c', h=H, w=W)

        # FFN
        x = shortcut + self.drop_path(x)
        x = x + self.drop_path(self.mlp(self.norm2(x), img_size=img_size))
        del attn_mask
        return x

    def flops(self):
        flops = 0
        H, W = self.input_resolution
        # norm1
        flops += self.dim * H * W
        # W-MSA/SW-MSA
        flops += self.attn.flops(H, W)
        # norm2
        flops += self.dim * H * W
        # mlp
        flops += self.mlp.flops(H,W)
        print("LeWin:{%.2f}"%(flops/1e9))
        return flops


#########################################
########### Basic layer of ShadowFormer ################
class BasicShadowFormer(nn.Module):
    def __init__(self, dim, output_dim, input_resolution, depth, num_heads, win_size,
                 mlp_ratio=4., qkv_bias=True, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., norm_layer=nn.LayerNorm, use_checkpoint=False,
                 token_projection='linear',token_mlp='ffn',se_layer=False,cab=False):

        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.depth = depth
        self.use_checkpoint = use_checkpoint
        # build blocks
        if cab:
            self.blocks = nn.ModuleList([
                CA_DWT_TransformerBlock(dim=dim, input_resolution=input_resolution,
                                      num_heads=num_heads, win_size=win_size,
                                      shift_size=0 if (i % 2 == 0) else win_size // 2,
                                      mlp_ratio=mlp_ratio,
                                      qkv_bias=qkv_bias, qk_scale=qk_scale,
                                      drop=drop, attn_drop=attn_drop,
                                      drop_path=drop_path[i] if isinstance(drop_path, list) else drop_path,
                                      norm_layer=norm_layer, token_projection=token_projection, token_mlp=token_mlp,
                                      se_layer=se_layer)
                for i in range(depth)])
        else:
            self.blocks = nn.ModuleList([
                CA_DWT_SIMTransformerBlock(dim=dim, input_resolution=input_resolution,
                                     num_heads=num_heads, win_size=win_size,
                                     shift_size=0 if (i % 2 == 0) else win_size // 2,
                                     mlp_ratio=mlp_ratio,
                                     qkv_bias=qkv_bias, qk_scale=qk_scale,
                                     drop=drop, attn_drop=attn_drop,
                                     drop_path=drop_path[i] if isinstance(drop_path, list) else drop_path,
                                     norm_layer=norm_layer,token_projection=token_projection,token_mlp=token_mlp,se_layer=se_layer)
                for i in range(depth)])

    def extra_repr(self) -> str:
        return f"dim={self.dim}, input_resolution={self.input_resolution}, depth={self.depth}"    

    def forward(self, x, dino_mat=None, point=None, normal=None, mask=None, img_size=(128,128)):
        for blk in self.blocks:
            if self.use_checkpoint:
                x = checkpoint.checkpoint(blk, x)
            else:
                x = blk(x, dino_mat, point, normal, mask, img_size)
        return x

    def flops(self):
        flops = 0
        for blk in self.blocks:
            flops += blk.flops()
        return flops

class GrayWorldRetinex(nn.Module):
    def __init__(self, eps=1e-6):
        super(GrayWorldRetinex, self).__init__()
        self.eps = eps

    def forward(self, x):
        B, C, H, W = x.shape
        mean = x.mean(dim=(2, 3), keepdim=True)  # [B, C, 1, 1]
        gray_mean = mean.mean(dim=1, keepdim=True)  # [B, 1, 1, 1]
        gain = gray_mean / (mean + self.eps)
        x = x * gain  # white balance
        x_log = torch.log(x + self.eps)
        x_log = x_log - x_log.mean(dim=(2, 3), keepdim=True)
        x_out = torch.exp(x_log)
        x_min = x_out.amin(dim=(-2, -1), keepdim=True)
        x_max = x_out.amax(dim=(-2, -1), keepdim=True)
        x_out = (x_out - x_min) / (x_max - x_min + self.eps)
        return x_out



def _rgb2luma_tensor(x):
    """x: [B,3,H,W]"""
    return 0.2126*x[:,0:1,:,:] + 0.7152*x[:,1:2,:,:] + 0.0722*x[:,2:3,:,:]
    
class PhaSR(nn.Module):
    def __init__(self, img_size=256, in_chans=3,
                embed_dim=32, depths=[2, 2, 2, 2, 2, 2, 2, 2, 2], 
                num_heads=[1, 2, 4, 8, 16, 16, 8, 4, 2],
                win_size=8, mlp_ratio=4., qkv_bias=True, qk_scale=None,
                drop_rate=0., attn_drop_rate=0., drop_path_rate=0.1,
                norm_layer=nn.LayerNorm, patch_norm=True,
                use_checkpoint=False, token_projection='linear', token_mlp='leff', 
                se_layer=True, dowsample=Downsample, upsample=Upsample, 
                use_white_balance=True, use_axs=True, **kwargs):
        super().__init__()

        self.num_enc_layers = len(depths)//2
        self.num_dec_layers = len(depths)//2
        self.embed_dim = embed_dim
        self.patch_norm = patch_norm
        self.mlp_ratio = mlp_ratio
        self.token_projection = token_projection
        self.mlp = token_mlp
        self.win_size = win_size
        self.reso = img_size
        self.pos_drop = nn.Dropout(p=drop_rate)
        self.DINO_channel = 1024

        # stochastic depth
        enc_dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths[:self.num_enc_layers]))]
        conv_dpr = [drop_path_rate]*depths[4]
        dec_dpr = enc_dpr[::-1]

        # 為每個尺度創建 DINO 特徵融合模塊
        self.dino_proj_0 = nn.Conv2d(1024, embed_dim, kernel_size=1)
        self.dino_proj_1 = nn.Conv2d(1024, embed_dim*2, kernel_size=1)
        self.dino_proj_2 = nn.Conv2d(1024, embed_dim*4, kernel_size=1)
        self.dino_proj_3 = nn.Conv2d(1024, embed_dim*8, kernel_size=1)
        
        # 可學習的融合權重（可選）
        self.alpha_0 = nn.Parameter(torch.tensor(0.1))
        self.alpha_1 = nn.Parameter(torch.tensor(0.1))
        self.alpha_2 = nn.Parameter(torch.tensor(0.1))
        self.alpha_3 = nn.Parameter(torch.tensor(0.1))

        # 多尺度特徵融合權重
        self.fusion_weight_0 = nn.Parameter(torch.tensor(0.1))
        self.fusion_weight_1 = nn.Parameter(torch.tensor(0.1))
        self.fusion_weight_2 = nn.Parameter(torch.tensor(0.1))
        self.fusion_weight_3 = nn.Parameter(torch.tensor(0.1))

        # Input/Output
        self.input_proj = InputProj(in_channel=4, out_channel=embed_dim, kernel_size=3, stride=1, act_layer=nn.LeakyReLU)
        self.output_proj = OutputProj(in_channel=2*embed_dim, out_channel=in_chans, kernel_size=3, stride=1)

        # Encoder
        self.encoderlayer_0 = BasicShadowFormer(dim=embed_dim,
                            output_dim=embed_dim,
                            input_resolution=(img_size, img_size),
                            depth=depths[0],
                            num_heads=num_heads[0],
                            win_size=win_size,
                            mlp_ratio=self.mlp_ratio,
                            qkv_bias=qkv_bias, qk_scale=qk_scale,
                            drop=drop_rate, attn_drop=attn_drop_rate,
                            drop_path=enc_dpr[sum(depths[:0]):sum(depths[:1])],
                            norm_layer=norm_layer,
                            use_checkpoint=use_checkpoint,
                            token_projection=token_projection,token_mlp=token_mlp,se_layer=se_layer,cab=True)
        self.dowsample_0 = dowsample(embed_dim, embed_dim*2)
        
        self.encoderlayer_1 = BasicShadowFormer(dim=embed_dim*2,
                            output_dim=embed_dim*2,
                            input_resolution=(img_size // 2, img_size // 2),
                            depth=depths[1],
                            num_heads=num_heads[1],
                            win_size=win_size,
                            mlp_ratio=self.mlp_ratio,
                            qkv_bias=qkv_bias, qk_scale=qk_scale,
                            drop=drop_rate, attn_drop=attn_drop_rate,
                            drop_path=enc_dpr[sum(depths[:1]):sum(depths[:2])],
                            norm_layer=norm_layer,
                            use_checkpoint=use_checkpoint,
                            token_projection=token_projection,token_mlp=token_mlp,se_layer=se_layer, cab=True)
        self.dowsample_1 = dowsample(embed_dim*2, embed_dim*4)
        
        self.encoderlayer_2 = BasicShadowFormer(dim=embed_dim*4,
                            output_dim=embed_dim*4,
                            input_resolution=(img_size // (2 ** 2), img_size // (2 ** 2)),
                            depth=depths[2],
                            num_heads=num_heads[2],
                            win_size=win_size,
                            mlp_ratio=self.mlp_ratio,
                            qkv_bias=qkv_bias, qk_scale=qk_scale,
                            drop=drop_rate, attn_drop=attn_drop_rate,
                            drop_path=enc_dpr[sum(depths[:2]):sum(depths[:3])],
                            norm_layer=norm_layer,
                            use_checkpoint=use_checkpoint,
                            token_projection=token_projection,token_mlp=token_mlp,se_layer=se_layer)
        self.dowsample_2 = dowsample(embed_dim*4, embed_dim*8)

        # Bottleneck
        channel_conv = embed_dim*16
        self.conv = BasicShadowFormer(dim=channel_conv,
                            output_dim=channel_conv,
                            input_resolution=(img_size // (2 ** 3), img_size // (2 ** 3)),
                            depth=depths[4],
                            num_heads=num_heads[4],
                            win_size=win_size,
                            mlp_ratio=self.mlp_ratio,
                            qkv_bias=qkv_bias, qk_scale=qk_scale,
                            drop=drop_rate, attn_drop=attn_drop_rate,
                            drop_path=conv_dpr,
                            norm_layer=norm_layer,
                            use_checkpoint=use_checkpoint,
                            token_projection=token_projection,token_mlp=token_mlp,se_layer=se_layer)

        # Decoder
        self.upsample_0 = upsample(channel_conv, embed_dim*4)
        channel_0 = embed_dim*8
        self.decoderlayer_0 = BasicShadowFormer(dim=channel_0,
                            output_dim=channel_0,
                            input_resolution=(img_size // (2 ** 2), img_size // (2 ** 2)),
                            depth=depths[6],
                            num_heads=num_heads[6],
                            win_size=win_size,
                            mlp_ratio=self.mlp_ratio,
                            qkv_bias=qkv_bias, qk_scale=qk_scale,
                            drop=drop_rate, attn_drop=attn_drop_rate,
                            drop_path=dec_dpr[sum(depths[5:6]):sum(depths[5:7])],
                            norm_layer=norm_layer,
                            use_checkpoint=use_checkpoint,
                            token_projection=token_projection,token_mlp=token_mlp,se_layer=se_layer)
        self.upsample_1 = upsample(channel_0, embed_dim*2)
        
        channel_1 = embed_dim*4
        self.decoderlayer_1 = BasicShadowFormer(dim=channel_1,
                            output_dim=channel_1,
                            input_resolution=(img_size // 2, img_size // 2),
                            depth=depths[7],
                            num_heads=num_heads[7],
                            win_size=win_size,
                            mlp_ratio=self.mlp_ratio,
                            qkv_bias=qkv_bias, qk_scale=qk_scale,
                            drop=drop_rate, attn_drop=attn_drop_rate,
                            drop_path=dec_dpr[sum(depths[5:7]):sum(depths[5:8])],
                            norm_layer=norm_layer,
                            use_checkpoint=use_checkpoint,
                            token_projection=token_projection,token_mlp=token_mlp,se_layer=se_layer, cab=True)
        self.upsample_2 = upsample(channel_1, embed_dim)
        
        self.decoderlayer_2 = BasicShadowFormer(dim=embed_dim*2,
                            output_dim=embed_dim*2,
                            input_resolution=(img_size, img_size),
                            depth=depths[8],
                            num_heads=num_heads[8],
                            win_size=win_size,
                            mlp_ratio=self.mlp_ratio,
                            qkv_bias=qkv_bias, qk_scale=qk_scale,
                            drop=drop_rate, attn_drop=attn_drop_rate,
                            drop_path=dec_dpr[sum(depths[5:8]):sum(depths[5:9])],
                            norm_layer=norm_layer,
                            use_checkpoint=use_checkpoint,
                            token_projection=token_projection,token_mlp=token_mlp,se_layer=se_layer,cab=True)

        self.Conv = nn.Conv2d(self.DINO_channel * 4, embed_dim * 8, kernel_size=1)
        self.relu = nn.LeakyReLU()
        

        self.use_axs = use_axs
        if use_white_balance:
            self.wb = GrayWorldRetinex()
            self.use_white_balance = True
            if self.use_axs == False:
                self.wb_alpha = nn.Parameter(torch.zeros(1, 3, 1, 1), requires_grad=True)
        else:
            self.use_white_balance = False

            
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward(self, x, DINO_Mat_features=None, point=None, normal=None, mask=None):
        self.img_size = torch.tensor((x.shape[2], x.shape[3]))
        
        # 準備幾何特徵
        point_feature1 = grid_sample(point, self.img_size // 2)
        point_feature2 = grid_sample(point, self.img_size // 4)
        point_feature3 = grid_sample(point, self.img_size // 8)
        normal1 = grid_sample(normal, self.img_size // 2)
        normal2 = grid_sample(normal, self.img_size // 4)
        normal3 = grid_sample(normal, self.img_size // 8)

        # 上採樣並投影 DINO 特徵
        H0, W0 = int(self.img_size[0]), int(self.img_size[1])
        H1, W1 = H0 // 2, W0 // 2
        H2, W2 = H0 // 4, W0 // 4
        H3, W3 = H0 // 8, W0 // 8
        
        dino_0 = self.dino_proj_0(F.interpolate(DINO_Mat_features[0], size=(H0, W0), mode='bilinear', align_corners=False))
        dino_1 = self.dino_proj_1(F.interpolate(DINO_Mat_features[1], size=(H1, W1), mode='bilinear', align_corners=False))
        dino_2 = self.dino_proj_2(F.interpolate(DINO_Mat_features[2], size=(H2, W2), mode='bilinear', align_corners=False))
        dino_3 = self.dino_proj_3(F.interpolate(DINO_Mat_features[3], size=(H3, W3), mode='bilinear', align_corners=False))
        
        # 轉為 sequence
        dino_0_flat = dino_0.flatten(2).transpose(1, 2)
        dino_1_flat = dino_1.flatten(2).transpose(1, 2)
        dino_2_flat = dino_2.flatten(2).transpose(1, 2)
        dino_3_flat = dino_3.flatten(2).transpose(1, 2)
        
        # Bottleneck 的拼接特徵
        patch_feature_all = torch.cat((DINO_Mat_features[0], DINO_Mat_features[1],
                                        DINO_Mat_features[2], DINO_Mat_features[3]), dim=1)
        dino_mat_cat = self.Conv(patch_feature_all)
        dino_mat_cat = self.relu(dino_mat_cat)
        dino_mat_cat_flat = dino_mat_cat.flatten(2).transpose(1, 2)

        # 原始 attention 用的 DINO
        dino_mat = None
        dino_mat1 = None
        dino_mat2 = F.interpolate(DINO_Mat_features[-1], scale_factor=2, mode='bilinear', align_corners=False)
        dino_mat3 = DINO_Mat_features[-1]

        # White balance
        if self.use_white_balance:
            A = self.wb(x)
            if self.use_axs:
                Y_I = _rgb2luma_tensor(x)
                Y_A = _rgb2luma_tensor(A)
                S = torch.clamp(Y_I / (Y_A + 1e-6), 0.5, 2.0)
                x_corrected = torch.clamp(A * S, 0, 1)
            else:
                alpha = torch.sigmoid(self.wb_alpha)
                x_corrected = alpha * A + (1 - alpha) * x
        else:
            x_corrected = x
            
            
        # RGBD input
        xi = torch.cat((x_corrected, point[:,2,:].unsqueeze(1)), dim=1)
        y = self.input_proj(xi)
        y = self.pos_drop(y)

        # Encoder - 直接相加融合
        self.img_size = (H0, W0)
        conv0 = self.encoderlayer_0(y + self.alpha_0 * dino_0_flat, dino_mat, point, normal, mask, img_size=self.img_size)
        pool0 = self.dowsample_0(conv0, img_size=self.img_size)

        self.img_size = (H1, W1)
        conv1 = self.encoderlayer_1(pool0 + self.alpha_1 * dino_1_flat, dino_mat1, point_feature1, normal1, img_size=self.img_size)
        pool1 = self.dowsample_1(conv1, img_size=self.img_size)

        self.img_size = (H2, W2)
        conv2 = self.encoderlayer_2(pool1 + self.alpha_2 * dino_2_flat, dino_mat2, point_feature2, normal2, img_size=self.img_size)
        pool2 = self.dowsample_2(conv2, img_size=self.img_size)

        # Bottleneck
        self.img_size = (H3, W3)
        pool2_fused = pool2 + self.alpha_3 * dino_3_flat
        pool2_cat = torch.cat([pool2_fused, dino_mat_cat_flat], -1)
        conv3 = self.conv(pool2_cat, dino_mat3, point_feature3, normal3, img_size=self.img_size)

        # Decoder (保持不變)
        up0 = self.upsample_0(conv3, img_size=self.img_size)
        self.img_size = (H2, W2)
        deconv0 = torch.cat([up0, conv2], -1)
        deconv0 = self.decoderlayer_0(deconv0, dino_mat2, point_feature2, normal2, img_size=self.img_size)

        up1 = self.upsample_1(deconv0, img_size=self.img_size)
        self.img_size = (H1, W1)
        deconv1 = torch.cat([up1, conv1], -1)
        deconv1 = self.decoderlayer_1(deconv1, dino_mat1, point_feature1, normal1, img_size=self.img_size)

        up2 = self.upsample_2(deconv1, img_size=self.img_size)
        self.img_size = (H0, W0)
        deconv2 = torch.cat([up2, conv0], -1)
        deconv2 = self.decoderlayer_2(deconv2, dino_mat, point, normal, mask, img_size=self.img_size)

        y = self.output_proj(deconv2, img_size=self.img_size) + x
        return y

if __name__ == '__main__':
    import torch
    from thop import profile, clever_format
    from torchinfo import summary


    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # RGBD (4 channels)
    dummy_input = torch.randn(1, 3, 512, 512).to(device)
    dino_mat = [torch.randn(1, 1024, s, s).to(device) for s in [64, 64, 64, 64]]
    point = torch.randn(1, 3, 512, 512).to(device)
    normal = torch.randn(1, 3, 512, 512).to(device)

    model = PhaSR(img_size=512, win_size=16).to(device)

    flops, params = profile(model, inputs=(dummy_input, dino_mat, point, normal))

    flops, params = clever_format([flops, params], "%.3f")

    print(f"FLOPs: {flops}")
    print(f"Params: {params}")


    x = torch.randn(1, 3, 512, 512).to("cuda")        # RGB
    dino_mat = [torch.randn(1, 1024, s, s).to("cuda") for s in [64, 64, 64, 64]]
    point = torch.randn(1, 3, 512, 512).to("cuda")
    normal = torch.randn(1, 3, 512, 512).to("cuda")

    model = PhaSR(img_size=512).to("cuda")

    summary(model, input_data=(x, dino_mat, point, normal), device="cuda", col_names=["num_params"])
