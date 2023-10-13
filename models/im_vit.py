import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models.layers import DropPath, to_2tuple
from functools import partial
# from collections import OrderedDict
import numpy as np
import scipy.sparse as sp
import pdb                           #         pdb.set_trace() 



# A memory-efficient implementation of Swish function
class SwishImplementation(torch.autograd.Function):
    @staticmethod
    def forward(ctx, i):
        result = i * torch.sigmoid(i)
        ctx.save_for_backward(i)
        return result

    @staticmethod
    def backward(ctx, grad_output):
        i = ctx.saved_tensors[0]
        sigmoid_i = torch.sigmoid(i)
        return grad_output * (sigmoid_i * (1 + i * (1 - sigmoid_i)))

class MemoryEfficientSwish(nn.Module):
    def forward(self, x):
        return SwishImplementation.apply(x)


class Classifier_Head(nn.Module):
    def __init__(self, embed_dim, input_resolution, num_classes=1000, fc_dim=1280, head_dropout=0.):
        super().__init__()
        self.embed_dim = embed_dim
        self.fc_dim = fc_dim
        self.input_resolution = input_resolution
        self.num_classes = num_classes
        self._fc = nn.Conv2d(embed_dim, fc_dim, kernel_size=1)
        self.ap_fc = nn.Linear(embed_dim, fc_dim) 
        self._bn = nn.BatchNorm2d(fc_dim, eps=1e-5)
        self._swish = MemoryEfficientSwish()
        self._avg_pooling = nn.AdaptiveAvgPool2d((1,None)) 
        self._drop = nn.Dropout(head_dropout)
        self.head = nn.Linear(fc_dim, num_classes) if num_classes > 0 else nn.Identity()

    def forward(self, x, H, W):
        B, N, C = x.shape
        acm_point = x[:,0]
        acm_point = self.ap_fc(acm_point)
        acm_point = acm_point.unsqueeze(1)
        x = self._fc(x[:,1:].permute(0, 2, 1).reshape(B, C, H, W))
        x = self._bn(x)
        x = self._swish(x)
        x = x.flatten(2).transpose(1, 2)
        x = torch.cat((acm_point,x), dim=1)
        x = self._avg_pooling(x).flatten(1)
        x = self._drop(x)
        x = self.head(x)
        return x

    def flops(self):

        flops = 0
        Ho = self.input_resolution
        Wo = self.input_resolution
        flops += Ho * Wo * self.fc_dim * self.embed_dim
        flops += self.fc_dim * self.embed_dim
        flops += Ho * Wo * self.fc_dim
        flops += self.fc_dim * self.num_classes

        return flops

# class MlpHead(nn.Module):
#     """ MLP classification head
#     """
#     def __init__(self, embed_dim, num_classes=1000, mlp_ratio=4, act_layer=nn.GELU,
#         norm_layer=nn.LayerNorm, head_dropout=0., bias=True):
#         super().__init__()
#         hidden_features = int(mlp_ratio * embed_dim)
#         self.fc1 = nn.Linear(embed_dim, hidden_features, bias=bias)
#         self.act = act_layer()
#         self.norm = norm_layer(hidden_features)
#         self.fc2 = nn.Linear(hidden_features, num_classes, bias=bias)
#         self.head_dropout = nn.Dropout(head_dropout)


#     def forward(self, x):
#         x = self.fc1(x)
#         x = self.act(x)
#         x = self.norm(x)
#         x = self.head_dropout(x)
#         x = self.fc2(x)
#         return x

class PatchEmbed(nn.Module):
    r""" Image to Patch Embedding

    Args:
        img_size (int): Image size.  Default: 224.
        patch_size (int): Patch token size. Default: [4].
        in_chans (int): Number of input image channels. Default: 3.
        embed_dim (int): Number of linear projection output channels. Default: 96.
        norm_layer (nn.Module, optional): Normalization layer. Default: None
    """

    def __init__(self, img_size=224, patch_size=[8,12,16,20], in_c=3, stem_channel=16, embed_dim=384, norm_layer=None):
        super().__init__()
        feature_size = int(img_size // 16)
        self.img_size = to_2tuple(img_size)
        self.feature_size = feature_size
        self.patch_size = patch_size
        self.stem_channel = stem_channel
        self.in_c = in_c
        self.embed_dim = embed_dim
             
        self.stem_conv1 = nn.Conv2d(in_c, stem_channel, kernel_size=3, stride=2, padding=1, bias=True)
        self.stem_relu1 = nn.GELU()
        self.stem_norm1 = nn.BatchNorm2d(stem_channel, eps=1e-5)
            
        self.stem_conv2 = nn.Conv2d(stem_channel, stem_channel, kernel_size=3, stride=1, padding=1, bias=True)
        self.stem_relu2 = nn.GELU()
        self.stem_norm2 = nn.BatchNorm2d(stem_channel, eps=1e-5)
            
        self.stem_conv3 = nn.Conv2d(stem_channel, stem_channel, kernel_size=3, stride=1, padding=1, bias=True)
        self.stem_relu3 = nn.GELU()
        self.stem_norm3 = nn.BatchNorm2d(stem_channel, eps=1e-5)

        self.projs = nn.ModuleList()
        for i, ps in enumerate(patch_size):
        # if i == len(patch_size) - 1:
        #     dim = embed_dim // 2 ** i
        # else:
        #     dim = embed_dim // 2 ** (i + 1)
            dim = embed_dim // len(self.patch_size)
            stride = 8
            if i == 0:
                padding = 0
            else:
                padding = (ps - 8) // 2
            self.projs.append(nn.Conv2d(stem_channel, dim, kernel_size=ps, stride=stride, padding=padding))

        if norm_layer is not None:
            self.norm = norm_layer(embed_dim)
        else:
            self.norm = None
        #加上子聚点
        self.acm_point = nn.Parameter(torch.zeros(1, 1, embed_dim))
        nn.init.trunc_normal_(self.acm_point, std=.02)

    def forward(self, x):
        B, C, H, W = x.shape
        # FIXME look at relaxing size constraints
        assert H == self.img_size[0] and W == self.img_size[1], \
            f"Input image size ({H}*{W}) doesn't match model ({self.img_size[0]}*{self.img_size[1]})."

        x = self.stem_conv1(x)
        x = self.stem_relu1(x)
        x = self.stem_norm1(x)
            
        x = self.stem_conv2(x)
        x = self.stem_relu2(x)
        x = self.stem_norm2(x)
            
        x = self.stem_conv3(x)
        x = self.stem_relu3(x)
        x = self.stem_norm3(x)
        xs = []
        for i in range(len(self.projs)):
            tx = self.projs[i](x)
            # tx = self.projs[i](x).flatten(2).transpose(1, 2)
            # permute: [B, C, H, W] -> [B, H, W, C]
            # tx = tx.permute(0,2,3,1).contiguous()
            xs.append(tx)  # B Ph Pw C
        x = torch.cat(xs, dim=1)
 
        _, C, H, W = x.shape
        # flatten: [B, C, H, W] -> [B, C, HW]
        # transpose: [B, C, HW] -> [B, HW, C]

        x = x.flatten(2).transpose(1, 2)
        if self.norm is not None:
            x = self.norm(x)
        # 加上子聚点
        acm_point = self.acm_point.expand(B, -1, -1)
        x = torch.cat((acm_point,x), dim=1)    
        return x, H, W

    def flops(self):
        Ho = self.feature_size
        Wo = self.feature_size
        flops = 0       
        flops += (self.img_size[0] / 2) * (self.img_size[1] / 2) * self.stem_channel * self.in_c * 3 * 3
        flops += (self.img_size[0] / 2) * (self.img_size[1] / 2) * self.stem_channel * self.stem_channel * 3 * 3        
        flops += (self.img_size[0] / 2) * (self.img_size[1] / 2) * self.stem_channel * self.stem_channel * 3 * 3        
        flops += 3 * (self.img_size[0] / 2) * (self.img_size[1] / 2) * self.stem_channel
        for i, ps in enumerate(self.patch_size):
            # if i == len(self.patch_size) - 1:
            #     dim = self.embed_dim // 2 ** i
            # else:
            #     dim = self.embed_dim // 2 ** (i + 1)
            dim = self.embed_dim // len(self.patch_size)
            flops += Ho * Wo * dim * self.in_c * (self.patch_size[i] * self.patch_size[i])
        if self.norm is not None:
            flops += Ho * Wo * self.embed_dim
        return flops
        
class Mlp(nn.Module):
    """ MLP as used in Vision Transformer, MLP-Mixer and related networks
    """
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features

        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.drop1 = nn.Dropout(drop)
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop2 = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop1(x)
        x = self.fc2(x)
        x = self.drop2(x)
        return x


class InteractionMaskBlock(nn.Module):
    r"""Global feature extraction module.

    Args:
        dim (int): Number of input channels.
        input_resolution (int): Input resulotion.
        num_heads (int): Number of attention heads.
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim.
        qkv_bias (bool, optional): If True, add a learnable bias to query, key, value. Default: True
        drop_ratio (float, optional): Dropout rate. Default: 0.0
        attn_drop_ratio (float, optional): Attention dropout rate. Default: 0.0
        drop_path_ratio (float, optional): Stochastic depth rate. Default: 0.0
        act_layer (nn.Module, optional): Activation layer. Default: nn.GELU
        norm_layer (nn.Module, optional): Normalization layer.  Default: nn.LayerNorm
        loop2 (int): Used for DCL looping between multiple heads in different layers

    """
    def __init__(self, dim, input_resolution, num_heads,  grid_size, position_bias, 
                 mlp_ratio=4., qkv_bias=False,
                 drop_ratio=0., attn_drop_ratio=0., drop_path_ratio=0.,
                 act_layer=nn.GELU, norm_layer=nn.LayerNorm, loop=0,
                 attn_mask_scope = [1,2,3], interaction_mask = False
                 ):
        super().__init__()

        self.norm1 = norm_layer(dim)
        self.attn = IMAttention(dim, input_resolution, num_heads = num_heads, grid_size = to_2tuple(grid_size), 
                                position_bias = position_bias, qkv_bias = qkv_bias,
                                attn_drop_ratio = attn_drop_ratio, proj_drop_ratio = drop_ratio,loop = loop,
                                attn_mask_scope = attn_mask_scope,
                                interaction_mask = interaction_mask
                                )
        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path = DropPath(drop_path_ratio) if drop_path_ratio > 0. else nn.Identity()

        # self.res_scale2 = Scale(dim=dim) 
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop_ratio)
        self.input_resolution = input_resolution
        self.dim = dim
        self.mlp_ratio = mlp_ratio
        self.grid_size = grid_size
        self.layer_scale = nn.Parameter(torch.ones((1)), requires_grad=True)

    def forward(self, x):

        layer_scale = torch.clamp(self.layer_scale, min=0,  max=1)
        x = x + (self.drop_path(self.norm1(self.attn(x)))) * layer_scale
        x = x + (self.drop_path(self.norm2(self.mlp(x)))) * layer_scale

        # x = x + (self.drop_path(self.norm1(self.attn(x)))) 
        # x = x + (self.drop_path(self.norm2(self.mlp(x))))

        # x = self.res_scale1(x) + self.drop_path(self.norm1(self.attn(x)))
        # x = self.res_scale2(x) + self.drop_path(self.norm2(self.mlp(x)))
        return x   

    def flops(self):
        flops = 0
        H = self.input_resolution
        W = self.input_resolution
        # norm1
        flops += self.dim * (H * W + 1)
        nG = H * W / self.grid_size / self.grid_size  
        # MSA
        flops += nG * self.attn.flops(H * W + 1)
        # mlp
        flops += 2 * (H * W + 1) * self.dim  * self.mlp_ratio * self.dim
        # norm2
        flops += self.dim * (H * W + 1)
        return flops

class DynamicPosBias(nn.Module):
    def __init__(self, dim, num_heads):
        super().__init__()

        self.num_heads = num_heads
        self.pos_dim = dim // 4
        self.pos_proj = nn.Linear(2, self.pos_dim)
        self.pos1 = nn.Sequential(
            nn.LayerNorm(self.pos_dim),
            nn.ReLU(inplace=True),
            nn.Linear(self.pos_dim, self.pos_dim),
        )
        self.pos2 = nn.Sequential(
            nn.LayerNorm(self.pos_dim),
            nn.ReLU(inplace=True),
            nn.Linear(self.pos_dim, self.pos_dim)
        )
        self.pos3 = nn.Sequential(
            nn.LayerNorm(self.pos_dim),
            nn.ReLU(inplace=True),
            nn.Linear(self.pos_dim, self.num_heads)
        )
    def forward(self, biases):
        pos = self.pos3(self.pos2(self.pos1(self.pos_proj(biases))))
        return pos

    def flops(self, N):
        flops = N * 2 * self.pos_dim
        flops += N * self.pos_dim * self.pos_dim
        flops += N * self.pos_dim * self.pos_dim
        flops += N * self.pos_dim * self.num_heads
        return flops

class IMAttention(nn.Module):
    r"""Cyclic dilated list based multi-head self attention(IM_MHA)module.

    Args:
        dim (int): Number of input channels.
        input_resolution (int): Input resulotion.
        num_heads (int): Number of attention heads.
        qkv_bias (bool, optional): If True, add a learnable bias to query, key, value. Default: True
        attn_drop_ratio (float, optional): Attention dropout rate. Default: 0.0
        proj_drop_ratio (float, optional): Dropout rate. Default: 0.0
        loop2 (int): Used for DCL looping between multiple heads in different layers

    """
    def __init__(self,
                 dim,   # 输入token的dim
                 input_resolution,
                 num_heads,
                 grid_size,
                 position_bias,
                 qkv_bias=False,
                 attn_drop_ratio=0.,
                 proj_drop_ratio=0.,
                 loop=0,
                 attn_mask_scope = [1,4,7],
                 interaction_mask = False
                 ):
        super().__init__()
        self.num_heads = num_heads
        self.dim = dim
        self.interaction_mask = interaction_mask
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop_ratio)
        self.position_bias = position_bias
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop_ratio)
        self.input_resolution = input_resolution
        self.loop1 = loop // (len(attn_mask_scope)+1)
        self.loop2 = loop % (len(attn_mask_scope)+1)
        self.grid_size = grid_size

        self.scale = nn.Parameter(10 * torch.ones((num_heads, 1, 1)), requires_grad=True)

        if self.interaction_mask:
            self.pre_softmax_interaction_mask = nn.Conv2d(num_heads, num_heads, 1, bias = False)
            self.post_softmax_interaction_mask = nn.Conv2d(num_heads, num_heads, 1, bias = False)
        

        scope = attn_mask_scope
        adj1 = load_data(input_resolution, scope[0], add_acm_point = True)
        adj2 = load_data(input_resolution, scope[1], add_acm_point = True)
        adj3 = load_data(input_resolution, scope[2], add_acm_point = True)
        adj4 = torch.ones_like(adj1)
        self.repeat = int( num_heads // (len(attn_mask_scope)+1))
        self.register_buffer("adj1",adj1)
        self.register_buffer("adj2",adj2)
        self.register_buffer("adj3",adj3)
        self.register_buffer("adj4",adj4)

        if self.position_bias:

            self.pos = DynamicPosBias(self.dim // 4, self.num_heads)
            
            # generate mother-set
            position_bias_h = torch.arange(1 - (self.grid_size[0]+1), (self.grid_size[0]+1))
            position_bias_w = torch.arange(1 - (self.grid_size[0]+1), (self.grid_size[0]+1))
            biases = torch.stack(torch.meshgrid([position_bias_h, position_bias_w]))  # 2, 2Wh, 2Ww
            biases = biases.flatten(1).transpose(0, 1).float()
            self.register_buffer("biases", biases)

            # get pair-wise relative position index for each token inside the group
            coords_h = torch.arange(self.grid_size[0])
            coords_w = torch.arange(self.grid_size[1])
            coords = torch.stack(torch.meshgrid([coords_h, coords_w]))  # 2, Wh, Ww
            coords_flatten = torch.flatten(coords, 1)  # 2, Wh*Ww
            relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]  # 2, Wh*Ww, Wh*Ww
            relative_coords = relative_coords.permute(1, 2, 0).contiguous()  # Wh*Ww, Wh*Ww, 2
            relative_coords[:, :, 0] += self.grid_size[0] - 1  # shift to start from 0
            relative_coords[:, :, 1] += self.grid_size[1] - 1
            relative_coords[:, :, 0] *= 2 * self.grid_size[1] - 1
            relative_coords = relative_coords.sum(-1)  # Wh*Ww, Wh*Ww
            relative_position_index2 = F.pad(relative_coords,
                                       (1,0, 
                                        1,0),
                                        "constant",
                                        (2 * self.grid_size[1]-1)*(2 * self.grid_size[1]-1)+1)        
            self.register_buffer("relative_position_index2", relative_position_index2)

    def forward(self, x):
        # [batch_size, num_patches + 1, total_embed_dim]
        B, N, C = x.shape
        #循环移位列表构建
        Prior_adjacency_matrix = torch.cat((self.adj1.expand((self.repeat-self.loop1),-1,-1),
                                                self.adj2.expand((self.repeat-self.loop1),-1,-1),
                                                self.adj3.expand((self.repeat-self.loop1),-1,-1),
                                                self.adj4.expand((self.repeat+(3*self.loop1)),-1,-1)),dim=0)       

        if self.loop2 > 0 :
            a = Prior_adjacency_matrix[0:self.loop2*self.repeat,:,:]
            b = Prior_adjacency_matrix[self.loop2*self.repeat:,:,:]
            Prior_adjacency_matrix = torch.cat((b,a),dim=0)
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]  # make torchscript happy (cannot use tensor as tuple)
        # cosine attention
        attn = (F.normalize(q, dim=-1) @ F.normalize(k, dim=-1).transpose(-2, -1))
        scale = torch.clamp(self.scale, max=torch.tensor(1. / 0.01).to(self.scale.device))
        attn = attn * scale

        if self.position_bias:
            pos = self.pos(self.biases) # 2Wh-1 * 2Ww-1, heads
            # select position bias
            relative_position_bias = pos[self.relative_position_index2.view(-1)].view(
                self.grid_size[0] * self.grid_size[1]+1, self.grid_size[0] * self.grid_size[1]+1, -1)  # Wh*Ww,Wh*Ww,nH
            relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()  # nH, Wh*Ww, Wh*Ww 
            attn = attn + relative_position_bias.unsqueeze(0)

        if self.interaction_mask:
            attn = self.pre_softmax_interaction_mask(attn)


        zero_vec = -1e9*torch.ones_like(attn)
        attn = torch.where(Prior_adjacency_matrix > 0, attn, zero_vec)

        attn = attn.softmax(dim=-1)

        if self.interaction_mask:
            attn = self.post_softmax_interaction_mask(attn)
            
        attn = self.attn_drop(attn)
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)

        return x

    def flops(self, N):
        # calculate flops for l_fature_size with token length of N
        flops = 0
        # qkv = self.qkv(x)
        flops += N * self.dim * 3 * self.dim
        # attn = (q @ k.transpose(-2, -1))
        flops += self.num_heads * N * (self.dim // self.num_heads) * N
        # bias
        if self.position_bias:
            flops += self.pos.flops(N)    
        # Talking-Heads Attention
        if self.interaction_mask :
            flops += 2 * N * N * self.num_heads * self.num_heads
        #  x = (attn @ v)
        flops += self.num_heads * N * N * (self.dim // self.num_heads)
        # x = self.proj(x)
        flops += N * self.dim * self.dim
        return flops

def Build_Adjacency_matrix(scope, token_sum, feature_size):
    serial = np.arange(1,token_sum+1,dtype=int)
    serial = serial.reshape(feature_size,feature_size)
    mask_matrix = np.pad(serial,scope,'constant')
    a = []
    adj_matrix = []
    for i in range(feature_size+2*scope):
        for j in range(feature_size+2*scope):
            if (mask_matrix[i][j])>0:
                a=mask_matrix[i][j]
                for k in range(2*scope+1):
                    for m in range(2*scope+1):
                        adj_matrix.append(a)
                        adj_matrix.append(mask_matrix[i-scope+k][j-scope+m]) 
    adj_matrix = np.array(adj_matrix,dtype=int).reshape(-1,2)
    adj_matrix = adj_matrix[np.all(adj_matrix != 0, axis=1)]
    return adj_matrix
    
def load_data(feature_size, scope, add_acm_point = True):
    token_sum = feature_size * feature_size
    # print('Loading {} dataset...'.format(path))
    edges = Build_Adjacency_matrix(scope, token_sum, feature_size)
    a = np.ones_like(edges,int)
    edges = edges - a
    adj = sp.coo_matrix((np.ones(edges.shape[0]), (edges[:, 0], edges[:, 1])), shape=(token_sum, token_sum), dtype=np.float32)

    # pdb.set_trace() 
    # build symmetric adjacency matrix
    # adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)
    # adj = adj + sp.eye(adj.shape[0])
    adj = torch.FloatTensor(np.array(adj.todense()))
    if add_acm_point is True:
        # adj = np.pad(adj,((1,0),(1,0)),'constant', constant_values=1)
        adj = F.pad(adj,(1,0, 
                        1,0),
                        "constant",
                        1)   
    adj = adj.unsqueeze(0)
    return adj

class IM_VIT(nn.Module):
    def __init__(self, img_size=224, patch_size=8, in_c=3, num_classes=1000,
                 stem_channel=16, embed_dim=384, fc_dim = 1280, depths=12, num_heads=12, 
                 mlp_ratio=4.0, qkv_bias=True,grid_size=14,position_bias = True,
                 drop_ratio=0.1, attn_drop_ratio=0.1, drop_path_ratio=0.1, 
                 head_dropout=0.0, attn_mask_scope = [1,4,7], classifier_head=True,
                 embed_layer=PatchEmbed, norm_layer=None,
                 act_layer=nn.GELU,patch_norm=True,
                 interaction_mask = False):
        """
        Args:
            img_size (int | tuple(int)): Input image size. Default 224
            patch_size (int | tuple(int)): Patch size. Default: 4
            in_chans (int): Number of input image channels. Default: 3
            num_classes (int): Number of classes for classification head. Default: 1000
            embed_dim (int): Patch embedding dimension. Default: 48
            depth (int): depth of transformer
            num_heads (int): number of attention heads in different blocks   
            mlp_ratio (int): ratio of mlp hidden dim to embedding dim. Default: 4
            qkv_bias (bool): enable bias for qkv if True
            window_size (int): Window size. Default: 7
            drop_ratio (float): dropout rate
            attn_drop_ratio (float): attention dropout rate
            drop_path_ratio (float): stochastic depth rate
            embed_layer (nn.Module): patch embedding layer
            norm_layer: (nn.Module): normalization layer
            act_layer (nn.Module, optional): Activation layer. Default: nn.GELU
            patch_norm (bool): If True, add normalization after patch embedding. Default: True
        """
        super().__init__()

        self.num_classes = num_classes
        norm_layer = norm_layer or partial(nn.LayerNorm, eps=1e-6)
        self.patch_norm = patch_norm
        self.position_bias = position_bias 
        self.embed_dim = embed_dim
        self.patch_embed = embed_layer(img_size=img_size, patch_size=patch_size,in_c=in_c,stem_channel=stem_channel,
                                        embed_dim=self.embed_dim, norm_layer=norm_layer if self.patch_norm else None )
        self.patch_drop = nn.Dropout(p=drop_ratio)        
        # self.avgpool = nn.AdaptiveAvgPool2d((1,None)) 
        self.classifier_head = classifier_head
        self.mlp_ratio = mlp_ratio
        self.im_feature_size = self.patch_embed.feature_size

        if self.position_bias != True:
            self.num_patches = self.patch_embed.feature_size[0] * self.patch_embed.feature_size[1] 
            self.pos_embed = nn.Parameter(torch.zeros(1, self.num_patches + 1,self.embed_dim))
            self.pos_drop = nn.Dropout(p=drop_ratio)
            nn.init.trunc_normal_(self.pos_embed, std=0.02)

        dpr = [x.item() for x in torch.linspace(0, drop_path_ratio, depths)]  # stochastic depth decay rule

        self.blocks = nn.Sequential(*[
            InteractionMaskBlock(dim=self.embed_dim, input_resolution=self.im_feature_size, 
                                 num_heads=num_heads, grid_size = grid_size,
                                 position_bias = position_bias, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, 
                                 drop_ratio=drop_ratio, attn_drop_ratio=attn_drop_ratio, drop_path_ratio=dpr[i],
                                 norm_layer=norm_layer, act_layer=act_layer,loop=i,
                                 attn_mask_scope=attn_mask_scope, interaction_mask=interaction_mask)
            for i in range(depths)
        ])
        
        self.norm = norm_layer(self.embed_dim)
        # Classifier head(s)
        if self.classifier_head :
            # self.head = MlpHead(self.embed_dim, num_classes,head_dropout=head_dropout) if num_classes > 0 else nn.Identity()
            self.head = Classifier_Head(self.embed_dim, input_resolution=self.im_feature_size, num_classes=1000, fc_dim=fc_dim, head_dropout=head_dropout) if num_classes > 0 else nn.Identity()
        else :
            self.head = nn.Linear(self.embed_dim, num_classes) if num_classes > 0 else nn.Identity()

        self.apply(self._init_IM_VIT_weights)


    def forward(self,x):
        x, H, W = self.patch_embed(x) 
        x = self.patch_drop(x)
        if self.position_bias != True:
            x = self.pos_drop(x + self.pos_embed)
        x = self.blocks(x)
        x = self.norm(x)
        # x = x[:,0]
        # x = self.head(x)
        x = self.head(x, H, W)
        return x    

        
    def _init_IM_VIT_weights(self,m):
        """
        IM_VIT weight initialization
        :param m: module
        """
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=.01)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight, mode="fan_out")
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.LayerNorm):
            nn.init.zeros_(m.bias)
            nn.init.ones_(m.weight)
        elif isinstance(m, nn.BatchNorm2d):
            nn.init.zeros_(m.bias)
            nn.init.ones_(m.weight)

    def flops(self):
        Ho = self.im_feature_size
        Wo = self.im_feature_size
        flops = 0
        flops += self.patch_embed.flops()
        for _, block in enumerate(self.blocks):
            flops += block.flops()        
        # norm
        flops += self.embed_dim * Ho * Wo
        # Classifier head(s)
        if self.classifier_head :
            #Classifier_Head
            flops += self.head.flops()
            # MlpHead
            # flops += (Ho * Wo + 1) * self.embed_dim * self.mlp_ratio * self.num_classes
            # flops += (Ho * Wo + 1) * self.embed_dim * self.mlp_ratio
        else :
            flops += self.embed_dim * self.num_classes
        return flops
