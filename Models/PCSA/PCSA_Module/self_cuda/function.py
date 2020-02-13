import torch
import self_cuda_backend as _ext
import torch.nn as nn
import torch.autograd as autograd
import torch.cuda.comm as comm
import torch.nn.functional as F
from torch.autograd.function import once_differentiable



def _check_contiguous(*args):
    if not all([mod is None or mod.is_contiguous() for mod in args]):
        raise ValueError("Non-contiguous input")


class SA_Weight(autograd.Function):
    @staticmethod
    def forward(ctx, query, key):
        # Save context
        b, t, c, h, w = query.shape
        size = (b, t, 9*t, h, w)
        weight = torch.zeros(size, dtype=query.dtype, layout=query.layout, device=query.device)

        _ext.weight_forward(query, key, weight)

        # Output
        ctx.save_for_backward(query, key)

        return weight

    @staticmethod
    @once_differentiable
    def backward(ctx, dw):
        query, key = ctx.saved_tensors

        dquery = torch.zeros_like(query)
        dkey = torch.zeros_like(key)

        _ext.weight_backward(dw.contiguous(), query, key, dquery, dkey)

        _check_contiguous(dquery, dkey)

        return dquery, dkey


class SA_Map(autograd.Function):
    @staticmethod
    def forward(ctx, weight, proj):
        # Save context
        out = torch.zeros_like(proj)
        _ext.map_forward(weight, proj, out)

        # Output
        ctx.save_for_backward(weight, proj)

        return out

    @staticmethod
    @once_differentiable
    def backward(ctx, dout):
        weight, proj = ctx.saved_tensors

        dweight = torch.zeros_like(weight)
        dproj = torch.zeros_like(proj)

        _ext.map_backward(dout.contiguous(), weight, g, dweight, dproj)

        _check_contiguous(dweight, dproj)

        return dweight, dproj


sa_weight = SA_Weight.apply
sa_map = SA_Map.apply

class T_Moduel(nn.Module):
    def __init__(self,in_dim):
        super(T_Moduel, self).__init__()
        self.channel_in=in_dim
        self.query_conv=nn.Conv3d(in_channels=in_dim, out_channels=in_dim//8, kernel_size=1,bias=False)
        self.key_conv = nn.Conv3d(in_channels=in_dim, out_channels=in_dim//8, kernel_size=1,bias=False)
        self.proj_conv = nn.Conv3d(in_channels=in_dim, out_channels=in_dim, kernel_size=1,bias=False)
        self.gamma = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        x=x.permute(0,2,1,3,4).contiguous()#b c t h w
        query=self.query_conv(x).permute(0,2,1,3,4).contiguous()
        key=self.key_conv(x).permute(0,2,1,3,4).contiguous()
        proj=self.proj_conv(x).permute(0,2,1,3,4).contiguous()
        energy=sa_weight(query,key)
        attention=F.softmax(energy,2)
        out=sa_map(attention,proj)
        x = x.permute(0, 2, 1, 3, 4).contiguous()
        out=self.gamma*out+x
        return out

__all__ = ["T_Moduel", "sa_weight", "sa_map"]

if __name__=="__main__":
    x=torch.zeros(2,5,32,28,42).cuda()
    t_module=T_Moduel(32).cuda()
    print(t_module(x).shape)