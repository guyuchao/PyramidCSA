import torch
import self_cuda_backend as _ext
import torch.nn as nn
from math import sqrt
import torch.autograd as autograd
import torch.nn.functional as F
from torch.autograd.function import once_differentiable
import numpy as np

def _check_contiguous(*args):
    if not all([mod is None or mod.is_contiguous() for mod in args]):
        raise ValueError("Non-contiguous input")

class PCSA_Weight(autograd.Function):
    @staticmethod
    def forward(ctx, query, key, radius=1,dilation=1):
        # Save context
        ctx.radius=radius
        ctx.dilation=dilation

        b, t, c, h, w = query.shape
        local_size=2*radius+1
        size = (b, t, local_size*local_size*t, h, w)
        weight = torch.zeros(size, dtype=query.dtype, layout=query.layout, device=query.device)
        weight.fill_(-np.inf)
        _ext.weight_forward(query, key, weight,radius,dilation)
        # Output
        ctx.save_for_backward(query, key)

        return weight

    @staticmethod
    @once_differentiable
    def backward(ctx, dw):
        query, key= ctx.saved_tensors
        dquery = torch.zeros_like(query)
        dkey = torch.zeros_like(key)
        _ext.weight_backward(dw.contiguous(), query, key, dquery, dkey, ctx.radius,ctx.dilation)
        _check_contiguous(dquery, dkey)
        return dquery, dkey,None,None


class PCSA_Map(autograd.Function):
    @staticmethod
    def forward(ctx, weight, proj, radius=1,dilation=1):
        # Save context
        ctx.radius=radius
        ctx.dilation=dilation
        out = torch.zeros_like(proj)
        _ext.map_forward(weight, proj, out,radius,dilation)
        # Output
        ctx.save_for_backward(weight, proj)

        return out

    @staticmethod
    @once_differentiable
    def backward(ctx, dout):
        weight, proj= ctx.saved_tensors
        dweight = torch.zeros_like(weight)
        dproj = torch.zeros_like(proj)
        _ext.map_backward(dout.contiguous(), weight, proj, dweight, dproj, ctx.radius,ctx.dilation)
        _check_contiguous(dweight, dproj)
        return dweight, dproj,None,None

pcsa_weight = PCSA_Weight.apply
pcsa_map = PCSA_Map.apply

class T_Moduel(nn.Module):
    def __init__(self,channels_in=32,n_head=4,d_k=8,d_v=8):
        super(T_Moduel, self).__init__()
        self.n_head=n_head
        self.d_k=d_k
        self.query_conv=nn.Conv3d(channels_in,n_head*d_k,1,bias=False)
        self.key_conv=nn.Conv3d(channels_in,n_head*d_k,1,bias=False)
        self.value_conv=nn.Conv3d(channels_in,n_head*d_v,1,bias=False)
        self.output_Linear=nn.Conv3d(channels_in,channels_in,1,bias=False)

    def forward(self, x):
        dilation=[1,2,1,2]
        radius=[3,3,4,4]
        x_ = x.permute(0, 2, 1, 3, 4).contiguous()  # b c t h w
        query=self.query_conv(x_).permute(0,2,1,3,4)
        query_chunk=query.chunk(self.n_head,2)
        key=self.key_conv(x_).permute(0,2,1,3,4)
        key_chunk=key.chunk(self.n_head,2)
        value=self.value_conv(x_).permute(0,2,1,3,4)
        value_chunk=value.chunk(self.n_head,2)
        out=[]
        for i in range(self.n_head):
            tmp_query=query_chunk[i].contiguous()
            tmp_key=key_chunk[i].contiguous()
            tmp_value=value_chunk[i].contiguous()
            energy = pcsa_weight(tmp_query, tmp_key,radius[i],dilation[i])
            attention=energy/sqrt(8)
            attention = F.softmax(attention, 2)
            out.append(pcsa_map(attention,tmp_value,radius[i],dilation[i]))
        out=torch.cat(out,dim=2).permute(0, 2, 1, 3, 4)
        out=self.output_Linear(out)
        out=out.permute(0, 2, 1, 3, 4)
        return out+x


__all__ = ["T_Moduel", "sa_weight", "sa_map"]


if __name__=="__main__":
    x=torch.zeros(2,5,32,28,42).cuda()
    t_module=T_Moduel().cuda()
    print(t_module(x).shape)
