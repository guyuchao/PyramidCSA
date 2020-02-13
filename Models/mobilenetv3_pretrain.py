import torch.nn as nn
import torch
import numpy as np
import math
import torch.nn.functional as F
from config import config
from Models.lightrfb import LightRFB
__all__ = ['mobilenetv3_large']


def _make_divisible(v, divisor, min_value=None):
    """
    This function is taken from the original tf repo.
    It ensures that all layers have a channel number that is divisible by 8
    It can be seen here:
    https://github.com/tensorflow/models/blob/master/research/slim/nets/mobilenet/mobilenet.py
    :param v:
    :param divisor:
    :param min_value:
    :return:
    """
    if min_value is None:
        min_value = divisor
    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
    # Make sure that round down does not go down by more than 10%.
    if new_v < 0.9 * v:
        new_v += divisor
    return new_v


class h_sigmoid(nn.Module):
    def __init__(self, inplace=True):
        super(h_sigmoid, self).__init__()
        self.relu = nn.ReLU6(inplace=inplace)

    def forward(self, x):
        return self.relu(x + 3) / 6


class h_swish(nn.Module):
    def __init__(self, inplace=True):
        super(h_swish, self).__init__()
        self.sigmoid = h_sigmoid(inplace=inplace)

    def forward(self, x):
        return x * self.sigmoid(x)


class SELayer(nn.Module):
    def __init__(self, channel, reduction=4):
        super(SELayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
                nn.Linear(channel, channel // reduction),
                nn.ReLU(inplace=True),
                nn.Linear(channel // reduction, channel),
                h_sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y


def conv_3x3_bn(inp, oup, stride):
    return nn.Sequential(
        nn.Conv2d(inp, oup, 3, stride, 1, bias=False),
        nn.BatchNorm2d(oup),
        h_swish()
    )


def conv_1x1_bn(inp, oup):
    return nn.Sequential(
        nn.Conv2d(inp, oup, 1, 1, 0, bias=False),
        nn.BatchNorm2d(oup),
        h_swish()
    )


class InvertedResidual(nn.Module):
    def __init__(self, inp, hidden_dim, oup, kernel_size, stride, use_se, use_hs):
        super(InvertedResidual, self).__init__()
        assert stride in [1, 2]

        self.identity = stride == 1 and inp == oup

        if inp == hidden_dim:
            self.conv = nn.Sequential(
                # dw
                nn.Conv2d(hidden_dim, hidden_dim, kernel_size, stride, (kernel_size - 1) // 2, groups=hidden_dim, bias=False),
                nn.BatchNorm2d(hidden_dim),
                h_swish() if use_hs else nn.ReLU(inplace=True),
                # Squeeze-and-Excite
                SELayer(hidden_dim) if use_se else nn.Sequential(),
                # pw-linear
                nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias=False),
                nn.BatchNorm2d(oup),
            )
        else:
            self.conv = nn.Sequential(
                # pw
                nn.Conv2d(inp, hidden_dim, 1, 1, 0, bias=False),
                nn.BatchNorm2d(hidden_dim),
                h_swish() if use_hs else nn.ReLU(inplace=True),
                # dw
                nn.Conv2d(hidden_dim, hidden_dim, kernel_size, stride, (kernel_size - 1) // 2, groups=hidden_dim, bias=False),
                nn.BatchNorm2d(hidden_dim),
                # Squeeze-and-Excite
                SELayer(hidden_dim) if use_se else nn.Sequential(),
                h_swish() if use_hs else nn.ReLU(inplace=True),
                # pw-linear
                nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias=False),
                nn.BatchNorm2d(oup),
            )

    def forward(self, x):
        if self.identity:
            return x + self.conv(x)
        else:
            return self.conv(x)


class MobileNetV3(nn.Module):
    def __init__(self, cfgs, mode,width_mult=1.):
        super(MobileNetV3, self).__init__()
        # setting of inverted residual blocks
        self.cfgs = cfgs
        assert mode in ['large', 'small']

        # building first layer
        input_channel = _make_divisible(16 * width_mult, 8)
        layers = [conv_3x3_bn(3, input_channel, 2)]
        # building inverted residual blocks
        block = InvertedResidual
        for k, exp_size, c, use_se, use_hs, s in self.cfgs:
            output_channel = _make_divisible(c * width_mult, 8)
            layers.append(block(input_channel, exp_size, output_channel, k, s, use_se, use_hs))
            input_channel = output_channel
        self.features = nn.Sequential(*layers)
        self.load_backbone(
            torch.load(config.backbone_imagenet_pretrain,map_location=torch.device('cpu')))

    def forward(self, x):
        x = self.features[:4](x)
        low_level=x
        high_level = self.features[4:](x)
        return low_level,high_level

    def load_backbone(self,pretrained_dict):
        # step2: get model state_dict
        model_dict = self.state_dict()
        # step3: remove pretrained_dict params which is not in model_dict
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if (k in model_dict)}
        # step4: update model_dict using pretrained_dict
        model_dict.update(pretrained_dict)
        # step5: update model using model_dict
        self.load_state_dict(model_dict)

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                n = m.weight.size(1)
                m.weight.data.normal_(0, 0.01)
                m.bias.data.zero_()

class conbine_feature(nn.Module):
    def __init__(self):
        super(conbine_feature, self).__init__()
        self.up2_high = DilatedParallelConvBlockD2(32, 16)
        self.up2_low = nn.Conv2d(24, 16, 1, stride=1, padding=0,bias=False)
        self.up2_bn2 = nn.BatchNorm2d(16)
        self.up2_act = nn.PReLU(16)
        self.refine=nn.Sequential(nn.Conv2d(16,16,3,padding=1,bias=False),nn.BatchNorm2d(16),nn.PReLU())

    def forward(self, low_fea,high_fea):
        high_fea = self.up2_high(high_fea)
        low_fea = self.up2_bn2(self.up2_low(low_fea))
        refine_feature = self.refine(self.up2_act(high_fea+low_fea))
        return refine_feature


class DilatedParallelConvBlockD2(nn.Module):
    def __init__(self, nIn, nOut, add=False):
        super(DilatedParallelConvBlockD2, self).__init__()
        n = int(np.ceil(nOut / 2.))
        n2 = nOut - n

        self.conv0 = nn.Conv2d(nIn, nOut, 1, stride=1, padding=0, dilation=1, bias=False)
        self.conv1 = nn.Conv2d(n, n, 3, stride=1, padding=1, dilation=1, bias=False)
        self.conv2 = nn.Conv2d(n2, n2, 3, stride=1, padding=2, dilation=2, bias=False)

        self.bn = nn.BatchNorm2d(nOut)
        #self.act = nn.PReLU(nOut)
        self.add = add

    def forward(self, input):
        in0 = self.conv0(input)
        in1, in2 = torch.chunk(in0, 2, dim=1)
        b1 = self.conv1(in1)
        b2 = self.conv2(in2)
        output = torch.cat([b1, b2], dim=1)

        if self.add:
            output = input + output
        output = self.bn(output)
        #output = self.act(output)

        return output

class Fastnet(nn.Module):
    def __init__(self):
        super(Fastnet, self).__init__()
        self.feature_extractor=mobilenetv3_large()
        self.dilation_conv_group=LightRFB()
        self.combine=conbine_feature()
        self.SegNIN = nn.Sequential(nn.Dropout2d(0.1),nn.Conv2d(16, 1, kernel_size=1,bias=False))

    def forward(self, x):
        if len(x.shape)==4:
            origin_shape=x.shape
            low_feature,high_feature=self.feature_extractor(x)
            high_feature=self.dilation_conv_group(high_feature)
            high_feature = F.interpolate(high_feature, size=(low_feature.shape[-2], low_feature.shape[-1]),
                                         mode="bilinear",
                                         align_corners=False)
            out = self.combine(low_feature, high_feature)
            out = torch.sigmoid(F.interpolate(self.SegNIN(out), size=(origin_shape[-2], origin_shape[-1]), mode="bilinear",
                                              align_corners=False))
        else:
            origin_shape=x.shape
            x=x.view(-1,*origin_shape[2:])
            low_feature,high_feature=self.feature_extractor(x)
            high_feature=self.dilation_conv_group(high_feature)
            high_feature = F.interpolate(high_feature, size=(low_feature.shape[-2], low_feature.shape[-1]),
                                         mode="bilinear",
                                         align_corners=False)
            out = self.combine(low_feature, high_feature)
            out = torch.sigmoid(
                F.interpolate(self.SegNIN(out), size=(origin_shape[-2], origin_shape[-1]), mode="bilinear",
                              align_corners=False))
        return out


def mobilenetv3_large(**kwargs):
    """
    Constructs a MobileNetV3-Large model
    """
    cfgs = [
        # k, t, c, SE, NL, s 
        [3,  16,  16, 0, 0, 1],
        [3,  64,  24, 0, 0, 2],
        [3,  72,  24, 0, 0, 1],
        [5,  72,  40, 1, 0, 1],
        [5, 120,  40, 1, 0, 1],
        [5, 120,  40, 1, 0, 1],
        [3, 240,  80, 0, 1, 2],
        [3, 200,  80, 0, 1, 1],
        [3, 184,  80, 0, 1, 1],
        [3, 184,  80, 0, 1, 1],
        [3, 480, 112, 1, 1, 1],
        [3, 672, 112, 1, 1, 1],
        [5, 672, 160, 1, 1, 1],
        [5, 672, 160, 1, 1, 1],
        [5, 960, 160, 1, 1, 1]
    ]
    return MobileNetV3(cfgs, mode='large', **kwargs)

if __name__=="__main__":
    from torchstat import stat
    a=torch.zeros(1,3,224,336)
    mobile=Fastnet()
    mobile(a)
    dsa


    print(mobile(a)[0].shape)
    total_paramters = sum([np.prod(p.size()) for p in mobile.parameters()])
    print('Total network parameters: ' + str(total_paramters))