import torch
import torch.nn as nn


def autopad(k, p=None, d=1):
    if d > 1:
        k = d * (k - 1) + 1 if isinstance(k, int) else [d * (x - 1) + 1 for x in k]  # actual kernel-size
    if p is None:
        p = k // 2 if isinstance(k, int) else [x // 2 for x in k] # actual padding
    return p


class Conv(nn.Module):
    default_act = nn.SiLU()

    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, groups=1, dilation=1, act=True):
        super(Conv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, 
                              kernel_size=kernel_size, stride=stride, 
                              padding=autopad(kernel_size, padding, dilation), 
                              groups=groups, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        self.act = self.default_act if act is True else act if isinstance(act, nn.Module) else nn.Identity()

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.act(x)
        return x


class Focus(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=1, stride=1, padding=0, groups=1, act=True):
        super(Focus, self).__init__()
        self.conv = Conv(in_channels*4, out_channels, kernel_size, stride, padding, groups, act)

    def forward(self, x):
        return self.conv(torch.cat([x[..., ::2, ::2], x[..., ::2, 1::2], x[..., 1::2, ::2], x[..., 1::2, 1::2]], 1))


class GhostConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=1, stride=1, groups=1, act=True):
        super(GhostConv, self).__init__()
        c = out_channels // 2
        self.conv1 = Conv(in_channels, c, kernel_size, stride, None, groups, act)
        self.conv2 = Conv(c, c, 5, 1, None, c, act)

    def forward(self, x):
        y = self.conv1(x)
        return torch.cat([y, self.conv2(y)], 1)


class SPPF(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=5):
        super(SPPF, self).__init__()
        c = out_channels // 2
        self.conv1 = Conv(in_channels, c, 1, 1)
        self.conv2 = Conv(c * 4, out_channels, 1, 1)
        self.pool = nn.MaxPool2d(kernel_size, stride=1, padding=kernel_size // 2)

    def forward(self, x):
        x = self.conv1(x)
        y1 = self.pool(x)
        y2 = self.pool(y1)
        return self.conv2(torch.cat([x, y1, y2, self.pool(y2)], 1))

