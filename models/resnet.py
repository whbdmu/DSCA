from collections import OrderedDict
from copy import deepcopy
from pickle import FALSE
from sympy import false, true
import torch.nn.functional as F
import torchvision
import torch
import math
from torch import nn
from torch.nn.modules.utils import _pair
from spcl.models.dsbn import DSBN2d, DSBN1d
from utils import add_module_after_block


class Backbone(nn.Module):
    def __init__(self, resnet, use_filter):
        super().__init__()

        # resnet
        self.conv1 = resnet.conv1
        self.bn1 = resnet.bn1
        self.relu = resnet.relu
        self.maxpool = resnet.maxpool
        # Add filter after first block
        if use_filter:
            self.layer1 = add_module_after_block(resnet.layer1, 1, AdaptiveFilter(256, gap_size=(1, 1)))
            self.layer2 = add_module_after_block(resnet.layer2, 1, AdaptiveFilter(512, gap_size=(1, 1)))
            self.layer3 = add_module_after_block(resnet.layer3, 1, AdaptiveFilter(1024, gap_size=(1, 1)))
        else:
            self.layer1 = resnet.layer1
            self.layer2 = resnet.layer2
            self.layer3 = resnet.layer3

        self.out_channels = 1024

    def _forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        if torch.isnan(x).int().sum() > 0:
            print(torch.isnan(x).int().sum())
        return x

    def forward(self, x):
        feat = self._forward(x)
        return OrderedDict([["feat_res4", feat]])


class Res5Head(nn.Module):
    def __init__(self, layer4, use_filter):
        super().__init__()  # res5
        self.layer4 = deepcopy(layer4)
        if use_filter:
            self.layer4 = add_module_after_block(self.layer4, 1, AdaptiveFilter(2048, gap_size=(1, 1)))
        self.out_channels = [1024, 2048]

    def forward(self, x):
        feat = self.layer4(x)
        x = F.adaptive_max_pool2d(x, 1)
        feat = F.adaptive_max_pool2d(feat, 1)
        return OrderedDict([["feat_res4", x], ["feat_res5", feat]])


class ReidRes5Head(nn.Module):
    def __init__(self, layer4, use_filter):
        super().__init__()  # res5
        self.layer4 = deepcopy(layer4)
        if use_filter:
            self.layer4 = add_module_after_block(self.layer4, 1, AdaptiveFilter(2048, gap_size=(1, 1)))
        self.out_channels = [1024, 2048]

    def bottleneck_forward(self, bottleneck, x, is_source):
        identity = x

        out = bottleneck.conv1(x)
        out = bottleneck.bn1(out, is_source)
        out = bottleneck.relu(out)
        out = bottleneck.conv2(out)
        out = bottleneck.bn2(out, is_source)
        out = bottleneck.relu(out)
        out = bottleneck.conv3(out)
        out = bottleneck.bn3(out, is_source)
        if bottleneck.downsample is not None:
            for module in bottleneck.downsample:
                if not isinstance(module, DSBN2d):
                    identity = module(x)
                else:
                    identity = module(identity, is_source)
        out += identity
        out = bottleneck.relu(out)
        return out

    def forward(self, x, is_source=True):
        # x = self.att(x)
        # 对于reid head的dsbn特殊处理
        # 需要取出没有child的module组成list一次执行，可以避免递归中重新实现所有带is_source的forward
        # Bottleneck的forward步骤有缺失
        module_seq = []
        for _, (_, child) in enumerate(self.named_modules()):
            if isinstance(child, torchvision.models.resnet.Bottleneck):
                module_seq.append(child)
            if isinstance(child, AdaptiveFilter):
                module_seq.append(child)

        feat = x.clone()
        for module in module_seq:
            if isinstance(module, AdaptiveFilter):
                feat = module(feat, is_source)
            else:
                feat = self.bottleneck_forward(module, feat, is_source)

        x = F.adaptive_max_pool2d(x, 1)
        feat = F.adaptive_max_pool2d(feat, 1)
        return OrderedDict([["feat_res4", x], ["feat_res5", feat]])


class AdaptiveFilter(nn.Module):
    def __init__(self, channel, gap_size, level=2):
        super().__init__()
        self.avg_gap = nn.AdaptiveAvgPool2d(gap_size)
        self.max_gap = nn.AdaptiveMaxPool2d(gap_size)
        self.mlp = nn.Sequential(
            nn.Linear(channel, channel),
            nn.BatchNorm1d(channel),
            nn.ReLU(inplace=True),
            nn.Linear(channel, channel),
            nn.BatchNorm1d(channel),
            nn.Sigmoid(),
        )

        self.epsilon = 1e-16
        self.level = level
        self.alpha = nn.Parameter(torch.ones(1, channel, 1, 1), requires_grad=True)
        self.beta = nn.Parameter(torch.zeros(1, channel, 1, 1), requires_grad=True)

    def compute_threshold(self, x_abs, is_source=None):
        x_avg = self.avg_gap(x_abs)
        x_avg = torch.flatten(x_avg, 1)
        x_max = self.max_gap(x_abs)
        x_max = torch.flatten(x_max, 1)

        x_max_scores_1 = self.mlp(x_max) if is_source is None else self._process_with_DSBN(x_max, is_source, self.mlp)
        channel_threshold = (x_avg * x_max_scores_1).unsqueeze(-1).unsqueeze(-1)

        return channel_threshold

    def _process_with_DSBN(self, features, is_source, body):
        for module in body:
            if isinstance(module, DSBN1d):
                features = module(features, is_source)
            else:
                features = module(features)
        return features

    def apply_filter(self, x_abs, threshold):
        up_mask = (x_abs > threshold).float()  # Mask tensor
        up_sub = x_abs * up_mask  # Raw hard threshold filter
        sqrt_p = torch.pow(
            F.relu(torch.pow(up_sub, self.level) - torch.pow(threshold, self.level)) + self.epsilon,
            1.0 / self.level,
        )  # High-Order Soft Threshold filter
        return (self.alpha * F.relu(up_sub - threshold) + self.beta * (sqrt_p - self.epsilon)) / (
            self.alpha + self.beta
        )

    def forward(self, x, is_source=None):
        x = x.float()
        x_raw = x
        x_abs = torch.abs(x)

        # Channel threshold
        channel_threshold = self.compute_threshold(x_abs, is_source)
        # Apply adaptive channel filter
        up_sub = self.apply_filter(x_abs, channel_threshold)

        return torch.mul(torch.sign(x_raw), up_sub)


def build_resnet(name="resnet50", pretrained=True):
    resnet = torchvision.models.resnet.__dict__[name](pretrained=pretrained)

    # freeze layers
    resnet.conv1.weight.requires_grad_(False)
    resnet.bn1.weight.requires_grad_(False)
    resnet.bn1.bias.requires_grad_(False)

    return (
        Backbone(resnet, use_filter=True),
        Res5Head(resnet.layer4, use_filter=False),
        ReidRes5Head(resnet.layer4, use_filter=False),
    )
