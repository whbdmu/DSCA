import torch
import torch.nn as nn

# Domain-specific BatchNorm


class DSBN2d(nn.Module):
    def __init__(self, planes):
        super().__init__()
        self.num_features = planes
        self.BN_S = nn.BatchNorm2d(planes)
        self.BN_T = nn.BatchNorm2d(planes)

    def forward(self, x, is_source):
        if is_source:
            return self.BN_S(x)
        else:
            return self.BN_T(x)


class DSBN1d(nn.Module):
    def __init__(self, planes):
        super().__init__()
        self.num_features = planes
        self.BN_S = nn.BatchNorm1d(planes)
        self.BN_T = nn.BatchNorm1d(planes)

    def forward(self, x, is_source):
        if is_source:
            return self.BN_S(x)
        else:
            return self.BN_T(x)


def convert_dsbn(model):
    for _, (child_name, child) in enumerate(model.named_children()):
        assert not next(model.parameters()).is_cuda
        if isinstance(child, nn.BatchNorm2d):
            m = DSBN2d(child.num_features)
            m.BN_S.load_state_dict(child.state_dict())
            m.BN_T.load_state_dict(child.state_dict())
            setattr(model, child_name, m)
            print("convert_dsbn 2D")
            print(child_name)
        elif isinstance(child, nn.BatchNorm1d):
            m = DSBN1d(child.num_features)
            m.BN_S.load_state_dict(child.state_dict())
            m.BN_T.load_state_dict(child.state_dict())
            setattr(model, child_name, m)
            print("convert_dsbn 1D")
            print(child_name)
        else:
            convert_dsbn(child)


def convert_bn(model, use_target=True):
    for _, (child_name, child) in enumerate(model.named_children()):
        assert not next(model.parameters()).is_cuda
        if isinstance(child, DSBN2d):
            m = nn.BatchNorm2d(child.num_features)
            if use_target:
                m.load_state_dict(child.BN_T.state_dict())
            else:
                m.load_state_dict(child.BN_S.state_dict())
            setattr(model, child_name, m)
        elif isinstance(child, DSBN1d):
            m = nn.BatchNorm1d(child.num_features)
            if use_target:
                m.load_state_dict(child.BN_T.state_dict())
            else:
                m.load_state_dict(child.BN_S.state_dict())
            setattr(model, child_name, m)
        else:
            convert_bn(child, use_target=use_target)
