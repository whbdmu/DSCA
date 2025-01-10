from abc import ABC
import torch
import torch.nn.functional as F
from torch import nn, autograd


class CM(autograd.Function):
    @staticmethod
    def forward(ctx, inputs, targets, features, momentum):
        ctx.features = features
        ctx.momentum = momentum
        ctx.save_for_backward(inputs, targets)
        outputs = inputs.mm(ctx.features.t())

        return outputs

    @staticmethod
    def backward(ctx, grad_outputs):
        inputs, targets = ctx.saved_tensors
        grad_inputs = None
        if ctx.needs_input_grad[0]:
            grad_inputs = grad_outputs.mm(ctx.features)

        # momentum update
        for x, y in zip(inputs, targets):
            ctx.features[y] = ctx.momentum * ctx.features[y] + (1.0 - ctx.momentum) * x
            ctx.features[y] /= ctx.features[y].norm()

        return grad_inputs, None, None, None


def cm(inputs, indexes, features, momentum=0.5):
    return CM.apply(inputs, indexes, features, torch.Tensor([momentum]).to(inputs.device))


class ClusterProxyMemory(nn.Module, ABC):
    def __init__(self, num_features, num_samples, source_classes, temp=0.05, momentum=0.2):
        super().__init__()
        self.num_features = num_features
        self.num_samples = num_samples
        self.source_classes = source_classes
        self.momentum = momentum
        self.temp = temp

        self.register_buffer("features", torch.zeros(num_samples, num_features))

    def forward(self, inputs, targets, is_source=True):
        targets = torch.cat(targets)
        targets = targets - 1
        inds = targets >= 0
        targets = targets[inds]
        for i in range(len(targets)):
            if (targets[i] == 5554) and is_source:
                targets[i] = self.source_classes - 1

        valid_inputs = inputs[inds.unsqueeze(1).expand_as(inputs)].view(-1, self.num_features)
        outputs = cm(valid_inputs, targets, self.features, self.momentum)

        outputs /= self.temp
        loss = F.cross_entropy(outputs, targets, ignore_index=self.source_classes - 1)

        return loss
