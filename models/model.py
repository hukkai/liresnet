from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .blocks import LiResConv, LiResMLP
from .layers import Conv2d, Sequential, build_activation


def build_stem(input_size: int, width: int, act_name: str) -> nn.Module:
    """Build the first layer to inputs."""

    if input_size == 64:  # Tiny ImageNet
        conv = Conv2d(3, width, kernel_size=5, stride=2, input_size=64)
        output_size = 32
    elif input_size == 32:  # CIFAR10/100
        conv = Conv2d(3, width, kernel_size=5, stride=2, input_size=32)
        output_size = 16
    # We use the patchify stem as in ViTs for ImageNet
    elif input_size == 224:
        patch_size = round((width / 3)**.5)
        conv = Conv2d(3,
                      width,
                      kernel_size=patch_size,
                      stride=patch_size,
                      padding=0,
                      input_size=224)
        output_size = 224 // patch_size
    else:
        raise ValueError('Unsupported `input_size`!')

    activation = build_activation(act_name, dim=1, channels=width)
    stem_layer = Sequential(conv, activation)
    return stem_layer, output_size


class Map2Vec(nn.Module):
    def __init__(self, feat_dim: int, feat_size: int, out_dim: int,
                 activation: nn.Module) -> None:
        super(Map2Vec, self).__init__()
        assert feat_size % 4 == 0
        mid_size = feat_size // 4
        mid_dim = feat_dim * mid_size**2

        kernel = torch.randn(feat_dim, feat_dim * 16)
        kernel = kernel / feat_dim**.5 / 4
        self.kernel = nn.Parameter(kernel)

        weight = torch.randn(out_dim, mid_dim) / mid_dim**.5
        self.weight = nn.Parameter(weight)

        self.bias = nn.Parameter(torch.zeros(out_dim))

        self.feat_dim = feat_dim
        self.mid_size = mid_size
        self.activation = activation

    def get_weight(self):
        Sigma = self.kernel @ self.kernel.T
        eps = Sigma.diag().mean().div(1000.).item()
        Sigma = Sigma + eps * torch.eye(
            Sigma.shape[0], device=Sigma.device, dtype=Sigma.dtype)
        L = torch.linalg.cholesky(Sigma)
        kernel_ = torch.linalg.solve_triangular(L, self.kernel, upper=False)
        kernel_ = kernel_.reshape(self.feat_dim, self.feat_dim, 4, 4)

        Sigma = self.weight @ self.weight.T
        eps = Sigma.diag().mean().div(1000.).item()
        Sigma = Sigma + eps * torch.eye(
            Sigma.shape[0], device=Sigma.device, dtype=Sigma.dtype)
        L = torch.linalg.cholesky(Sigma)
        weight_ = torch.linalg.solve_triangular(L, self.weight, upper=False)
        return kernel_, weight_

    def forward(self, x):
        kernel, weight = self.get_weight()
        x = F.conv2d(x, kernel, stride=4)
        x = x.reshape(x.shape[0], -1)
        x = F.linear(x, weight, self.bias)
        x = self.activation(x)
        return x

    def lipschitz(self):
        if self.training:
            return 1.

        kernel, weight = self.get_weight()
        kernel = kernel.reshape(self.feat_dim, -1)
        lc = kernel.svd().S.max()
        lc = lc * weight.svd().S.max()
        return lc.item()


class head(nn.Linear):
    """Build the head to outputs."""
    def __init__(self, num_features: int, num_classes: int,
                 use_lln: bool) -> None:
        super(head, self).__init__(num_features, num_classes)
        self.use_lln = use_lln
        # self.register_buffer('mean', torch.zeros(num_features)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        if self.training:
            mu = x.mean(dim=0)
            x = x - mu
            self.mean += (mu - self.mean).detach() * 0.1
        else:
            x = x - self.mean
        """
        weight = self.get_weight()
        x = F.linear(x, weight, self.bias)
        return x

    def get_weight(self):
        if self.use_lln:
            weight = F.normalize(self.weight, dim=1)
        else:
            weight = self.weight
        return weight


class SDPBasedLipschitzLinearLayer(nn.Module):
    def __init__(self, cin, cout, epsilon=1e-6):
        super(SDPBasedLipschitzLinearLayer, self).__init__()

        self.activation = nn.ReLU(inplace=False)
        self.weights = nn.Parameter(torch.empty(cout, cin))
        self.bias = nn.Parameter(torch.empty(cout))
        self.q = nn.Parameter(torch.rand(cout))

        nn.init.xavier_normal_(self.weights)
        fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weights)
        bound = 1 / np.sqrt(fan_in)
        nn.init.uniform_(self.bias, -bound, bound)  # bias init

        self.epsilon = epsilon

    def forward(self, x):
        res = F.linear(x, self.weights, self.bias)
        res = self.activation(res)
        q_abs = torch.abs(self.q) + self.epsilon
        q = q_abs[None, :]
        q_inv = (1 / q_abs)[:, None]
        T = 2 / torch.abs(q_inv * self.weights @ self.weights.T * q).sum(1)
        res = T * res
        res = F.linear(res, self.weights.t())
        out = x - res
        return out

    def lipschitz(self):
        return 1.0


class GloroNet(nn.Module):
    def __init__(self,
                 depth: int = 12,
                 width: int = 128,
                 input_size: int = 32,
                 num_classes: int = 10,
                 num_lc_iter: int = 10,
                 act_name: str = 'MinMax',
                 use_lln: bool = True,
                 **kwargs):
        super(GloroNet, self).__init__()

        stem, feature_size = build_stem(input_size, width, act_name)
        self.stem = stem
        kwargs['input_size'] = feature_size

        activation = build_activation(act_name, dim=1, channels=width)
        self.conv = LiResConv(width=width,
                              depth=depth,
                              input_size=feature_size,
                              activation=activation)

        self.depth = depth

        out_dim = 2048

        self.neck = Map2Vec(feat_dim=width,
                            feat_size=feature_size,
                            out_dim=out_dim,
                            activation=activation)
        self.linear = LiResMLP(num_features=out_dim,
                               depth=8,
                               activation=activation)

        self.head = head(out_dim, num_classes, use_lln)

        self.num_lc_iter = num_lc_iter
        self.set_num_lc_iter()

    def set_num_lc_iter(self, num_lc_iter: Optional[int] = None) -> None:
        if num_lc_iter is None:
            num_lc_iter = self.num_lc_iter
        for m in self.modules():
            setattr(m, 'num_lc_iter', num_lc_iter)

    def forward(self,
                x: torch.Tensor,
                return_feat: bool = False) -> torch.Tensor:
        """
        Args:
            x (torch.Tensor): input image tensors in [0, 1]
            return_feat (bool): if true, only return the extracted features.

        """
        x = x.sub(.5)
        x = self.stem(x)
        x = self.conv(x)
        x = self.neck(x)
        x = self.linear(x)
        if return_feat:
            return x
        x = self.head(x)
        return x

    def sub_lipschitz(self) -> torch.Tensor:
        """Compute the lipschitz constant of the model except the head."""
        lc = self.stem.lipschitz()
        lc = lc * self.neck.lipschitz()
        lc = lc * self.conv.lipschitz()
        lc = lc * self.linear.lipschitz()
        return lc
