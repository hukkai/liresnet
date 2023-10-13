import torch
import torch.distributed.nn as dist_nn
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules.utils import _pair

from .cholesky_grad import CholeskyOrth


class LiResConv(nn.Module):
    def __init__(self,
                 width: int,
                 depth: int,
                 input_size: int,
                 activation: nn.Module,
                 centering: bool = True,
                 num_lc_iter: int = 10) -> None:
        super(LiResConv, self).__init__()
        weights = torch.randn(depth, width, width, 3, 3)
        weights = weights / (width * 9)
        self.weights = nn.Parameter(weights)

        gamma = torch.ones(depth, width, 1, 1, 1)
        self.gamma = nn.Parameter(gamma)

        self.bias = nn.Parameter(torch.zeros(depth, width))
        if centering:
            running_mean = torch.zeros(depth, width)
            self.register_buffer('running_mean', running_mean)
        else:
            self.running_mean = 0

        identity = torch.zeros(width, width, 3, 3)
        identity[:, :, 1, 1] = torch.eye(width)
        identity = torch.stack([identity for _ in range(depth)])
        self.register_buffer('identity', identity)

        init_x = torch.ones(1, depth * width, *_pair(input_size))
        self.register_buffer('init_x', init_x)

        self.act = activation

        self.depth = depth
        self.width = width
        self.scale = depth**-.5
        self.num_lc_iter = num_lc_iter
        self.centering = centering

    def get_weight(self):
        W = self.weights * self.gamma
        return self.identity + W * self.scale

    def forward(self, x):
        weights = self.get_weight()
        if not (self.centering and self.training):
            biases = self.bias - self.running_mean
            for weight, bias in zip(weights, biases):
                x = F.conv2d(x, weight, bias, padding=1)
                x = self.act(x)
            return x

        weights = weights - self.identity
        all_means = []
        for weight, bias in zip(weights, self.bias):
            out = F.conv2d(x, weight, padding=1)
            mean = out.mean((0, 2, 3))
            all_means.append(mean.detach())
            out = out + (bias - mean).view(-1, 1, 1)
            x = self.act(x + out)

        all_means = torch.stack(all_means)
        self.running_mean += (all_means - self.running_mean) * 0.1
        return x

    def lipschitz(self):
        W = self.get_weight().reshape(-1, self.width, 3, 3)
        x = self.init_x.data
        for _ in range(self.num_lc_iter):
            x = F.conv2d(x, W, padding=1, groups=self.depth)
            x = F.conv_transpose2d(x, W, padding=1, groups=self.depth)
            x = x.reshape(self.depth, -1)
            x = F.normalize(x, dim=1)
            x = x.reshape(self.init_x.shape)

        x = x.detach()

        self.init_x += (x - self.init_x).detach()
        x = F.conv2d(x, W, padding=1, groups=self.depth)
        norm = x.reshape(self.depth, -1).norm(dim=1)
        return norm.prod()

    def extra_repr(self) -> str:
        return f'depth={self.depth}, ' \
               f'width={self.width}, ' \
               f'centering={self.centering}'


class LiResMLP(nn.Module):
    def __init__(self,
                 num_features: int,
                 depth: int,
                 activation: nn.Module,
                 num_lc_iter: int = 10) -> None:
        super(LiResMLP, self).__init__()
        weights = torch.randn(depth, num_features, num_features)
        weights = weights / num_features
        self.weights = nn.Parameter(weights)

        self.gamma = nn.Parameter(torch.ones(depth, num_features, 1))
        self.bias = nn.Parameter(torch.zeros(depth, num_features))

        self.register_buffer('identity', torch.eye(num_features))
        self.register_buffer('eval_weight', torch.zeros_like(weights))

        self.act = activation

        self.scale = depth**-.5
        self.num_lc_iter = num_lc_iter
        self.depth = depth
        self.width = num_features

        flag = torch.distributed.is_initialized()
        self.flag = flag and depth % torch.distributed.get_world_size() == 0

    def get_weight(self):
        if self.flag:
            rank = torch.distributed.get_rank()
            world = torch.distributed.get_world_size()
            num_per_gpu = self.depth // world
            index = range(rank * num_per_gpu, (rank + 1) * num_per_gpu)
            _W = self.identity.data + self.weights[index] * self.gamma[
                index] * self.scale
            _W = CholeskyOrth(_W).contiguous()
            W = dist_nn.functional.all_gather(_W)
            W = torch.cat(W, dim=0)
            return W
        W = self.identity.data + self.weights * self.gamma * self.scale
        W = CholeskyOrth(W)
        return W

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.training:
            weights = self.get_weight()
        else:
            weights = self.eval_weight

        for weight, bias in zip(weights, self.bias):
            x = F.linear(x, weight, bias)
            x = self.act(x)
        return x

    def lipschitz(self):

        if self.training:
            return 1.0

        weights = self.get_weight()
        return torch.linalg.matrix_norm(weights, ord=2).prod()

    def train(self, mode):
        self.training = mode
        if mode is False:
            weights = self.get_weight().detach()
            self.eval_weight += weights - self.eval_weight
        return self

    def extra_repr(self) -> str:
        return f'depth={self.depth}, width={self.width}'
