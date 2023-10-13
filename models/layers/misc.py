from typing import Optional, Tuple, Union

import torch
import torch.nn.functional as F
from torch import Tensor, nn
from torch.nn.modules.utils import _pair


class AvgPool2d(nn.AvgPool2d):
    def __init__(self,
                 kernel_size: Union[int, Tuple[int, int]],
                 stride: Union[int, Tuple[int, int]] = None,
                 padding: Union[int, Tuple[int, int]] = 0,
                 input_size: Union[int, Tuple[int, int]] = 32) -> None:
        super(AvgPool2d, self).__init__(kernel_size, stride, padding)
        self.input_size = _pair(input_size)

    def lipschitz(self) -> float:
        if hasattr(self, '_computed'):
            return self._computed

        x = torch.randn(1, 1, *self.input_size)
        weight = torch.ones(1, 1, *_pair(self.kernel_size))
        weight /= weight.sum()
        for _ in range(1000):
            x = F.conv2d(x,
                         weight,
                         bias=None,
                         stride=self.stride,
                         padding=self.padding)
            x = F.conv_transpose2d(x,
                                   weight,
                                   bias=None,
                                   stride=self.stride,
                                   padding=self.padding)
            x = F.normalize(x, dim=(1, 2, 3))

        x = F.conv2d(x,
                     weight,
                     bias=None,
                     stride=self.stride,
                     padding=self.padding)
        self._computed = x.norm().item()
        return self._computed


class InvertibleDownsampling(nn.Module):
    def __init__(self,
                 pool_size: Union[int, Tuple[int]] = 2,
                 **kwargs) -> None:
        super(InvertibleDownsampling, self).__init__()
        self.pool_size = _pair(pool_size)
        self.unfold = nn.Unfold(self.pool_size, stride=self.pool_size)

    def forward(self, x: Tensor) -> Tensor:
        B, C, H, W = x.shape
        x = self.unfold(x)
        x = x.view(B, -1, H // self.pool_size[0], W // self.pool_size[1])
        return x

    def lipschitz(self) -> float:
        return 1.


class Shift2D(nn.Module):
    def __init__(self, num_features: int, threshold: int = 8) -> None:
        super(Shift2D, self).__init__()
        self.num_features = num_features
        self.threshold = threshold

        self.bias = nn.Parameter(torch.zeros(num_features))
        self.register_buffer('init_mean', torch.zeros(num_features))
        self.register_buffer('count', torch.tensor(0))

    def forward(self, x):
        if self.count < self.threshold:
            mu = x.mean(dim=(0, 2, 3)).detach()
            self.init_mean += (mu - self.init_mean) / (self.count + 1)
            self.count += 1
            bias = self.bias.mul(1e-2).sub(self.init_mean)
        else:
            bias = self.bias.sub(self.init_mean)

        return x.add(bias[None, :, None, None])

    def lipschitz(self) -> float:
        return 1.


class Affine2D(nn.Module):
    def __init__(self, num_features: int) -> None:
        super(Affine2D, self).__init__()
        self.weight = nn.Parameter(torch.ones(num_features))

    def forward(self, x):
        return x.mul(self.weight[None, :, None, None])

    def lipschitz(self) -> Tensor:
        return self.weight.abs().max()


class Centering(nn.Module):
    def __init__(self,
                 num_features: int,
                 dim: int = 1,
                 ndim: Optional[int] = None,
                 **kwargs) -> None:
        super(Centering, self).__init__()
        self.register_buffer('running_mean', torch.zeros(num_features))
        self.bias = nn.Parameter(torch.zeros(num_features))
        self.dim = dim

        if type(ndim) == int:
            self.mean_dim, self.view_shape = self.get_shape(ndim, dim)
            self.ndim = ndim
        else:
            self.ndim = None

    def forward(self, x: Tensor) -> Tensor:
        if self.ndim is None:
            ndim = x.ndim
            mean_dim, view_shape = self.get_shape(ndim, self.dim)
        else:
            mean_dim, view_shape = self.mean_dim, self.view_shape

        if self.training:
            mu = x.mean(dim=mean_dim)
            x = x + (self.bias - mu).view(view_shape)

            self.running_mean += (mu - self.running_mean) * 0.01
        else:
            x = x + (self.bias - self.running_mean).view(view_shape)
        return x

    def lipschitz(self) -> float:
        return 1.

    @staticmethod
    def get_shape(ndim: int, dim: int) -> Tuple[Tuple, Tuple]:
        mean_dim = [_ for _ in range(ndim) if _ != dim]
        mean_dim = tuple(mean_dim)
        view_shape = [1] * ndim
        view_shape[dim] = -1
        return tuple(mean_dim), tuple(view_shape)

    def extra_repr(self) -> str:
        return f'dim={self.dim}'


class Scale(nn.Module):
    def __init__(self, init_scale: float = 1., **kwargs) -> None:
        super(Scale, self).__init__()
        self.scale = nn.Parameter(torch.ones(1))
        nn.init.constant_(self.scale, init_scale)

    def forward(self, x: Tensor) -> Tensor:
        return x * self.scale

    def lipschitz(self) -> Tensor:
        return self.scale.abs()


class Sequential(nn.Sequential):
    def __init__(self, *args) -> None:
        super(Sequential, self).__init__(*args)

    def lipschitz(self) -> Union[int, Tensor]:
        lc = 1
        for module in self:
            if hasattr(module, 'lipschitz'):
                lc = lc * module.lipschitz()
            else:
                torch.warnings.warn(
                    'Lipschitz not implemented for one module!')
        return lc


class Flatten(nn.Flatten):
    def __init__(self, start_dim: int = 1, end_dim: int = -1) -> None:
        super(Flatten, self).__init__(start_dim, end_dim)

    def lipschitz(self) -> float:
        return 1.
