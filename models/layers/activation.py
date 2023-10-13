import numpy as np
import torch
from torch import Tensor, nn


class RAbs(nn.Module):
    def __init__(self, **kwargs) -> None:
        super(RAbs, self).__init__()

    def forward(self, x: Tensor) -> Tensor:
        x = x.abs()
        num_features = x.shape[1]
        p1, p2 = torch.split(x, num_features // 2, dim=1)
        x = torch.cat([p1 + p2, p1 - p2], dim=1)
        return x / 2**.5

    def lipschitz(self) -> float:
        return 1.

    def extra_repr(self) -> str:
        return 'dim=1'


class MinMax(nn.Module):
    def __init__(self, dim: int = 1, **kwargs) -> None:
        super(MinMax, self).__init__()
        self.dim = dim

    def forward(self, x: Tensor) -> Tensor:

        num_features = x.size(self.dim) // 2
        part1, part2 = torch.split(x, num_features, dim=self.dim)

        index = part1 > part2
        max_part = torch.where(index, part1, part2)
        min_part = torch.where(index, part2, part1)

        x = torch.cat([max_part, min_part], dim=self.dim)
        return x

    def lipschitz(self) -> float:
        return 1.

    def extra_repr(self) -> str:
        return f'dim={self.dim}'


class HalfAbs(nn.Module):
    def __init__(self, dim: int = 1, **kwargs) -> None:
        super(HalfAbs, self).__init__()
        self.dim = dim

    def forward(self, x: Tensor) -> Tensor:
        num_features = x.shape[self.dim]
        part1, part2 = torch.split(x, num_features // 2, dim=self.dim)
        x = torch.cat([part1, part2.abs()], dim=self.dim)
        return x

    def lipschitz(self) -> float:
        return 1.

    def extra_repr(self) -> str:
        return f'dim={self.dim}'


class ReLU(nn.LeakyReLU):
    def __init__(self, negative_slope: float = 0.0, **kwargs) -> None:
        super(ReLU, self).__init__(negative_slope=negative_slope)
        self.negative_slope = negative_slope

    def lipschitz(self) -> float:
        return 1.

    def extra_repr(self) -> str:
        return f'negative_slope={self.negative_slope}'


class HouseHolder_Order_2(nn.Module):
    """Copied from:

    https://github.com/singlasahil14/SOC/blob/561c7acb240bfc83b3217a72d67c42066dea5639/custom_activations.py#L39
    """
    def __init__(self, channels, **kwargs):
        super(HouseHolder_Order_2, self).__init__()
        assert (channels % 2) == 0
        self.num_groups = channels // 2

        self.theta0 = nn.Parameter(
            (np.pi * torch.rand(self.num_groups)).cuda(), requires_grad=True)
        self.theta1 = nn.Parameter(
            (np.pi * torch.rand(self.num_groups)).cuda(), requires_grad=True)
        self.theta2 = nn.Parameter(
            (np.pi * torch.rand(self.num_groups)).cuda(), requires_grad=True)

    def forward(self, z, axis=1):
        theta0 = torch.clamp(self.theta0.view(1, -1, 1, 1), 0., 2 * np.pi)

        x, y = z.split(z.shape[axis] // 2, axis)
        z_theta = (torch.atan2(y, x) - (0.5 * theta0)) % (2 * np.pi)

        theta1 = torch.clamp(self.theta1.view(1, -1, 1, 1), 0., 2 * np.pi)
        theta2 = torch.clamp(self.theta2.view(1, -1, 1, 1), 0., 2 * np.pi)
        theta3 = 2 * np.pi - theta1
        theta4 = 2 * np.pi - theta2

        ang1 = 0.5 * (theta1)
        ang2 = 0.5 * (theta1 + theta2)
        ang3 = 0.5 * (theta1 + theta2 + theta3)
        ang4 = 0.5 * (theta1 + theta2 + theta3 + theta4)

        select1 = torch.logical_and(z_theta >= 0, z_theta < ang1)
        select2 = torch.logical_and(z_theta >= ang1, z_theta < ang2)
        select3 = torch.logical_and(z_theta >= ang2, z_theta < ang3)
        select4 = torch.logical_and(z_theta >= ang3, z_theta < ang4)

        a1 = x
        b1 = y

        a2 = x * torch.cos(theta0 + theta1) + y * torch.sin(theta0 + theta1)
        b2 = x * torch.sin(theta0 + theta1) - y * torch.cos(theta0 + theta1)

        a3 = x * torch.cos(theta2) + y * torch.sin(theta2)
        b3 = -x * torch.sin(theta2) + y * torch.cos(theta2)

        a4 = x * torch.cos(theta0) + y * torch.sin(theta0)
        b4 = x * torch.sin(theta0) - y * torch.cos(theta0)

        a = (a1 * select1) + (a2 * select2) + (a3 * select3) + (a4 * select4)
        b = (b1 * select1) + (b2 * select2) + (b3 * select3) + (b4 * select4)

        z = torch.cat([a, b], dim=axis)
        return z

    def extra_repr(self):
        return 'channels={}'.format(self.num_groups * 2)

    def lipschitz(self) -> float:
        return 1.


def build_activation(act_name: str, **kwargs) -> nn.Module:
    act_name = act_name.lower()
    if act_name == 'minmax':
        act_fn = MinMax
    elif act_name == 'halfabs':
        act_fn = HalfAbs
    elif act_name == 'relu':
        act_fn = ReLU
    elif 'householder' in act_name:
        act_fn = HouseHolder_Order_2
    elif act_name == 'rabs':
        act_fn = RAbs
    else:
        raise ValueError('Unsupported `act_name`')

    return act_fn(**kwargs)
