from typing import Optional, Tuple, Union

import torch
import torch.nn.functional as F
from torch import Tensor, nn
from torch.nn.modules.utils import _pair


class Conv2d(nn.Conv2d):
    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 kernel_size: Union[int, Tuple[int, int]],
                 stride: Union[int, Tuple[int, int]] = 1,
                 padding: Optional[Union[int, Tuple[int, int]]] = None,
                 groups: int = 1,
                 bias: bool = True,
                 num_lc_iter: int = 10,
                 input_size: Union[int, Tuple[int, int]] = 32,
                 orthogonal_init: bool = True,
                 **kwargs) -> None:
        if padding is None:
            if type(kernel_size) == int:
                padding = kernel_size // 2
            elif len(kernel_size) == 2:
                padding = (kernel_size[0] // 2, kernel_size[1] // 2)
        super(Conv2d, self).__init__(in_channels=in_channels,
                                     out_channels=out_channels,
                                     kernel_size=kernel_size,
                                     stride=stride,
                                     padding=padding,
                                     groups=groups,
                                     bias=bias)

        if orthogonal_init:
            nn.init.orthogonal_(self.weight)

        self.num_lc_iter = num_lc_iter
        self.input_size = _pair(input_size)

        init_x = torch.ones(1, self.in_channels, *self.input_size)
        self.register_buffer('init_x', init_x)

        self.output_padding = self.compute_output_padding()

    def lipschitz(self) -> Tensor:
        x = self.init_x.data
        for _ in range(self.num_lc_iter):
            x = F.conv2d(x,
                         self.weight,
                         bias=None,
                         stride=self.stride,
                         padding=self.padding,
                         groups=self.groups)
            x = F.conv_transpose2d(x,
                                   self.weight,
                                   bias=None,
                                   stride=self.stride,
                                   padding=self.padding,
                                   output_padding=self.output_padding,
                                   groups=self.groups)
            x = F.normalize(x, dim=(1, 2, 3))

        x = x.detach()
        self.init_x += (x - self.init_x).detach()
        x = F.conv2d(x,
                     self.weight,
                     bias=None,
                     stride=self.stride,
                     padding=self.padding,
                     groups=self.groups)
        return x.norm()

    def compute_output_padding(self) -> Tuple:
        h, w = self.input_size
        s1, s2 = _pair(self.stride)
        k1, k2 = _pair(self.kernel_size)
        p1, p2 = _pair(self.padding)
        o1 = (h + 2 * p1 - k1) // s1 * s1 - 2 * p1 + k1
        o2 = (w + 2 * p2 - k2) // s2 * s2 - 2 * p2 + k2
        return h - o1, w - o2

    def get_weight(self) -> Tensor:
        return self.weight
