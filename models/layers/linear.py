import torch
import torch.nn.functional as F
from torch import Tensor, nn


class Linear(nn.Linear):
    def __init__(self,
                 in_features: int,
                 out_features: int,
                 bias: bool = True,
                 num_lc_iter: int = 10,
                 **kwargs) -> None:
        super(Linear, self).__init__(in_features, out_features, bias)
        self.num_lc_iter = num_lc_iter

        init_x = torch.ones(1, self.in_features)
        self.register_buffer('init_x', init_x)

    def lipschitz(self) -> Tensor:
        x = self.init_x.data
        W = self.weight
        WT = W.T.contiguous()

        for _ in range(self.num_lc_iter * 2):
            x = F.linear(x, W)
            x = F.linear(x, WT)
            x = F.normalize(x, dim=1)

        x = x.detach()
        self.init_x += (x - self.init_x).detach()
        x = F.linear(x, W)
        return x.norm()
