from typing import Union

import torch
from torch import Tensor, nn


def trades_loss(model: nn.Module,
                x: Tensor,
                label: Tensor,
                eps: float,
                lc: Union[float, Tensor] = None,
                lambda_kl: float = 1.0,
                return_loss: bool = True,
                **kwargs):
    """
    Args:
        model (nn.Module): the trained model.
        x (Tensor): the input of the model.
        label (Tensor): the target of the model.
        eps (float): the robustness radius.
        lc (float or torch.Tensor): The lipschitz of the model backbone.
        lambda_kl (float): loss weight for the TRADES part.
        return_loss (bool): if True, compute and return the loss.
    """
    if hasattr(model, 'module'):
        head = model.module.head.get_weight()
    else:
        head = model.head.get_weight()
    y = model(x)
    pred = y.argmax(1)
    head_j = head[pred].unsqueeze(1)  # batch, 1, dim
    head_ji = head_j - head.unsqueeze(0)  # batch, num_class, dim
    head_ji = head_ji.norm(dim=-1)  # batch, num_class
    y_ = y + lc * eps * head_ji
    y_ = y_.scatter(1, pred.view(-1, 1), -10.**10)
    y_ = y_.max(1)[0].reshape(-1, 1)
    y_ = torch.cat([y, y_], dim=1)
    if return_loss:
        loss = nn.CrossEntropyLoss()(y, label)
        # If you are not clear why we compute the KL loss in this way,
        # please refer to https://github.com/hukkai/gloro_res/issues/2.
        KL_loss = y.log_softmax(dim=-1)[:, 0]
        KL_loss = KL_loss - y_.log_softmax(dim=-1)[:, 0]
        KL_loss = KL_loss.mean()
        loss = loss + KL_loss * lambda_kl
    else:
        loss = None
    return y, y_, loss


def margin_loss(model: nn.Module,
                x: Tensor,
                label: Tensor,
                eps: float,
                lc: Union[float, Tensor] = None,
                return_loss: bool = True,
                **kwargs):
    """
    Args:
        model (nn.Module): the trained model.
        x (Tensor): the input of the model.
        label (Tensor): the target of the model.
        eps (float): the robustness radius.
        lc (float or torch.Tensor): The lipschitz of the model backbone.
        return_loss (bool): if True, compute and return the loss.
    """
    if hasattr(model, 'module'):
        head = model.module.head.get_weight()
    else:
        head = model.head.get_weight()
    y = model(x)

    head_j = head[label].unsqueeze(1)  # batch, 1, dim
    head_ji = head_j - head.unsqueeze(0)  # batch, num_class, dim
    head_ji = head_ji.norm(dim=-1)  # batch, num_class
    margin = lc * head_ji
    y_ = y + eps * margin
    if return_loss:
        loss = nn.CrossEntropyLoss()(y_, label)
    else:
        loss = None
    return y, y_, loss


def emma_loss(model: nn.Module,
              x: Tensor,
              label: Tensor,
              eps: float,
              lc: Union[float, Tensor] = None,
              return_loss: bool = True,
              **kwargs):
    """
    Args:
        model (nn.Module): the trained model.
        x (Tensor): the input of the model.
        label (Tensor): the target of the model.
        eps (float): the robustness radius.
        lc (float or torch.Tensor): The lipschitz of the model backbone.
        return_loss (bool): if True, compute and return the loss.
    """
    if hasattr(model, 'module'):
        head = model.module.head.get_weight()
    else:
        head = model.head.get_weight()
    y = model(x)

    y = y - y.gather(dim=1, index=label.reshape(-1, 1))
    head_j = head[label].unsqueeze(1)  # batch, 1, dim
    head_ji = head_j - head.unsqueeze(0)  # batch, num_class, dim
    head_ji = head_ji.norm(dim=-1)  # batch, num_class
    margin = lc * head_ji

    eps_ji = -y / (margin * eps).clip(1e-8)
    eps_ji = eps_ji.clip(0.01, 1)  # .sqrt()

    y_ = y + eps_ji.detach() * eps * margin
    if return_loss:
        loss = nn.CrossEntropyLoss()(y_, label)
    else:
        loss = None
    return y, y_, loss
