import math


class lr_scheduler(object):
    def __init__(self, iter_per_epoch, max_epoch, warmup_epoch=5):

        self.warmup_iters = iter_per_epoch * warmup_epoch
        self.current_iter = 0
        self.num_iters = iter_per_epoch * (max_epoch - warmup_epoch)
        self.base_lr = 0.0

    def step(self, optimizer):
        if self.current_iter == 0:
            for group in optimizer.param_groups:
                lr = group['lr']
                group.setdefault('initial_lr', group['lr'])
                self.base_lr = max(self.base_lr, lr)

        self.current_iter += 1
        if self.current_iter < self.warmup_iters:
            lr_ratio = self.current_iter / self.warmup_iters
        else:
            lr_ratio = self._cosine_get_lr()

        for param_group in optimizer.param_groups:
            param_group['lr'] = lr_ratio * param_group['initial_lr']
        return lr_ratio * self.base_lr

    def _cosine_get_lr(self):
        process = (self.current_iter - self.warmup_iters) / self.num_iters
        lr_ratio = .5 * (1 + math.cos(process * math.pi))
        return max(lr_ratio, 1e-5)
