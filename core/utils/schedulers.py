import math
from typing import List

from torch.optim.lr_scheduler import _LRScheduler


class PolyLR(_LRScheduler):
    def __init__(self, optimizer, max_iter, decay_iter=1, power=0.9, last_epoch=-1) -> None:
        self.decay_iter = decay_iter
        self.max_iter = max_iter
        self.power = power
        super().__init__(optimizer, last_epoch=last_epoch)

    def get_lr(self):
        if self.last_epoch % self.decay_iter or self.last_epoch % self.max_iter:
            return self.base_lrs
        else:
            factor = (1 - self.last_epoch / float(self.max_iter)) ** self.power
            return [factor * lr for lr in self.base_lrs]


class WarmupLR(_LRScheduler):
    def __init__(self, optimizer, warmup_iter=500, warmup_ratio=5e-4, warmup='exp', last_epoch=-1) -> None:
        self.warmup_iter = warmup_iter
        self.warmup_ratio = warmup_ratio
        self.warmup = warmup
        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        ratio = self.get_lr_ratio()
        return [ratio * lr for lr in self.base_lrs]

    def get_lr_ratio(self):
        return self.get_warmup_ratio() if self.last_epoch < self.warmup_iter else self.get_main_ratio()

    def get_main_ratio(self):
        raise NotImplementedError

    def get_warmup_ratio(self):
        assert self.warmup in ['linear', 'exp']
        alpha = self.last_epoch / self.warmup_iter

        return self.warmup_ratio + (
                1. - self.warmup_ratio) * alpha if self.warmup == 'linear' else self.warmup_ratio ** (1. - alpha)


class WarmupPolyLR(WarmupLR):
    def __init__(self, optimizer, power, max_iter, warmup_iter=500, warmup_ratio=5e-4, warmup='exp',
                 last_epoch=-1) -> None:
        self.power = power
        self.max_iter = max_iter
        super().__init__(optimizer, warmup_iter, warmup_ratio, warmup, last_epoch)

    def get_main_ratio(self):
        real_iter = self.last_epoch - self.warmup_iter
        real_max_iter = self.max_iter - self.warmup_iter
        alpha = real_iter / real_max_iter

        return (1 - alpha) ** self.power


class WarmupExpLR(WarmupLR):
    def __init__(self, optimizer, gamma, interval=1, warmup_iter=500, warmup_ratio=5e-4, warmup='exp',
                 last_epoch=-1) -> None:
        self.gamma = gamma
        self.interval = interval
        super().__init__(optimizer, warmup_iter, warmup_ratio, warmup, last_epoch)

    def get_main_ratio(self):
        real_iter = self.last_epoch - self.warmup_iter
        return self.gamma ** (real_iter // self.interval)


class WarmupCosineLR(WarmupLR):
    def __init__(self, optimizer, max_iter, eta_ratio=0, warmup_iter=500, warmup_ratio=5e-4, warmup='exp',
                 last_epoch=-1) -> None:
        self.eta_ratio = eta_ratio
        self.max_iter = max_iter
        super().__init__(optimizer, warmup_iter, warmup_ratio, warmup, last_epoch)

    def get_main_ratio(self):
        real_iter = self.last_epoch - self.warmup_iter
        real_max_iter = self.max_iter - self.warmup_iter

        return self.eta_ratio + (1 - self.eta_ratio) * (1 + math.cos(math.pi * self.last_epoch / real_max_iter)) / 2


class RepeatedWarmupPolyLR(_LRScheduler):

    def __init__(self, optimizer, warmup_iter=10, warmup_ratio=0.1, warmup='exp', last_epoch=-1, max_iter=100,
                 power=0.9, restart_warmup_epochs: List[int] = None) -> None:
        self.warmup_iter = warmup_iter
        self.warmup_ratio = warmup_ratio
        self.warmup = warmup

        self.max_iter = max_iter
        self.power = power

        self.restart_warmup_epochs = restart_warmup_epochs
        if restart_warmup_epochs is not None:
            assert len(restart_warmup_epochs) > 0, 'restart_warmup_epochs should be a list of epochs'
            self.restart_warmup_epochs.sort()
            assert self.restart_warmup_epochs[
                       0] >= self.warmup_iter, 'restart_warmup_epochs should be greater than warmup_iter'
        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        ratio = self.get_lr_ratio()
        return [ratio * lr for lr in self.base_lrs]

    def get_lr_ratio(self):
        # return self.get_warmup_ratio() if self.last_epoch < self.warmup_iter else self.get_main_ratio()
        if self.last_epoch < self.warmup_iter:
            return self.get_warmup_ratio()
        else:
            if self.restart_warmup_epochs is not None:
                if self.last_epoch in self.restart_warmup_epochs:
                    current_epoch = self.last_epoch
                    self.last_epoch = -1
                    self.restart_warmup_epochs.pop(0)
                    self.restart_warmup_epochs = [epoch - current_epoch for epoch in self.restart_warmup_epochs]
                    self.restart_warmup_epochs = [epoch for epoch in self.restart_warmup_epochs if epoch > 0]

                    # print(
                    #     f'warmup iter is set to: {self.warmup_iter}, restart_warmup_epochs: {self.restart_warmup_epochs}')

                    return self.get_warmup_ratio()
            return self.get_main_ratio()

    def get_main_ratio(self):
        real_iter = self.last_epoch - self.warmup_iter
        real_max_iter = self.max_iter - self.warmup_iter
        alpha = real_iter / real_max_iter
        ratio = (1 - alpha) ** self.power
        return ratio

    def get_warmup_ratio(self):
        assert self.warmup in ['linear', 'exp']
        alpha = self.last_epoch / self.warmup_iter

        ratio = self.warmup_ratio + (
                1. - self.warmup_ratio) * alpha if self.warmup == 'linear' else self.warmup_ratio ** (1. - alpha)
        return ratio
# %%
