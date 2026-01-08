import torch.nn as nn
import torch.nn.functional as F

from .scheduler import FlowMatchScheduler

class WanLoss(nn.Module):
    def __init__(self, scheduler):
        super().__init__()
        self.scheduler = scheduler

    def forward(self, pred, target, timestep):
        # TODO (limou)
        # check dtype
        loss = F.mse_loss(pred.float(), target.float(), reduction="mean")
        return loss * self.scheduler.training_weight(timestep)

def build_wan_loss(job_config, parallel_dims, ft_manager):
    # TODO (limou)
    # set scheduler arguments in config file
    scheduler = FlowMatchScheduler(shift=5.0, sigma_min=0.0, extra_one_step=True)
    scheduler.set_timesteps(1000, training=True)
    return WanLoss(scheduler)
