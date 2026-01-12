import torch.nn as nn
import torch.nn.functional as F

from torchtitan.tools.logging import logger

class WanLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, pred, target, timestep, scheduler):
        logger.info("compute wan loss ...")
        # TODO (limou)
        # check dtype
        loss = F.mse_loss(pred.float(), target.float(), reduction="mean")
        return loss * scheduler.training_weight(timestep)

def build_wan_loss(job_config, parallel_dims, ft_manager):
    logger.info("build wan loss ..")
    # TODO (limou)
    # set scheduler arguments in config file
    return WanLoss()
