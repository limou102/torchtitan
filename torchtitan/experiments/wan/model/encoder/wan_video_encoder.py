import torch

from torchtitan.tools.logging import logger

from .wan_video_text_encoder import WanTextEncoder
from .wan_video_vae import WanVideoVAE38, WanVideoVAE
from ..args import WanModelArgs

class WanVideoEncoder(torch.nn.Module):
    def __init__(self, model_args : WanModelArgs):
        logger.info("WanPreProcessor init ...")
        logger.info(f"model_args={model_args}")
        super().__init__()
        self.vae = WanVideoVAE() if model_args.vae_type == "wan_video_vae" else WanVideoVAE38()
        self.text_encoder = WanTextEncoder()

        pass
