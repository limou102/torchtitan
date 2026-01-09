from dataclasses import dataclass, field
from typing import List

from torchtitan.protocols import BaseModelArgs
from torchtitan.tools.logging import logger

@dataclass
class WanModelArgs(BaseModelArgs):
    # DiT model parameters
    dit_hidden_size: int = 3072
    dit_num_layers: int = 30
    dit_num_heads: int = 24
    dit_intermediate_size: int = 14336
    dit_patch_size: List[int] = field(default_factory=lambda: [1, 2, 2])
    dit_in_channels: int = 48
    dit_out_channels: int = 48

    dit_freq_dim: int = 256
    dit_text_dim: int = 4096
    dit_eps: float = 1.0e-6

    def update_from_config(self, job_config, **kwargs) -> None:
        # TODO (limou)
        logger.info("update WanModelArgs from config.")
        pass

    def get_nparams_and_flops(self, model, seq_len: int) -> tuple[int, float]:
        # Calculate params
        nparams = sum(p.numel() for p in model.parameters())

        # TODO (zirui)
        # Approximate FLOPS (dummy for now)
        flops_per_token = 1.0

        # TODO (zirui)
        # Calculate memory for debug
        mem_parameters = sum(p.numel() * p.element_size() for p in model.parameters())
        mem_buffers = sum(b.numel() * b.element_size() for b in model.buffers())
        logger.info("Wan DIT model estimated memory {} MB".format((
            mem_parameters + mem_buffers) / 1024**2))

        return nparams, flops_per_token
