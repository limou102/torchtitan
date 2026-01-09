import torch

from torchtitan.config import JobConfig, TORCH_DTYPE_MAP
from torchtitan.distributed import utils as dist_utils
from torchtitan.train import main, Trainer
from torchtitan.tools.logging import logger

class WanTrainer(Trainer):
    def __init__(self, job_config: JobConfig):
        super().__init__(job_config)
        
        # TODO (limou)
        # check determinism mode
        dist_utils.set_determinism(
            self.parallel_dims,
            self.device,
            job_config.debug,
            distinct_seed_mesh_dims=["fsdp", "dp_replicate"],
        )

        # NOTE: self._dtype is the data type used for encoders (image encoder, T5 text encoder, CLIP text encoder).
        # We cast the encoders and it's input/output to this dtype.  If FSDP with mixed precision training is not used,
        # the dtype for encoders is torch.float32 (default dtype for Flux Model).
        # Otherwise, we use the same dtype as mixed precision training process.
        self._dtype = (
            TORCH_DTYPE_MAP[job_config.training.mixed_precision_param]
            if self.parallel_dims.dp_shard_enabled
            else torch.float32
        )

        model_args = self.train_spec.model_args[job_config.model.flavor]

    def forward_backward_step(
        self, input_dict: dict[str, torch.Tensor], labels: torch.Tensor
    ) -> torch.Tensor:
        logger.info(f"wan forward_backward_step, input_dict={input_dict}")
        return torch.zeros((1,))


if __name__ == "__main__":
    main(WanTrainer)
