import logging

import torch

from torchtitan.config import JobConfig, TORCH_DTYPE_MAP
from torchtitan.train import main, Trainer

from torchtitan.experiments.wan.debug_utils import print_tensor
from torchtitan.experiments.wan.infra.parallelize import parallelize_encoders
from torchtitan.experiments.wan.model.encoder import WanVideoEncoder
from torchtitan.experiments.wan.flow_match_scheduler import FlowMatchScheduler

logger = logging.getLogger(__name__)

class WanTrainer(Trainer):
    def __init__(self, job_config: JobConfig):
        super().__init__(job_config)
        
        # NOTE: self._dtype is the data type used for encoders (image encoder, T5 text encoder, CLIP text encoder).
        # We cast the encoders and it's input/output to this dtype.  If FSDP with mixed precision training is not used,
        # the dtype for encoders is torch.float32 (default dtype for Flux Model).
        # Otherwise, we use the same dtype as mixed precision training process.
        self._dtype = (
            TORCH_DTYPE_MAP[job_config.training.mixed_precision_param]
            if self.parallel_dims.dp_shard_enabled
            else torch.float32
        )

        # TODO : (limou)
        # bfloat16 mixed precision
        self.encoder = WanVideoEncoder(job_config).to(
            device=self.device, dtype=torch.float32).eval().requires_grad_(False)
        self.encoder = parallelize_encoders(
            self.encoder,
            parallel_dims=self.parallel_dims,
            job_config=job_config)
        
        # TODO (limou)
        # remove these checks
        assert not self.encoder.training
        assert not self.encoder.vae.training
        assert not self.encoder.text_encoder.training

        assert not next(self.encoder.vae.parameters()).requires_grad
        assert not next(self.encoder.text_encoder.parameters()).requires_grad
 
        # TODO (limou)
        # flow_match_scheduler stateful load && save
        self.flow_match_scheduler = FlowMatchScheduler()
        # logger.info(f"job_config={job_config}")
        # end __init__


    def forward_backward_step(
        self, input_dict: dict[str, torch.Tensor], labels: torch.Tensor
    ) -> torch.Tensor:
 
        # TODO (limou)
        # remove these checks
        assert not input_dict["video"].requires_grad
        assert not input_dict["input_ids"].requires_grad
        assert not input_dict["attention_mask"].requires_grad
        

        with torch.no_grad():
            model_inputs = self.encoder(input_dict, self.flow_match_scheduler)

        with self.maybe_enable_amp:
            pred = self.model_parts[0](
                x=model_inputs["latents"],
                context=model_inputs["context"],
                timestep=model_inputs["timestep"],
                # TODO (limou)
                # attention_mask ?
            )

            loss = self.loss_fn(pred, model_inputs["training_target"],
                model_inputs["timestep"], self.flow_match_scheduler)

        del pred
        loss.backward()
        return loss

if __name__ == "__main__":
    main(WanTrainer)
