import logging

import torch

from torchtitan.config import JobConfig, TORCH_DTYPE_MAP
from torchtitan.train import main, Trainer
from torchtitan.experiments.wan.debug_utils import print_tensor

from .infra.parallelize import parallelize_encoders
from .model.encoder import WanVideoEncoder
from .scheduler import FlowMatchScheduler

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

        model_args = self.train_spec.model_args[job_config.model.flavor]

        # TODO : (limou)
        # bfloat16 mixed precision
        self.encoder = WanVideoEncoder(model_args).to(
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
        
        pass

    def forward_backward_step(
        self, input_dict: dict[str, torch.Tensor], labels: torch.Tensor
    ) -> torch.Tensor:
        
        # TODO (limou)
        # < -- load input_dict from local file, dataset hasn't been implemented now
        if not hasattr(self, "inputs_from_local"):
            self.inputs_from_local = torch.load("/data/limou/common_modules/fake_inputs_wan.pt")
            def move_to_device(x):
                if torch.is_tensor(x):
                    return x.to(self.device)
                if isinstance(x, dict):
                    return {k: move_to_device(v) for k, v in x.items()}
                return x
            self.inputs_from_local = move_to_device(self.inputs_from_local)
            
            self.step_idx = 0
        
        self.step_idx += 1
        if self.step_idx > 3:
            import sys
            logger.warning("force exit")
            sys.exit()
        logger.info(f"step_idx = {self.step_idx}")

        input_dict = self.inputs_from_local[self.step_idx]
        input_dict["video"] = input_dict.pop("input")

        print_tensor(input_dict["video"], "video")
        # -->

        # TODO (limou)
        assert not input_dict["video"].requires_grad
        assert not input_dict["input_ids"].requires_grad
        assert not input_dict["attention_mask"].requires_grad
        
        # logger.info(f"wan forward_backward_step, input_dict={input_dict}")
        with torch.no_grad():
            model_inputs = self.encoder(input_dict, self.flow_match_scheduler)
        # logger.info(f"after encoder, model_inputs={model_inputs}")

        if self.step_idx == 1:
            print_tensor(model_inputs["latents"], "latents")
            print_tensor(model_inputs["context"], "context")
            print_tensor(model_inputs["timestep"], "timestep")
        
        with self.maybe_enable_amp:
            pred = self.model_parts[0](
                x=model_inputs["latents"],
                context=model_inputs["context"],
                timestep=model_inputs["timestep"],
                # TODO (limou)
                # attention_mask ?
            )
            if self.step_idx == 1:
                print_tensor(pred, "pred")

            loss = self.loss_fn(pred, model_inputs["training_target"],
                model_inputs["timestep"], self.flow_match_scheduler)

        del pred
        loss.backward()
        return loss

if __name__ == "__main__":
    main(WanTrainer)
