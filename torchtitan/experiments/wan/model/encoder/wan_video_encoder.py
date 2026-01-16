import logging
from typing import Any

from einops import repeat
import torch

from torchtitan.tools.utils import device_module
from torchtitan.config import JobConfig

from .wan_video_text_encoder import WanTextEncoder
from .wan_video_vae import WanVideoVAE38, WanVideoVAE

logger = logging.getLogger(__name__)

# TODO (limou)
PATTERN = "B C H W"

class WanVideoEncoder(torch.nn.Module):
    def __init__(self, job_config : JobConfig):
        super().__init__()
        self.device = device_module.current_device()

        encoder_config = job_config.encoder

        assert encoder_config.vae_type in ("wan_video_vae", "wan_video_vae_38")
        self.vae = WanVideoVAE() if encoder_config.vae_type == "wan_video_vae" else WanVideoVAE38()
        self.text_encoder = WanTextEncoder()

        # The following parameters are used for shape check.
        self.height_division_factor = 16
        self.width_division_factor = 16
        self.time_division_factor = 4
        self.time_division_remainder = 1

        self.load_model(encoder_config.vae_checkpoint_path, encoder_config.t5_checkpoint_path)

    def load_model(self, vae_checkpoint_path : str = None, text_encoder_checkpoint_path : str = None):
        if vae_checkpoint_path:
            logger.info(f"loading VAE from {vae_checkpoint_path}")
            vae_state_dict = torch.load(vae_checkpoint_path, map_location="cpu")
            # Check if we need to add 'model.' prefix
            if "model.encoder.conv1.weight" not in vae_state_dict and "encoder.conv1.weight" in vae_state_dict:
                logger.info("Detected missing 'model.' prefix in VAE checkpoint. Adding it...")
                new_vae_state_dict = {}
                for k, v in vae_state_dict.items():
                    new_vae_state_dict[f"model.{k}"] = v
                vae_state_dict = new_vae_state_dict

            self.vae.load_state_dict(vae_state_dict, strict=True, assign=True)
            logger.info("VAE loaded.")

        if text_encoder_checkpoint_path:
            logger.info(f"Loading T5 from {text_encoder_checkpoint_path}")
            t5_state_dict = torch.load(
                text_encoder_checkpoint_path, map_location="cpu"
            )
            self.text_encoder.load_state_dict(t5_state_dict, strict=True, assign=True)
            logger.info("T5 loaded.")
        pass

    def encode_prompt(self, input_ids, attention_mask, device="cuda"):
        seq_lens = attention_mask.gt(0).sum(dim=1).long()
        prompt_emb = self.text_encoder(input_ids, attention_mask)
        for i, v in enumerate(seq_lens):
            prompt_emb[i, v:] = 0
        return prompt_emb

    def preprocess_video(
        self,
        video,
        dtype=None,
        device=None,
        pattern="B C T H W",
        min_value=-1,
        max_value=1,
    ):
        # Support both list of frames (single video) and batch tensor inputs
        if isinstance(video, torch.Tensor) and video.ndim == 5:
            # Assume input shape is (B, C, T, H, W)
            return video
        # Original behavior for list of PIL images or tensors per frame
        video = [repeat(image, f"H W C -> {PATTERN}", **({"B": 1} if "B" in PATTERN else {})) for image in video]
        video = torch.stack(video, dim=pattern.index("T") // 2)
        return video


    def generate_noise(
        self,
        shape,
        seed=None,
        rand_device="cpu",
        rand_dtype=torch.float32,
        device=None,
        dtype=None,
    ):
        # Initialize Gaussian noise
        generator = None if seed is None else torch.Generator(rand_device).manual_seed(seed)
        noise = torch.randn(shape, generator=generator, device=rand_device, dtype=rand_dtype)
        noise = noise.to(dtype=dtype or self.dtype, device=device or self.device)
        return noise

    def check_resize_height_width(self, height, width, num_frames=None):
        # Shape check
        if height % self.height_division_factor != 0:
            height = (
                (height + self.height_division_factor - 1) // self.height_division_factor * self.height_division_factor
            )
            logger.info(f"height % {self.height_division_factor} != 0. We round it up to {height}.")

        if width % self.width_division_factor != 0:
            width = (width + self.width_division_factor - 1) // self.width_division_factor * self.width_division_factor
            logger.info(f"width % {self.width_division_factor} != 0. We round it up to {width}.")

        if num_frames is not None:
            if num_frames % self.time_division_factor != self.time_division_remainder:
                num_frames = (
                    num_frames + self.time_division_factor - 1
                ) // self.time_division_factor * self.time_division_factor + self.time_division_remainder
                logger.info(
                    f"num_frames % {self.time_division_factor} != {self.time_division_remainder}. We round it up to {num_frames}."
                )

        return height, width, num_frames

    def noise_initialize(self, height, width, num_frames, seed, rand_device, batch_size=1):
        length = (num_frames - 1) // 4 + 1
        shape = (
            batch_size,
            self.vae.model.z_dim,
            length,
            height // self.vae.upsampling_factor,
            width // self.vae.upsampling_factor,
        )
        noise = self.generate_noise(shape, seed=seed, rand_device=rand_device)
        return noise

    def embed_input_video(self, input_video, noise, tiled, tile_size, tile_stride):
        input_video = self.preprocess_video(input_video)  # B, C, T, H, W
        input_latents = self.vae.encode(
            input_video,
            device=self.device,
            tiled=tiled,
            tile_size=tile_size,
            tile_stride=tile_stride,
        ).to(dtype=self.dtype, device=self.device)
        return input_latents



    def forward(self, inputs: dict[str, Any], scheduler):

        def get_model_device_dtype(model):
            try:
                t = next(model.parameters())
            except StopIteration:
                t = next(model.buffers())
            return t.device, t.dtype
    
        # TODO (limou)
        self.deivce, self.dtype = get_model_device_dtype(self)

        video = inputs["video"]

        # TODO (limou)
        # remove ununsed keys-values
        defaults = {
            "input_ids": None,
            "attention_mask": None,
            "cfg_scale": 1,
            "cfg_merge": False,
            "vace_scale": 1,
            "seed": None,
            "vace_reference_image": None,
            "reference_image": None,
            "tiled": False,
            "tile_size": None,
            "tile_stride": None,
            "end_image": None,
            "camera_control_direction": None,
            "camera_control_speed": None,
            "camera_control_origin": None,
            "control_video": None,
            "motion_bucket_id": None,
            "vace_video": None,
            "vace_video_mask": None,
            # TODO (limou)
            # "input_image": video.select(2, 0) if video.ndim == 5 else video[0],
            "input_image" : None,
        }
        for k, v in defaults.items():
            inputs.setdefault(k, v)

        # Recover height/width/num_frames
        if "height" not in inputs:
            if video.ndim == 5:
                # [B, C, F, H, W]
                inputs["num_frames"], inputs["height"], inputs["width"] = (
                    video.shape[2:5]
                )
            else:
                inputs["num_frames"], inputs["height"], inputs["width"] = (
                    video.shape[:3]
                )

        height, width, num_frames = self.check_resize_height_width(
            inputs["height"], inputs["width"], inputs["num_frames"]
        )
        inputs.update({"height": height, "width": width, "num_frames": num_frames})

        if inputs.get("video") is not None and isinstance(inputs["video"], torch.Tensor):
            vid = inputs["video"]
            if vid.ndim == 5:
                # (B, C, F, H, W)
                # Ensure dtype matches model
                if vid.dtype != self.dtype:
                    logger.info(f"Casting video from {vid.dtype} to {self.dtype}")
                    vid = vid.to(self.dtype)
                inputs["video"] = vid

        batch_size = 1
        if inputs.get("video") is not None and isinstance(inputs["video"], torch.Tensor) and inputs["video"].ndim == 5:
             batch_size = inputs["video"].shape[0]
        elif inputs.get("input_ids") is not None:
             batch_size = inputs["input_ids"].shape[0]
            
        noise = self.noise_initialize(
            inputs["height"],
            inputs["width"],
            inputs["num_frames"],
            inputs["seed"],
            "cpu",
            batch_size=batch_size,
        )
        inputs.update({"noise": noise})

        if inputs["video"] is not None:
            input_latents = self.embed_input_video(
                inputs["video"],
                noise,
                inputs["tiled"],
                inputs["tile_size"],
                inputs["tile_stride"],
            )
            if not scheduler.training:
                latents = scheduler.add_noise(input_latents, noise, timestep=scheduler.timesteps[0])
                inputs.update({"latents": latents})
            else:
                inputs.update(
                    {"latents": noise, "input_latents": input_latents}
                )  # this 'latents' actually will not be used in training.
        else:
            inputs.update({"latents": noise})

        # might need to be checked.
        context = self.encode_prompt(inputs["input_ids"], inputs["attention_mask"], device=self.device)
        inputs.update({"context": context})

        # Sample random timestep
        max_timestep_boundary = int(1 * scheduler.num_train_timesteps)
        min_timestep_boundary = int(0 * scheduler.num_train_timesteps)
        # TODO (limou)
        # check whether differnet dp rank should use the same timestep_id when setting seed
        timestep_id = torch.randint(min_timestep_boundary, max_timestep_boundary, (1,))
        timestep = scheduler.timesteps[timestep_id]
        # logger.info("timestep_id={}, timestep={}, len(scheduler.timesteps)={}".format(
        #     timestep_id, timestep, len(scheduler.timesteps),))

        inputs["timestep"] = timestep
        
        # Compute training target
        training_target = scheduler.training_target(
            inputs["input_latents"],
            inputs["noise"],
            timestep,
        )
        inputs["training_target"] = training_target
        
        # Add noise
        inputs["latents"] = scheduler.add_noise(
            inputs["input_latents"],
            inputs["noise"],
            timestep,
        )

        return inputs