from dataclasses import dataclass, field

@dataclass
class Training:
    dataset_folder: str = None

@dataclass
class Encoder:
    t5_checkpoint_path: str = None
    vae_checkpoint_path: str = None

    # (wan_video_vae, wan_video_vae_38)
    vae_type: str = "wan_video_vae"

@dataclass
class JobConfig:
    training: Training = field(default_factory=Training)
    encoder: Encoder = field(default_factory=Encoder)