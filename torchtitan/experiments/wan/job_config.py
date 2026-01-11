from dataclasses import dataclass, field

@dataclass
class Encoder:
    t5_checkpoint_path: str = None
    vae_checkpoint_path: str = None
    vae_type: str = "wan_video_vae_38"

@dataclass
class JobConfig:
    encoder: Encoder = field(default_factory=Encoder)