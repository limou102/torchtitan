from torchtitan.components.lr_scheduler import build_lr_schedulers
from torchtitan.components.optimizer import build_optimizers
from torchtitan.protocols.train_spec import TrainSpec
from .wan_dataset import build_wan_dataloader
from .wan_loss import build_wan_loss
from .infra.parallelize import parallelize_wan
from .model.args import WanModelArgs
from .model.model import WanDitModel

wan_args = {
    "default": WanModelArgs(),
    "wan2.1_t2v_debug": WanModelArgs(
        dit_hidden_size=1536,
        dit_num_layers=10,
        dit_num_heads=12,
        dit_intermediate_size=8960,
        dit_in_channels=16,
        dit_out_channels=16,
    ),
    "wan2.1_t2v_1.3b": WanModelArgs(
        dit_hidden_size=1536,
        dit_num_layers=30,
        dit_num_heads=12,
        dit_intermediate_size=8960,
        dit_in_channels=16,
        dit_out_channels=16,
    ),
    "wan2.1_t2v_14b": WanModelArgs(
        dit_hidden_size=5120,
        dit_num_layers=48,
        dit_num_heads=40,
        dit_intermediate_size=13824,
        dit_in_channels=16,
        dit_out_channels=16,
    ),
}

def get_train_spec() -> TrainSpec:
    return TrainSpec(
        model_cls=WanDitModel,
        model_args=wan_args,
        parallelize_fn=parallelize_wan,
        pipelining_fn=None,  # Pipeline parallel not implemented yet
        build_optimizers_fn=build_optimizers,
        # TODO (limou)
        # build scheduler from torchtitan base Trainer
        build_lr_schedulers_fn=build_lr_schedulers,
        build_dataloader_fn=build_wan_dataloader,
        build_tokenizer_fn=None,
        build_loss_fn=build_wan_loss,
        state_dict_adapter=None,  # Implement adapter for checkpointing later
    )
