# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import torch
import torch.nn as nn
from torch.distributed.device_mesh import DeviceMesh
from torch.distributed.fsdp import CPUOffloadPolicy, fully_shard, MixedPrecisionPolicy

from torchtitan.config import JobConfig, TORCH_DTYPE_MAP
from torchtitan.distributed import ParallelDims
from torchtitan.tools.logging import logger

def parallelize_wan(
    model: nn.Module,
    parallel_dims: ParallelDims,
    job_config: JobConfig,
):
    """
    Apply parallelism to the Wan model using FSDP2 (fully_shard).
    Mimics Flux parallelization strategy.
    
    Args:
        model: WanVideoModel wrapper (contains .model which is WanVideoForConditionalGeneration)
        parallel_dims: ParallelDims object
        job_config: JobConfig object
    """
    
    logger.info("parallelize_wan ...")
    if job_config.activation_checkpoint.mode != "none":
        apply_ac(model, job_config.activation_checkpoint)

    if parallel_dims.fsdp_enabled:
        names = (
            ["dp_replicate", "fsdp"] if parallel_dims.dp_replicate_enabled else ["fsdp"]
        )

        dp_mesh = parallel_dims.get_mesh(names)
        apply_fsdp(
            model,
            dp_mesh,
            param_dtype=TORCH_DTYPE_MAP[job_config.training.mixed_precision_param],
            reduce_dtype=TORCH_DTYPE_MAP[job_config.training.mixed_precision_reduce],
            cpu_offload=job_config.training.enable_cpu_offload,
        )

        if parallel_dims.dp_replicate_enabled:
            logger.info("Applied HSDP to the model")
        else:
            logger.info("Applied FSDP to the model")

        # TODO (limou)
        # Apply Context Parallelism

    return model


def apply_fsdp(
    model: nn.Module,
    dp_mesh: DeviceMesh,
    param_dtype: torch.dtype,
    reduce_dtype: torch.dtype,
    cpu_offload: bool = False,
):
    mp_policy = MixedPrecisionPolicy(param_dtype=param_dtype, reduce_dtype=reduce_dtype)
    fsdp_config = {"mesh": dp_mesh, "mp_policy": mp_policy}
    if cpu_offload:
        fsdp_config["offload_policy"] = CPUOffloadPolicy()

    for block in model.blocks:
        fully_shard(block, **fsdp_config)

    # TODO (limou)
    # whether to shard text_embed, time_embded, head ?

    fully_shard(model, **fsdp_config)



def apply_ac(model: nn.Module, ac_config):
    """Apply activation checkpointing to the model."""
    from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import (
        checkpoint_wrapper as ptd_checkpoint_wrapper,
        CheckpointImpl,
        offload_wrapper as ptd_offload_wrapper,
    )

    for layer_id, block in model.blocks.named_children():
        # TODO: Check config for mode (full vs selective vs offload?)
        # Here keeping it simple like Flux: simple wrapper.
        # If offload is needed, use offload_wrapper.
        
        # Use ptd_checkpoint_wrapper
        block = ptd_checkpoint_wrapper(block, preserve_rng_state=False)
        model.blocks.register_module(layer_id, block)

    logger.info(f"Applied {ac_config.mode} activation checkpointing to the Wan model")
