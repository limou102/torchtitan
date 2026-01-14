
import logging
from typing import Any, Callable, Optional
from dataclasses import asdict

from datasets import load_dataset, Dataset
from datasets.distributed import split_dataset_by_node
from torch.distributed.checkpoint.stateful import Stateful
from torch.utils.data import IterableDataset

import torch
from torchtitan.config import JobConfig
from torchtitan.components.tokenizer import BaseTokenizer
from torchtitan.hf_datasets import DatasetConfig
from torchtitan.components.dataloader import ParallelAwareDataloader

from .data_processor import VIDGEN1MDataProcessor

logger = logging.getLogger(__name__)

DATASETS = {
    # TODO (limou)
    # use remote dataset
    "vidgen-1m": DatasetConfig(
        path = "/data/limou/VIDGEN-1M/meta_data.json",
        loader = lambda path: load_dataset("json", data_files=path, split="train"),
        sample_processor = VIDGEN1MDataProcessor(),
    ),
}

def _validate_dataset(dataset_name : str, dataset_path: Optional[str] = None):
    if dataset_name not in DATASETS:
        raise ValueError(
            f"Dataset {dataset_name} is not supported. "
            f"Supported datasets are: {list(DATASETS.keys())}"
        )
    config = DATASETS[dataset_name]
    path = dataset_path or config.path
    logger.info(f"Preparing {dataset_name} dataset from {path}")
    return path, config.loader, config.sample_processor

class WanDataset(IterableDataset, Stateful):
    def __init__(
        self,
        dataset_name: str,
        dataset_path: Optional[str],
        job_config: Optional[JobConfig] = None,
        dp_rank: int = 0,
        dp_world_size: int = 1,
    ) -> None:
        data_path, data_loader, data_processor = _validate_dataset(dataset_name.lower(), dataset_path)

        self.data_processor = data_processor
        ds = data_loader(data_path)
        self.data = split_dataset_by_node(ds, dp_rank, dp_world_size)
        self.sample_idx = 0

    def _get_data_iter(self):
        if isinstance(self.data, Dataset):
            if self.sample_idx == len(self.data):
                return iter([])
            else:
                return iter(self.data.skip(self.sample_idx))

        return iter(self.data)

    def __iter__(self):
        dataset_iterator = self._get_data_iter()
        while True:
            try:
                sample = next(dataset_iterator)
            except StopIteration:
                logger.info("run out of data")
                break
            
            # logger.info(f"dataset iter, sample={sample}")
            inputs = self.data_processor(sample)
            self.sample_idx += 1
            yield inputs

    def load_state_dict(self, state_dict):
        if isinstance(self.data, Dataset):
            self.sample_idx = state_dict["sample_idx"]
        else:
            assert "data" in state_dict
            self.data.load_state_dict(state_dict["data"])

    def state_dict(self):
        if isinstance(self.data, Dataset):
            return {"sample_idx": self.sample_idx}
        else:
            return {"data": self.data.state_dict()}


class WanCollator:
    def __init__(self):
        pass

    def __call__(self, batch):
        logger.info("wan collator ...")
        input_dict = {"input": batch}
        # Create dummy labels because diffusion target is computed in forward
        labels = torch.zeros(1)
        return input_dict, labels

def build_wan_dataloader(
    dp_world_size: int,
    dp_rank: int,
    job_config: JobConfig,
    tokenizer: Optional[BaseTokenizer] = None,
    ) -> ParallelAwareDataloader:

    dataset_name = job_config.training.dataset
    dataset_path = job_config.training.dataset_path

    ds = WanDataset(
        dataset_name=dataset_name,
        dataset_path=dataset_path,
        job_config=job_config,
        dp_rank=dp_rank,
        dp_world_size=dp_world_size,
    )

    dataloader_kwargs = {
        **asdict(job_config.training.dataloader),
        "batch_size": job_config.training.local_batch_size,
        "collate_fn" : WanCollator(),
    }


    return ParallelAwareDataloader(
        dataset=ds,
        dp_rank=dp_rank,
        dp_world_size=dp_world_size,
        **dataloader_kwargs,
    )