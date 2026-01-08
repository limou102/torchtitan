
from typing import Any, Callable, Optional

from torchtitan.config import JobConfig
from torchtitan.components.tokenizer import BaseTokenizer

DATASETS = {
    "example_video": get_example_video_dataset(),
    "vidgen-1m": get_vidgen_1m_dataset(),
}

def _get_data_processor(dataset_name : str):
    pass

class WanDataset(IterableDataset, Stateful):
    def __init__(
        self,
        dataset_name: str,
        dataset_path: Optional[str],
        job_config: Optional[JobConfig] = None,
        dp_rank: int = 0,
        dp_world_size: int = 1,
    ) -> None:
        self.data_processor = _get_data_processor(dataset_name.lower())

        pass