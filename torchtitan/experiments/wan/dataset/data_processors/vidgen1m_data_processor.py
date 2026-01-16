
import os
import logging
import collections
from typing import Any, Dict, List, Optional, Union, Sequence
from pathlib import Path

import numpy as np

import torch

from PIL import Image

# TODO (limou)
# check if transformers library is needed
from transformers import AutoTokenizer
from transformers.image_utils import ImageInput
from transformers.image_processing_utils import BaseImageProcessor
from transformers.utils import TensorType

from datasets import load_dataset

from torchtitan.config import JobConfig
from torchtitan.hf_datasets import DatasetConfig

logger = logging.getLogger(__name__)

def get_dataset_config_vidgen1m(job_config : JobConfig) -> DatasetConfig:
    # TODO (limou)
    # for test, this is local dataset
    # change to remote dataset with streaming=True
    return DatasetConfig(
            path = "Fudan-FUXI/VIDGEN-1M",
            loader = lambda data_path: load_dataset(Path(data_path).suffix.lstrip("."),
                data_files=data_path, split="train"),

            # TODO (limou)
            # use job_config as arguments to pass more data processor parameters
            sample_processor = VIDGEN1MDataProcessor(
                data_folder=job_config.training.dataset_folder or \
                    os.path.dirname(job_config.training.dataset_path),
                text_tokenizer_id = "google/umt5-xxl",
                num_frames=81),
        )

class VisionCollator:

    def __init__(self, tokenizer):
        self.tokenizer = tokenizer

    def _pad_sequence(self, input_ids, batch_first, padding_value):
        if self.tokenizer.padding_side == "left":
            input_ids = [torch.flip(_input_ids, [0]) for _input_ids in input_ids]
        input_ids = torch.nn.utils.rnn.pad_sequence(input_ids, batch_first=batch_first, padding_value=padding_value)
        if self.tokenizer.padding_side == "left":
            input_ids = torch.flip(input_ids, [1])
        return input_ids

    def __call__(self, instances: Sequence[Dict]) -> Dict[str, torch.Tensor]:
        if isinstance(instances[0], list):
            instances = [inst for instance in instances for inst in instance]
            
        inputs = collections.defaultdict(list)
        for instance in instances:
            for key, values in instance.items():
                inputs[key].append(values)

        batched_inputs = {}
        if "input_ids" in inputs.keys():
            input_ids = inputs.pop("input_ids")
            input_ids = self._pad_sequence(
                input_ids,
                batch_first=True,
                padding_value=self.tokenizer.pad_token_id,
            )
            batched_inputs["input_ids"] = input_ids
        
        # TODO (limou)
        # remove this
        if "labels" in inputs.keys():
            labels = inputs.pop("labels")
            labels = self._pad_sequence(
                labels,
                batch_first=True,
                padding_value=-100,
            )
            batched_inputs["labels"] = labels

        if "attention_mask" in inputs.keys():
            inputs.pop("attention_mask")

        attention_mask = input_ids.ne(self.tokenizer.pad_token_id).long()
        batched_inputs["attention_mask"] = attention_mask

        # for the other keys
        for key, values in inputs.items():
            # Handle scalar/boolean values ( use_audio_in_video)
            if isinstance(values[0], bool) or (
                isinstance(values[0], (int, float)) and not isinstance(values[0], torch.Tensor)
            ):
                batched_inputs[key] = values[0]
            else:
                batched_inputs[key] = torch.stack(values, dim=0)

        return batched_inputs
    
class VideoProcessor(BaseImageProcessor):
    """
    Image/Video processor for WanVideo models.

    Args:
        do_resize: Whether to resize the image/video frames.
        size: Target size for resizing.
        do_center_crop: Whether to center crop.
        crop_size: Size for center cropping.
        do_normalize: Whether to normalize pixel values.
        image_mean: Mean values for normalization.
        image_std: Standard deviation values for normalization.
        do_convert_rgb: Whether to convert to RGB.
    """

    model_input_names = ["pixel_values"]

    def __init__(
        self,
        do_resize: bool = True,
        size: Dict[str, int] = None,
        do_center_crop: bool = True,
        crop_size: Dict[str, int] = None,
        do_normalize: bool = True,
        image_mean: Optional[Union[float, List[float]]] = None,
        image_std: Optional[Union[float, List[float]]] = None,
        do_convert_rgb: bool = True,
        **kwargs,
    ):
        super().__init__(**kwargs)

        self.do_resize = do_resize
        self.size = size or {"height": 480, "width": 832}
        self.do_center_crop = do_center_crop
        self.crop_size = crop_size or {"height": 480, "width": 832}
        self.do_normalize = do_normalize
        self.image_mean = image_mean or [0.5, 0.5, 0.5]
        self.image_std = image_std or [0.5, 0.5, 0.5]
        self.do_convert_rgb = do_convert_rgb

    def resize(
        self,
        image: np.ndarray,
        size: Dict[str, int],
        **kwargs,
    ) -> np.ndarray:
        """Resize image or video frame."""
        from PIL import Image as PILImage

        image = PILImage.fromarray(image.astype(np.uint8))
        # TODO: Here, we align with DiffSynth's resize logic for debugging(may remove in the future)
        if os.getenv("ALIGN_WITH_DIFFSYNTH") == "1":
            # Match DiffSynth logic: scale based on max dimension ratio
            width, height = image.size
            target_width = size["width"]
            target_height = size["height"]
            
            scale = max(target_width / width, target_height / height)
            new_width = round(width * scale)
            new_height = round(height * scale)
            
            from torchvision.transforms import functional as F
            from torchvision.transforms import InterpolationMode
            
            # DiffSynth uses torchvision.transforms.resize with BILINEAR
            # We must use F.resize to match exactly (antialias behavior etc)
            image = F.resize(image, (new_height, new_width), interpolation=InterpolationMode.BILINEAR)
        else:
            image = image.resize((size["width"], size["height"]), PILImage.LANCZOS)
        return np.array(image)

    def center_crop(
        self,
        image: np.ndarray,
        crop_size: Dict[str, int],
        **kwargs,
    ) -> np.ndarray:
        """Center crop image or video frame."""
        h, w = image.shape[:2]
        crop_h, crop_w = crop_size["height"], crop_size["width"]

        top = (h - crop_h) // 2
        left = (w - crop_w) // 2
        if image.ndim == 3:
            return image[top : top + crop_h, left : left + crop_w, :]
        else:
            return image[top : top + crop_h, left : left + crop_w]

    def normalize(
        self,
        image: np.ndarray,
        mean: List[float],
        std: List[float],
        **kwargs,
    ) -> np.ndarray:
        """Normalize image or video frame."""
        image = image.astype(np.float32) / 255.0
        mean = np.array(mean).reshape(1, 1, -1)
        std = np.array(std).reshape(1, 1, -1)
        return (image - mean) / std

    def preprocess(
        self,
        images: ImageInput,
        do_resize: Optional[bool] = None,
        size: Optional[Dict[str, int]] = None,
        do_center_crop: Optional[bool] = None,
        crop_size: Optional[Dict[str, int]] = None,
        do_normalize: Optional[bool] = None,
        image_mean: Optional[Union[float, List[float]]] = None,
        image_std: Optional[Union[float, List[float]]] = None,
        do_convert_rgb: Optional[bool] = None,
        num_frames: Optional[int] = None,
        return_tensors: Optional[Union[str, TensorType]] = None,
        **kwargs,
    ) -> Dict[str, Any]:
        """
        Preprocess images or video frames.

        Args:
            images: Input images or video frames.
            return_tensors: Type of tensors to return ("pt" for PyTorch).

        Returns:
            Dictionary with preprocessed pixel values.
        """
        do_resize = do_resize if do_resize is not None else self.do_resize
        size = size if size is not None else self.size
        do_center_crop = do_center_crop if do_center_crop is not None else self.do_center_crop
        crop_size = crop_size if crop_size is not None else self.crop_size
        do_normalize = do_normalize if do_normalize is not None else self.do_normalize
        image_mean = image_mean if image_mean is not None else self.image_mean
        image_std = image_std if image_std is not None else self.image_std
        do_convert_rgb = do_convert_rgb if do_convert_rgb is not None else self.do_convert_rgb

        # Handle single image or list of images (video frames)
        if not isinstance(images, list):
            images = [images]

        processed_images = []
        for image in images:
            if isinstance(image, Image.Image):
                image = np.array(image)

            # Handle 4D tensor case (batch of video frames)
            if isinstance(image, np.ndarray) and image.ndim == 4:
                # Process each frame in the batch
                batch_processed_frames = []
                for i in range(image.shape[0]):
                    frame = image[i]  # Get single frame
                    if frame.ndim == 3 and (frame.shape[0] == 3 or frame.shape[0] == 1):
                        frame = np.transpose(frame, (1, 2, 0))  # (C, H, W) -> (H, W, C)

                    # Convert to RGB if needed
                    if do_convert_rgb and frame.shape[-1] != 3:
                        if len(frame.shape) == 2:  # Grayscale
                            frame = np.stack([frame] * 3, axis=-1)
                        elif frame.shape[-1] == 4:  # RGBA
                            frame = frame[..., :3]

                    # Resize
                    if do_resize:
                        frame = self.resize(frame, size)

                    # Center crop
                    if do_center_crop:
                        frame = self.center_crop(frame, crop_size)

                    # Normalize
                    if do_normalize:
                        frame = self.normalize(frame, image_mean, image_std)

                    batch_processed_frames.append(frame)

                # Stack frames back into 4D tensor
                processed_image = np.stack(batch_processed_frames, axis=0)
            else:
                # Handle single image case (3D or 2D)
                # Convert to RGB if needed
                if do_convert_rgb and image.shape[-1] != 3:
                    if len(image.shape) == 2:  # Grayscale
                        image = np.stack([image] * 3, axis=-1)
                    elif image.shape[-1] == 4:  # RGBA
                        image = image[..., :3]

                # Resize
                if do_resize:
                    image = self.resize(image, size)

                # Center crop
                if do_center_crop:
                    image = self.center_crop(image, crop_size)

                # Normalize
                if do_normalize:
                    image = self.normalize(image, image_mean, image_std)

                processed_image = image

            processed_images.append(processed_image)

        # Stack frames for video
        processed_images = np.stack(processed_images, axis=0)  # B, T, H, W, C

        # Temporal Handling (Interpolate or Truncate)
        if num_frames is not None:
             current_frames = processed_images.shape[1]
             if current_frames > num_frames:
                 logger.info(f"Truncating video frames from {current_frames} to {num_frames}")
                 processed_images = processed_images[:, :num_frames, ...]
             elif current_frames < num_frames:
                 logger.info(f"Interpolating video frames from {current_frames} to {num_frames}")
                 # Interpolate requires (B, C, T, H, W) or (B, C, H, W) - we have (B, T, H, W, C)
                 # Permute to (B, C, T, H, W) for interpolate
                 vid_tensor = torch.from_numpy(processed_images).permute(0, 4, 1, 2, 3)
                 
                 # Interpolate
                 vid_tensor = torch.nn.functional.interpolate(
                     vid_tensor, 
                     size=(num_frames, vid_tensor.shape[3], vid_tensor.shape[4]), 
                     mode='trilinear', 
                     align_corners=False
                 )
                 
                 # Permute back to (B, T, H, W, C) and convert to numpy
                 processed_images = vid_tensor.permute(0, 2, 3, 4, 1).numpy()

        # Convert to tensor if requested
        if return_tensors == "pt":
            processed_images = torch.from_numpy(processed_images)
            # Rearrange to (B, C, T, H, W) for video (since input was B, T, H, W, C)
            if processed_images.ndim == 5:
                processed_images = processed_images.permute(0, 4, 1, 2, 3)
            elif processed_images.ndim == 4:
                # (T, H, W, C) -> (C, T, H, W) if it was list of frames
                processed_images = processed_images.permute(3, 0, 1, 2)
            # Add batch dimension
            # processed_images = processed_images.unsqueeze(0)

        return {"pixel_values": processed_images}
    
class VIDGEN1MDataProcessor:
    def __init__(self,
        data_folder : str,
        text_tokenizer_id : str,
        num_frames : int):
        self.data_folder = data_folder
        self.num_frames = num_frames
        self.text_tokenizer = AutoTokenizer.from_pretrained(text_tokenizer_id)
        self.video_processor = VideoProcessor()
        logger.info(f"init VIDGEN1MDataProcessor, data_folder={data_folder}")

    def __call__(self, inputs : dict[str, Any]) -> Dict[str, Any]:
        """
        sample of inputs :
        {
            'prompt': 'The video shows a man fishing on a boat...',
            'video': '4c4gus833J4-Scene-0107.mp4',
        }
        """
        prompt = inputs.get("prompt", "")
        video_path = os.path.join(self.data_folder, inputs["video"])
        video_frames = self._load_video(video_path)

        text_inputs = self.text_tokenizer(
            prompt,
            return_tensors="pt",
            padding="max_length",
            truncation=True,
            max_length=512,
        )
        video_inputs = self.video_processor.preprocess(
            video_frames,
            num_frames = self.num_frames,
            return_tensors = "pt",
        )
        pixel_values = video_inputs["pixel_values"]

        outputs = {
            "video": pixel_values.squeeze(0),  # C, T, H, W
            "input_ids": text_inputs["input_ids"].squeeze(0),
            "attention_mask": text_inputs["attention_mask"].squeeze(0),
            "num_frames": pixel_values.shape[2],  # (1, C, T, H, W) -> T used shape[2]
        }
        return outputs

    def _load_video(self, video_path : str):
        # TODO (limou)
        # multiple video decode libraries
        # return self._load_video_decord(video_path)
        return self._load_video_imageio(video_path)

    def _load_video_decord(self, video_path : str):
        from decord import VideoReader, cpu

        vr = VideoReader(video_path, ctx=cpu(0), num_threads=1)

        total_frames = len(vr)
        # Enforce VAE divisibility: (n - 1) % 4 == 0
        actual_nframes = min(self.num_frames, total_frames)
        
        valid_nframes = ((actual_nframes - 1) // 4) * 4 + 1 if actual_nframes > 1 else 1
            
        uniform_sampled_frames = np.arange(valid_nframes, dtype=int)
            
        frame_idx = uniform_sampled_frames.tolist()
        spare_frames = vr.get_batch(frame_idx).asnumpy()
        
        return spare_frames

    def _load_video_imageio(self, video_path : str):
        import imageio

        reader = imageio.get_reader(video_path)
        total_frames = int(reader.count_frames())

        actual_nframes = min(self.num_frames, total_frames)
        valid_nframes = ((actual_nframes - 1) // 4) * 4 + 1 if actual_nframes > 1 else 1

        frames = []
        for i, frame in enumerate(reader):
            if i >= valid_nframes:
                break
            frames.append(frame)
        return np.array(frames)

    def get_collator(self):
        return VisionCollator(self.text_tokenizer)