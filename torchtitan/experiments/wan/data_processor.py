

import logging
from typing import Any, Dict, List, Optional, Union

import numpy as np

import torch

from transformers import AutoTokenizer
from transformers.image_utils import ImageInput
from transformers.image_processing_utils import BaseImageProcessor
from transformers.utils import TensorType

logger = logging.getLogger(__name__)

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

        # Convert to tensor if requested
        if return_tensors == "pt":
            processed_images = torch.from_numpy(processed_images)
            # Rearrange to (C, T, H, W) for video
            # if len(processed_images.shape) == 4:  # (T, H, W, C)
            #     processed_images = processed_images.permute(3, 0, 1, 2)
            # Add batch dimension
            # processed_images = processed_images.unsqueeze(0)

        return {"pixel_values": processed_images}
    
class VIDGEN1MDataProcessor:
    def __init__(self):
        self.text_tokenizer = AutoTokenizer.from_pretrained("google/umt5-xxl")
        self.video_processor = VideoProcessor()

    def __call__(self, inputs : dict[str, Any]) -> Dict[str, Any]:
        # logger.info(f"VIDGEN1MDataProcessor.__call__, inputs={inputs}")
        return torch.zeros((3,3))