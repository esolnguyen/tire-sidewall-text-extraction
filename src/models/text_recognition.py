import logging

import torch
import cv2
import numpy as np
from PIL import Image
from typing import List
from torchvision import transforms as T

from models.trba import TRBA, Tokenizer

logger = logging.getLogger(__name__)


class TextRecognitionModel:
    def __init__(self, model_path: str, device: str = "cpu",
                 img_h: int = 32, img_w: int = 128,
                 charset: str = None, batch_max_length: int = 25):
        """Initialize text recognition model.

        Args:
            model_path: Path to the model checkpoint (.pth file)
            device: Device to run inference on ("cpu" or "cuda")
            img_h: Height of input images
            img_w: Width of input images
            charset: Character set for recognition. If None, uses default alphanumeric + symbols
            batch_max_length: Maximum length of text sequence (must match training parameter)
        """
        self.device = device
        self.img_h = img_h
        self.img_w = img_w
        self.batch_max_length = batch_max_length

        self.charset = charset
        self.num_class = len(charset) + 2  # +2 for [GO] and [s] tokens

        # Initialize tokenizer
        self.tokenizer = Tokenizer(charset)

        # Load model
        self.model = self._load_model(model_path)

        # Image preprocessing transforms for grayscale input
        self.transform = T.Compose([
            T.Grayscale(num_output_channels=1),  # Convert to grayscale
            T.Resize((img_h, img_w), T.InterpolationMode.BICUBIC),
            T.ToTensor(),
            T.Normalize(mean=[0.5], std=[0.5])  # Single channel normalization
        ])

    def _load_model(self, model_path: str):
        """Load the TRBA model from checkpoint."""
        # Create model with grayscale input (matching training configuration)
        # The model was trained without --rgb flag, so input_channel=1
        model = TRBA(
            img_h=self.img_h,
            img_w=self.img_w,
            num_fiducial=20,
            input_channel=1,  # Grayscale input as per training config
            output_channel=512,
            hidden_size=256,
            num_class=self.num_class,
            batch_max_length=self.batch_max_length
        )

        logger.info(f"Loading text recognition model from {model_path}")
        checkpoint = torch.load(model_path, map_location=self.device)

        # Handle different checkpoint formats
        if isinstance(checkpoint, dict):
            if 'state_dict' in checkpoint:
                state_dict = checkpoint['state_dict']
                logger.debug(f"Found PyTorch Lightning checkpoint with keys: {list(checkpoint.keys())}")
                state_dict = {k.replace('model.', ''): v for k, v in state_dict.items()}
            else:
                state_dict = checkpoint
                logger.debug(f"Found direct state_dict with {len(state_dict)} keys")
        else:
            state_dict = checkpoint
            logger.debug("Found raw checkpoint")

        # Remove 'module.' prefix from DataParallel checkpoint if present
        if any(k.startswith('module.') for k in state_dict.keys()):
            logger.debug("Removing 'module.' prefix from DataParallel checkpoint")
            state_dict = {k.replace('module.', '', 1): v for k, v in state_dict.items()}

        incompatible = model.load_state_dict(state_dict, strict=False)

        if incompatible.missing_keys:
            logger.warning(f"Missing keys in checkpoint: {len(incompatible.missing_keys)}")
            logger.debug(f"First 5 missing keys: {incompatible.missing_keys[:5]}")

        if incompatible.unexpected_keys:
            logger.warning(f"Unexpected keys in checkpoint: {len(incompatible.unexpected_keys)}")
            logger.debug(f"First 5 unexpected keys: {incompatible.unexpected_keys[:5]}")

        model = model.to(self.device)
        model.eval()

        logger.info("Text recognition model loaded successfully")
        return model

    def recognize_text(self, image: np.ndarray) -> str:
        """Recognize text from a cropped image region.

        Args:
            image: Input image as numpy array (BGR format from OpenCV)

        Returns:
            Recognized text string
        """
        # Convert BGR to RGB
        if len(image.shape) == 3 and image.shape[2] == 3:
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        else:
            image_rgb = image

        # Convert to PIL Image
        pil_image = Image.fromarray(image_rgb)

        # Apply transformation
        img_tensor = self.transform(pil_image).unsqueeze(0).to(self.device)

        # Perform inference
        with torch.no_grad():
            preds = self.model(img_tensor, is_train=False)

        # Decode prediction
        pred_probs = torch.softmax(preds, dim=2)
        labels, confidences = self.tokenizer.decode(pred_probs)

        return labels[0]

    def recognize_batch(self, images: List[np.ndarray]) -> List[str]:
        """Recognize text from multiple cropped image regions.

        Args:
            images: List of input images as numpy arrays

        Returns:
            List of recognized text strings
        """
        if not images:
            return []

        # Preprocess all images
        img_tensors = []
        for image in images:
            # Convert BGR to RGB
            if len(image.shape) == 3 and image.shape[2] == 3:
                image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            else:
                image_rgb = image

            # Convert to PIL and apply transform
            pil_image = Image.fromarray(image_rgb)
            img_tensor = self.transform(pil_image)
            img_tensors.append(img_tensor)

        # Stack into batch
        batch_tensor = torch.stack(img_tensors).to(self.device)

        # Perform inference
        with torch.no_grad():
            preds = self.model(batch_tensor, is_train=False)

        # Decode predictions
        pred_probs = torch.softmax(preds, dim=2)
        labels, confidences = self.tokenizer.decode(pred_probs)

        return labels
