import cv2
import numpy as np
from typing import Literal


def resize_image(image: np.ndarray, target_size: tuple) -> np.ndarray:
    """Resize image to target size using linear interpolation

    Args:
        image: Input image
        target_size: Target size as (width, height)

    Returns:
        Resized image
    """
    return cv2.resize(image, target_size, interpolation=cv2.INTER_LINEAR)


def normalize_image(image: np.ndarray) -> np.ndarray:
    """Normalize image to 0-255 range

    Args:
        image: Input image

    Returns:
        Normalized image
    """
    return cv2.normalize(image, None, 0, 255, cv2.NORM_MINMAX)


# ============================================================================
# Three preprocessing methods for comparison (Thesis implementation)
# ============================================================================

def linear_stretching(image: np.ndarray, min_percentile: float = 2.0,
                      max_percentile: float = 98.0) -> np.ndarray:
    """Linear Stretching (Contrast Stretching) method [14]

    This method linearly stretches the intensity values to improve contrast.
    It maps the intensity range [p_min, p_max] to [0, 255], where p_min and 
    p_max are determined by percentiles to handle outliers.

    Formula: I_out = (I_in - p_min) * (255 / (p_max - p_min))

    Args:
        image: Input image (BGR or grayscale)
        min_percentile: Lower percentile for minimum intensity (default: 2.0)
        max_percentile: Upper percentile for maximum intensity (default: 98.0)

    Returns:
        Enhanced image with stretched contrast

    Reference:
        [14] Linear Stretching for contrast enhancement
    """
    # Convert to grayscale if needed
    is_color = len(image.shape) == 3
    if is_color:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image.copy()

    # Calculate percentile values to handle outliers
    p_min = np.percentile(gray, min_percentile)
    p_max = np.percentile(gray, max_percentile)

    # Avoid division by zero
    if p_max - p_min < 1e-6:
        return image.copy()

    # Apply linear stretching
    stretched = np.clip(
        (gray - p_min) * (255.0 / (p_max - p_min)), 0, 255).astype(np.uint8)

    # Convert back to BGR if input was color
    if is_color:
        return cv2.cvtColor(stretched, cv2.COLOR_GRAY2BGR)
    return stretched


def histogram_equalization(image: np.ndarray) -> np.ndarray:
    """Histogram Equalization (HE) method [15]

    Global histogram equalization redistributes intensity values to achieve
    a uniform histogram distribution, enhancing overall contrast.

    This method applies the cumulative distribution function (CDF) transformation:
    I_out = (CDF(I_in) - CDF_min) * (L-1) / (M*N - CDF_min)
    where L=256 is the number of intensity levels, M*N is image size.

    Args:
        image: Input image (BGR or grayscale)

    Returns:
        Enhanced image with equalized histogram

    Reference:
        [15] Histogram Equalization for global contrast enhancement
    """
    # Convert to grayscale if needed
    is_color = len(image.shape) == 3
    if is_color:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image.copy()

    # Apply histogram equalization
    equalized = cv2.equalizeHist(gray)

    # Convert back to BGR if input was color
    if is_color:
        return cv2.cvtColor(equalized, cv2.COLOR_GRAY2BGR)
    return equalized


def clahe_enhancement(image: np.ndarray, clip_limit: float = 4.0,
                      tile_grid_size: tuple = (8, 8)) -> np.ndarray:
    """Contrast-Limited Adaptive Histogram Equalization (CLAHE) [16]

    CLAHE is an adaptive method that applies histogram equalization to small
    regions (tiles) of the image rather than the entire image. The contrast
    limiting prevents over-amplification of noise in homogeneous regions.

    Key advantages:
    - Adapts to local image characteristics
    - Prevents excessive noise amplification via clip limit
    - Better preserves local details than global HE

    Args:
        image: Input image (BGR or grayscale)
        clip_limit: Threshold for contrast limiting (default: 4.0)
                   Higher values = more contrast
        tile_grid_size: Size of grid for histogram equalization (default: (8, 8))
                       Smaller tiles = more local adaptation

    Returns:
        Enhanced image using CLAHE

    Reference:
        [16] Contrast-Limited Adaptive Histogram Equalization
    """
    # Convert to grayscale if needed
    is_color = len(image.shape) == 3
    if is_color:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image.copy()

    # Create CLAHE object
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)

    # Apply CLAHE
    enhanced = clahe.apply(gray)

    # Convert back to BGR if input was color
    if is_color:
        return cv2.cvtColor(enhanced, cv2.COLOR_GRAY2BGR)
    return enhanced


# ============================================================================
# Unified preprocessing function
# ============================================================================

PreprocessingMethod = Literal['linear', 'histogram_eq', 'clahe', 'none']


def preprocess_image(image: np.ndarray,
                     method: PreprocessingMethod = 'clahe',
                     target_size: tuple = None,
                     **kwargs) -> np.ndarray:
    """Apply preprocessing to image with selectable method

    This function supports multiple preprocessing methods for comparison:
    - 'linear': Linear Stretching [14]
    - 'histogram_eq': Histogram Equalization [15]
    - 'clahe': Contrast-Limited Adaptive HE [16]
    - 'none': No enhancement (only resize if target_size provided)

    Args:
        image: Input image as numpy array
        method: Preprocessing method to use
        target_size: Optional target size as (width, height) for resizing
        **kwargs: Additional parameters for specific methods:
                 - For 'linear': min_percentile, max_percentile
                 - For 'clahe': clip_limit, tile_grid_size

    Returns:
        Preprocessed image

    Examples:
        >>> # Use CLAHE with default parameters
        >>> enhanced = preprocess_image(img, method='clahe')

        >>> # Use Linear Stretching with custom percentiles
        >>> enhanced = preprocess_image(img, method='linear', 
        ...                            min_percentile=1, max_percentile=99)

        >>> # Use Histogram Equalization
        >>> enhanced = preprocess_image(img, method='histogram_eq')
    """
    # Resize first if target size is specified
    if target_size is not None:
        image = resize_image(image, target_size)

    # Apply selected enhancement method
    if method == 'linear':
        min_perc = kwargs.get('min_percentile', 2.0)
        max_perc = kwargs.get('max_percentile', 98.0)
        enhanced = linear_stretching(image, min_perc, max_perc)
    elif method == 'histogram_eq':
        enhanced = histogram_equalization(image)
    elif method == 'clahe':
        clip_limit = kwargs.get('clip_limit', 4.0)
        tile_size = kwargs.get('tile_grid_size', (8, 8))
        enhanced = clahe_enhancement(image, clip_limit, tile_size)
    elif method == 'none':
        enhanced = image
    else:
        raise ValueError(f"Unknown preprocessing method: {method}. "
                         f"Choose from: 'linear', 'histogram_eq', 'clahe', 'none'")

    return enhanced
