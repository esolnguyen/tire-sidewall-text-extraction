"""Custom exceptions for the tire extraction pipeline."""


class TireExtractionError(Exception):
    """Base exception for tire extraction errors."""


class TireDetectionError(TireExtractionError):
    """Raised when tire/rim detection fails."""


class SidewallFlatteningError(TireExtractionError):
    """Raised when sidewall flattening fails."""


class TextDetectionError(TireExtractionError):
    """Raised when text detection fails."""


class TextRecognitionError(TireExtractionError):
    """Raised when text recognition fails."""


class LLMExtractionError(TireExtractionError):
    """Raised when LLM extraction fails."""


class ImageLoadError(TireExtractionError):
    """Raised when image loading fails."""
