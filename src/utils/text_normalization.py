def normalize_text(text: str) -> str:
    """
    Normalize the input text by converting it to lowercase and stripping whitespace.
    This helps in ensuring consistency for comparison against ground truth.
    """
    return text.lower().strip()


def remove_special_characters(text: str) -> str:
    """
    Remove special characters from the text to ensure clean output.
    This can help in further processing and comparison.
    """
    return ''.join(e for e in text if e.isalnum() or e.isspace())


def clean_text(text: str) -> str:
    """
    Clean the text by normalizing and removing special characters.
    This function combines normalization and special character removal.
    """
    normalized = normalize_text(text)
    cleaned = remove_special_characters(normalized)
    return cleaned