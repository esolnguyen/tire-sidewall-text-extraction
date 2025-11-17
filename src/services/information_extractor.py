from typing import List, Dict, Any

def extract_tire_information(ocr_text: str, known_candidates: List[str]) -> Dict[str, Any]:
    """
    Extract structured tire information from OCR text based on known candidates.

    Args:
        ocr_text (str): The text recognized from the tire sidewall.
        known_candidates (List[str]): A list of known tire information candidates.

    Returns:
        Dict[str, Any]: A dictionary containing extracted tire information.
    """
    tire_info = {}
    
    # Example extraction logic (to be customized based on actual requirements)
    for candidate in known_candidates:
        if candidate.lower() in ocr_text.lower():
            tire_info[candidate] = ocr_text.lower().split(candidate.lower())[1].strip().split()[0]

    return tire_info