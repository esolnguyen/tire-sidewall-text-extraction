"""Gemini LLM service for tire information extraction."""

import json
import logging
import time
from decimal import Decimal
from google import genai
from google.genai import types
from typing import Optional, List

logger = logging.getLogger(__name__)


# ── Prompt ───────────────────────────────────────────────────────────────

response_schema = {
    "type": "object",
    "properties": {
        "Manufacturer": {
            "type": "string",
            "description": "The manufacturer of the tire",
        },
        "Model": {"type": "string", "description": "The model name or number"},
        "Size": {"type": "string", "description": "The tire size dimensions"},
        "LoadSpeed": {
            "type": "string",
            "description": "The load index and speed rating",
        },
        "DOT": {"type": "string", "description": "The 4-digit WWYY DOT date code"},
        "SpecialMarkings": {
            "type": "array",
            "items": {"type": "string"},
            "description": "A list of any special markings found",
        },
    },
    "required": [
        "Manufacturer",
        "Model",
        "Size",
        "LoadSpeed",
        "DOT",
        "SpecialMarkings",
    ],
}

EXTRACT_ENTIRE_INFORMATION_PROMPT = """
Your task is to perform a complete analysis of the provided tire sidewall. You have access to both OCR-extracted text and an image of the flattened tire sidewall. Use both sources to identify the manufacturer, model, and other technical details, and return them in a single JSON object.

---
**Instructions:**
---
1.  **Identify Manufacturer:** First, determine the most likely `Manufacturer`. Analyze the `OCR Text` and the tire sidewall image, and choose the best option from the `Known Manufacturer Candidates` list, which includes similarity scores. If no candidate seems correct, use your own knowledge. The image may help identify the brand logo or confirm unclear OCR text.
2.  **Identify Model (CRITICAL):** Find the `Model` series. The `Known Model Series Candidates` list contains the most probable models from our database. Use this list, the OCR text, and the visual information from the image to select the correct one. Correct for OCR errors (e.g., a fragment like `LAZEX3` might correspond to `MILAZE X3` in the list). The image can help clarify ambiguous characters.
3.  **Extract Size & Load/Speed:** These are often combined (e.g., '225/45ZR1791V'). Separate them into `Size` ('225/45ZR17') and `LoadSpeed` ('91V'). Use the image to verify or correct OCR errors in these critical numbers.
4. DOT & Date Code (Advanced Logic - Priority Step):**
   - **Target Format:** `WWYY` (4 digits).
   - **Step A: Candidate Search:** Identify all 4-digit number candidates in the text.
   - **Step B: Prioritization:**
     - If multiple 4-digit numbers exist, choose the one physically closest to the text "DOT" or enclosed in an oval border in the image.
   - **Step C: Validation & OCR Correction (The "Guessing" Logic):**
     - **Week Check:** `WW` must be `01` to `53`.
     - **Year Check:** `YY` represents the year (e.g., `19` = 2019).
     - **Future Date Handling:**
       - If `YY` > `25` (e.g., "3528" implies 2028), this is physically impossible (Future Date).
       - **Action:** Analyze the image and the digits to guess the correction. OCR often mistakes similar shapes:
         - `8` might be `3`, `6`, or `0`. (e.g., "3528" -> likely "3523" or "3520").
         - `9` might be `0` or `8`.
       - If a logical correction brings the year to the past/present (<= 25), apply it.
       - If no logical correction works, mark as "Not found".

5.  **Extract Special Markings:** Identify any special markings from the text and image. Refer to the `Known Special Markings` list for guidance. Return only the short-form term or symbol (e.g., 'MO', 'AO', 'BMW Star', 'XL'). Visual inspection of the image may reveal markings that OCR missed.
    - **Crucially:** If you see 'M+S', 'AM+S', 'M+SA', 'M.S', 'AM.S', 'M.SA', 'MS', 'AMS', 'MSA', 'MAS' or similar variations, this is a '3PMSF' (Three-Peak Mountain Snowflake) indicator.

---

**Input Data:**
---
**OCR Text with Bounding Boxes:**
{full_text}

Note: Each text entry includes its bounding box coordinates (x1, y1, x2, y2) showing its position on the flattened tire image. Use this spatial information to understand the layout and relative positions of text elements.

**Flattened Tire Sidewall Image:**
[Image provided above - use this to verify and supplement the OCR text. The bounding box coordinates correspond to positions in this image.]

---
**Verification & Confidence Check:**
---
**IMPORTANT:** If you are not confident/certain about your extraction result for any field, use the provided OCR text with bounding boxes to verify and validate your result.

Specifically:
- Cross-reference your visual interpretation with the OCR text provided
- Check if OCR recognized similar-looking characters (e.g., O vs 0, I vs 1, l vs L, u vs v)
- Verify spatial positioning: does the text location match the typical tire layout?
- If OCR shows a different value than what you see visually, analyze both carefully and choose the most reliable source
- Use the bounding box information to confirm text is in the expected location (e.g., DOT code should be at bottom of tire)

---
**JSON Output Schema:**
---
Please structure your response in this exact JSON format. If a value is not found, use "Not found".

---
**Known Special Markings Reference:**
---
- 'A', 'X', '+': 'BMW Star' (Also 'X', 'A', '+')
- 'AMx': 'Aston Martin OE Fitments'
- 'AO': 'Audi original manufacturer fitment'
- 'B': 'Bias belted motorcycle tire'
- 'BSW': 'Black sidewall'
- 'E4': 'ECE-regulations approved'
- 'ELT': 'Pirelli Elect (for electric cars)'
- 'J': 'Jaguar original manufacturer fitment'
- 'M+S', 'M.S', 'MS': '3PMSF'
- 'A/T', 'AT': 'All Terrain'
- 'M/T', 'MT': 'Mud Terrain'
- 'MGT': 'Maserati Genuine Tire'
- 'MO': 'Mercedes-Benz original tires'
- 'MOE': 'Mercedes-Benz Extended Mobility'
- 'N0'-'N6': 'Porsche original tires'
- 'OWL': 'Outlined white lettering'
- 'RF': 'Reinforced'
- 'RFT': 'Run-flat tire'
- 'SL': 'Standard load'
- 'TPC': 'GM OE fitment'
- 'XL': 'Extra load'
- 'ZP': 'Michelin zero-pressure (run-flat)'

Now, analyze the provided data and respond with only the populated JSON object.
"""


# ── Service ──────────────────────────────────────────────────────────────


class GeminiService:
    def __init__(self, api_key: str, model: str = "gemini-2.0-flash-exp"):
        self.client = genai.Client(api_key=api_key)
        self.model = model

    def _process_response(
        self, response: types.GenerateContentResponse, duration: float
    ) -> dict:
        """Process Gemini API responses and handle token counting."""
        if not response.candidates:
            feedback = getattr(response, "prompt_feedback", None)
            block_reason = getattr(feedback, "block_reason", "Unknown")
            safety_ratings = getattr(feedback, "safety_ratings", "N/A")
            logger.error(
                f"Gemini response blocked or empty. Reason: {block_reason}. Safety: {safety_ratings}"
            )
            raise Exception(f"Gemini response blocked. Reason: {block_reason}")

        usage_metadata = getattr(response, "usage_metadata", None)
        input_tokens = 0
        output_tokens = 0

        if usage_metadata:
            input_tokens = getattr(usage_metadata, "prompt_token_count", 0) or 0
            thought_tokens = getattr(usage_metadata, "thoughts_token_count", 0) or 0
            output_tokens = getattr(usage_metadata, "candidates_token_count", 0) or 0
            output_tokens += thought_tokens

        response_content = ""
        candidate_content = getattr(response.candidates[0], "content", None)
        if candidate_content is not None and getattr(candidate_content, "parts", None):
            response_content = "".join(
                part.text
                for part in candidate_content.parts
                if getattr(part, "text", None)
            ).strip()

        if not response_content:
            raise Exception("Received empty response content from Gemini.")

        try:
            parsed_content = json.loads(response_content)
            return {
                "content": parsed_content,
                "input_tokens": input_tokens,
                "output_tokens": output_tokens,
                "duration": Decimal(str(duration)),
            }
        except json.JSONDecodeError:
            raise Exception(
                f"Failed to parse JSON from Gemini. Raw: {response_content}"
            )

    async def extract_tire_info(
        self,
        ocr_text: str,
        known_tire_candidates: str,
        prompt_template: str,
    ) -> dict:
        """Async version of tire info extraction."""
        logger.info(f"Sending request to Gemini model: {self.model}")

        start_time = time.time()
        formatted_prompt = prompt_template.format(
            full_text=ocr_text, know_tire=known_tire_candidates
        )
        contents = types.Content(
            role="user",
            parts=[types.Part.from_text(text=formatted_prompt)],
        )
        config = types.GenerateContentConfig(
            response_mime_type="application/json",
            temperature=0,
            top_k=1,
            top_p=1,
            max_output_tokens=2048,
            seed=42,
            response_schema=response_schema,
        )
        response = await self.client.aio.models.generate_content(
            model=self.model,
            contents=contents,
            config=config,
        )
        duration = time.time() - start_time
        return self._process_response(response, duration)["content"]

    def extract_tire_info_sync(
        self,
        ocr_text: str,
        known_tire_candidates: str,
        prompt_template: str,
        flattened_image: Optional[bytes] = None,
    ) -> dict:
        """Synchronous tire info extraction with optional image."""
        logger.info(f"Sending request to Gemini model: {self.model}")

        start_time = time.time()
        formatted_prompt = prompt_template.format(
            full_text=ocr_text, know_tire=known_tire_candidates
        )
        parts = [types.Part.from_text(text=formatted_prompt)]
        if flattened_image is not None:
            parts.append(
                types.Part.from_bytes(data=flattened_image, mime_type="image/jpeg")
            )

        contents = types.Content(role="user", parts=parts)
        config = types.GenerateContentConfig(
            response_mime_type="application/json",
            temperature=0,
            top_k=1,
            top_p=1,
            max_output_tokens=2048,
            seed=42,
            response_schema=response_schema,
        )
        response = self.client.models.generate_content(
            model=self.model,
            contents=contents,
            config=config,
        )
        duration = time.time() - start_time
        return self._process_response(response, duration)["content"]


# ── Convenience wrappers ──────────────────────────────────────────────────


def extract_tire_information_raw(
    model: str,
    api_key: str,
    image_bytes: bytes,
) -> Optional[dict]:
    """Call Gemini directly with a raw image, no OCR preprocessing."""
    try:
        return extract_tire_information(
            model=model,
            ocr_texts=[],
            api_key=api_key,
            flattened_image=image_bytes,
        )
    except Exception as e:
        logger.error(f"Raw LLM call failed: {e}")
        return None


def extract_tire_information(
    model: str,
    ocr_texts: List[str],
    api_key: str,
    known_tire_candidates: str = "",
    flattened_image: Optional[bytes] = None,
) -> dict:
    """Extract tire information from OCR texts via Gemini."""
    combined_text = " ".join(ocr_texts)
    service = GeminiService(api_key=api_key, model=model)
    return service.extract_tire_info_sync(
        ocr_text=combined_text,
        known_tire_candidates=known_tire_candidates,
        prompt_template=EXTRACT_ENTIRE_INFORMATION_PROMPT,
        flattened_image=flattened_image,
    )
