import json
import logging
import time
from decimal import Decimal
from google import genai
from google.genai import types
from typing import Optional, List


EXTRACT_ENTIRE_INFORMATION_PROMPT = """
Your task is to perform a complete analysis of the provided tire sidewall. You have access to both OCR-extracted text and an image of the flattened tire sidewall. Use both sources to identify the manufacturer, model, and other technical details, and return them in a single JSON object.

---
**Instructions:**
---
1.  **Identify Manufacturer:** First, determine the most likely `Manufacturer`. Analyze the `OCR Text` and the tire sidewall image, and choose the best option from the `Known Manufacturer Candidates` list, which includes similarity scores. If no candidate seems correct, use your own knowledge. The image may help identify the brand logo or confirm unclear OCR text.
2.  **Identify Model:** Find the `Model` series. The `Known Model Series Candidates` list contains the most probable models from our database. Use this list, the OCR text, and the visual information from the image to select the correct one. Correct for OCR errors (e.g., a fragment like `LAZEX3` might correspond to `MILAZE X3` in the list). The image can help clarify ambiguous characters.
3.  **Extract Size & Load/Speed:** These are often combined (e.g., '225/45ZR1791V'). Separate them into `Size` ('225/45ZR17') and `LoadSpeed` ('91V'). Use the image to verify or correct OCR errors in these critical numbers.
4.  **Extract DOT Date:** Find the code starting with "DOT" and extract only the final 4-digit week/year part (WWYY). The image can help locate and verify this code if OCR missed it or got it wrong.
5.  **Extract Special Markings:** Identify any special markings from the text and image. Refer to the `Known Special Markings` list for guidance. Return only the short-form term or symbol (e.g., 'MO', 'AO', 'BMW Star', 'XL'). Visual inspection of the image may reveal markings that OCR missed.
    - **Crucially:** If you see 'M+S', 'AM+S', 'M+SA', 'M.S', 'AM.S', 'M.SA', 'MS', 'AMS', 'MSA', 'MAS' or similar variations, this is a '3PMSF' (Three-Peak Mountain Snowflake) indicator.

---

**Input Data:**
---
**OCR Text:**
{full_text}

**Flattened Tire Sidewall Image:**
[Image provided above - use this to verify and supplement the OCR text]

**Known Tire Manufacturer and Model Series Candidates (Brand, Model, Similarity Score):**
{know_tire}

---
**JSON Output Schema:**
---
Please structure your response in this exact JSON format. If a value is not found, use "Not found".

{{
    "Manufacturer": "string",
    "Model": "string",
    "Size": "string",
    "LoadSpeed": "string",
    "DOT": "string (4-digit WWYY)",
    "SpecialMarkings": ["string"]
}}

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

---
**Example Output:**
---
{{
    "Manufacturer": "Audi",
    "Model": "Pilot Sport 4S",
    "Size": "245/35ZR20",
    "LoadSpeed": "95Y",
    "DOT": "1023",
    "SpecialMarkings": ["AO", "XL"]
}}

Now, analyze the provided data and respond with only the populated JSON object.
"""


class GeminiService:
    def __init__(self, api_key: str, model: str = "gemini-2.0-flash-exp"):
        self.client = genai.Client(api_key=api_key)
        self.model = model
        self.logger = logging.getLogger(__name__)

    def _process_response(self, response: types.GenerateContentResponse, duration: float) -> dict:
        """Process Gemini API responses and handle token counting"""
        if not response.candidates:
            feedback = getattr(response, 'prompt_feedback', None)
            block_reason = getattr(feedback, 'block_reason', 'Unknown')
            safety_ratings = getattr(feedback, 'safety_ratings', 'N/A')
            self.logger.error(
                f"Gemini response blocked or empty. Reason: {block_reason}. Safety Ratings: {safety_ratings}"
            )
            raise Exception(
                f"Gemini response blocked or empty. Reason: {block_reason}. Safety Ratings: {safety_ratings}"
            )

        usage_metadata = getattr(response, 'usage_metadata', None)
        input_tokens = 0
        output_tokens = 0

        if usage_metadata:
            input_tokens = getattr(
                usage_metadata, 'prompt_token_count', 0) or 0
            thought_tokens = getattr(
                usage_metadata, 'thoughts_token_count', 0) or 0
            output_tokens = getattr(
                usage_metadata, 'candidates_token_count', 0) or 0
            output_tokens += thought_tokens
        else:
            self.logger.warning("Usage metadata not found in Gemini response.")

        response_content = ""
        candidate_content = getattr(response.candidates[0], 'content', None)
        if candidate_content is not None and getattr(candidate_content, 'parts', None):
            response_content = "".join(
                part.text for part in candidate_content.parts if getattr(part, 'text', None)
            ).strip()

        self.logger.info("LLM response received from Gemini.")
        self.logger.debug(f"Raw response snippet: {response_content[:200]}...")

        if not response_content:
            self.logger.warning("Received empty response content from Gemini.")
            raise Exception("Received empty response content from Gemini.")

        try:
            parsed_content = json.loads(response_content)
            self.logger.info("Successfully parsed LLM JSON response.")
            return {
                "content": parsed_content,
                "input_tokens": input_tokens,
                "output_tokens": output_tokens,
                "duration": Decimal(str(duration))
            }
        except json.JSONDecodeError as json_err:
            self.logger.error(
                f"Failed to parse JSON response from Gemini: {json_err}")
            self.logger.error(
                f"Content attempted to parse: {response_content}")
            raise Exception(
                f"Failed to parse JSON response from Gemini. Raw response: {response_content}"
            )

    async def extract_tire_info(
        self,
        ocr_text: str,
        known_tire_candidates: str,
        prompt_template: str
    ) -> dict:
        """
        Extract tire information using Gemini LLM

        Args:
            ocr_text: The OCR text extracted from tire sidewall
            known_tire_candidates: String containing known manufacturers and models with similarity scores
            prompt_template: The prompt template with placeholders {full_text} and {know_tire}

        Returns:
            Dictionary containing extracted tire information
        """
        self.logger.info(
            f"Sending tire information extraction request to Gemini model: {self.model}")

        try:
            start_time = time.time()

            # Format the prompt with actual data
            formatted_prompt = prompt_template.format(
                full_text=ocr_text,
                know_tire=known_tire_candidates
            )

            # Create the content for the API call
            contents = types.Content(
                role='user',
                parts=[types.Part.from_text(text=formatted_prompt)]
            )

            # Configure the API call
            config = types.GenerateContentConfig(
                response_mime_type="application/json",
                temperature=0.2,  # Lower temperature for more deterministic extraction
            )

            # Make the API call
            response = await self.client.aio.models.generate_content(
                model=self.model,
                contents=contents,
                config=config
            )

            duration = time.time() - start_time
            result = self._process_response(response, duration)

            return result["content"]

        except Exception as e:
            self.logger.error(
                f"An error occurred during the Gemini API call: {e}")
            raise Exception(
                f"An error occurred during tire info extraction: {e}")

    def extract_tire_info_sync(
        self,
        ocr_text: str,
        known_tire_candidates: str,
        prompt_template: str,
        flattened_image: Optional[bytes] = None
    ) -> dict:
        """
        Synchronous version of extract_tire_info

        Args:
            ocr_text: The OCR text extracted from tire sidewall
            known_tire_candidates: String containing known manufacturers and models with similarity scores
            prompt_template: The prompt template with placeholders {full_text} and {know_tire}
            flattened_image: Optional flattened tire sidewall image as bytes (JPEG format)

        Returns:
            Dictionary containing extracted tire information
        """
        self.logger.info(
            f"Sending tire information extraction request to Gemini model: {self.model}")

        try:
            start_time = time.time()

            # Format the prompt with actual data
            formatted_prompt = prompt_template.format(
                full_text=ocr_text,
                know_tire=known_tire_candidates
            )

            # Create parts list starting with the text prompt
            parts = [types.Part.from_text(text=formatted_prompt)]

            # Add image if provided
            if flattened_image is not None:
                self.logger.info(
                    "Including flattened tire image in LLM request")
                parts.append(types.Part.from_bytes(
                    data=flattened_image,
                    mime_type="image/jpeg"
                ))

            # Create the content for the API call
            contents = types.Content(
                role='user',
                parts=parts
            )

            # Configure the API call
            config = types.GenerateContentConfig(
                response_mime_type="application/json",
                temperature=0.2,
            )

            # Make the API call (synchronous)
            response = self.client.models.generate_content(
                model=self.model,
                contents=contents,
                config=config
            )

            duration = time.time() - start_time
            result = self._process_response(response, duration)

            return result["content"]

        except Exception as e:
            self.logger.error(
                f"An error occurred during the Gemini API call: {e}")
            raise Exception(
                f"An error occurred during tire info extraction: {e}")


def extract_tire_information(
    model: str,
    ocr_texts: List[str],
    api_key: str,
    known_tire_candidates: str = "",
    flattened_image: Optional[bytes] = None
) -> dict:
    """
    Simple wrapper to extract tire information from OCR texts.

    Args:
        ocr_texts: List of recognized text strings from the tire sidewall
        api_key: Google Gemini API key
        known_tire_candidates: Optional string with known tire models/manufacturers
        flattened_image: Optional flattened tire sidewall image as bytes (JPEG format)

    Returns:
        Dictionary with tire information (Manufacturer, Model, Size, LoadSpeed, DOT, SpecialMarkings)
    """
    # Combine all OCR texts
    combined_text = " ".join(ocr_texts)

    # Initialize Gemini service
    service = GeminiService(api_key=api_key, model=model)

    # Extract tire information
    result = service.extract_tire_info_sync(
        ocr_text=combined_text,
        known_tire_candidates=known_tire_candidates,
        prompt_template=EXTRACT_ENTIRE_INFORMATION_PROMPT,
        flattened_image=flattened_image
    )

    return result
