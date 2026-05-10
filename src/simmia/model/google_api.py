import os
import asyncio
import logging

from .base_model import APILLM


logger = logging.getLogger(__name__)


class GoogleAPI(APILLM):
    """
    This wrapper calls LLM APIs in an asynchronous manner.
    """

    def __init__(
        self, model_name_or_path: str, rank: int, max_retry: int = 3, **kwargs
    ):
        super().__init__()
        self.model_name_or_path = model_name_or_path
        self.rank = rank
        self.max_retry = max_retry

        from google import genai

        api_key = os.environ.get("GOOGLE_API_KEY")
        if not api_key:
            raise ValueError(
                "GOOGLE_API_KEY environment variable is required for Gemini models. "
                "Please set it using: export GOOGLE_API_KEY='your-api-key'"
            )
        self.client = genai.Client(api_key=api_key)

    async def async_generate(
        self,
        system_prompt: str,
        user_message: str,
        num_returns: int,
        max_output_tokens: int = 20,
        **kwargs
    ):
        from google.genai import types

        system_instruction = types.Part(text=system_prompt)
        contents = [{"role": "user", "parts": [{"text": user_message}]}]
        for try_time in range(1, self.max_retry + 1):
            try:
                resp = await self.client.aio.models.generate_content(
                    model=self.model_name_or_path[len("api:google/") :],
                    contents=contents,
                    config={
                        "system_instruction": system_instruction,
                        "max_output_tokens": max_output_tokens,
                        "candidate_count": num_returns,
                        "thinking_config": types.ThinkingConfig(thinking_budget=0),
                    },
                )
                sample_texts = []
                if resp.candidates:
                    for candidate in resp.candidates[:num_returns]:
                        if candidate.content and candidate.content.parts:
                            sample_text = candidate.content.parts[0].text.strip()
                            sample_texts.append(sample_text)
                return sample_texts
            except Exception as e:
                if try_time == self.max_retry:
                    logger.exception("Unexpected error calling Gemini: %s", e)
                    raise e
                await asyncio.sleep(1)
