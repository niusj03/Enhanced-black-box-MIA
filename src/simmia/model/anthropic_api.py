import os
import asyncio
import logging
import random
from .base_model import APILLM


logger = logging.getLogger(__name__)


class AnthropicAPI(APILLM):
    """
    This wrapper calls LLM APIs in an asynchronous manner.
    """

    def __init__(
        self,
        model_name_or_path: str,
        rank: int,
        max_retry: int = 3,
        rpm_limit: int = 50,
        **kwargs
    ):
        super().__init__()
        self.model_name_or_path = model_name_or_path
        self.rank = rank
        self.max_retry = max_retry
        self.min_interval = 60.0 / rpm_limit

        from anthropic import AsyncAnthropic

        api_key = os.environ.get("ANTHROPIC_API_KEY")
        if not api_key:
            raise ValueError(
                "ANTHROPIC_API_KEY environment variable is required for Claude models. "
                "Please set it using: export ANTHROPIC_API_KEY='your-api-key'"
            )
        self.client = AsyncAnthropic(
            api_key=api_key,
        )

    async def async_generate(
        self,
        system_prompt: str,
        user_message: str,
        num_returns: int,
        max_output_tokens: int = 3,
        **kwargs
    ):
        sample_texts = []
        for try_time in range(1, self.max_retry + 1):
            await asyncio.sleep(self.min_interval + random.uniform(0, 0.2))
            try:
                resp = await self.client.messages.create(
                    model=self.model_name_or_path[len("api:anthropic/") :],
                    max_tokens=max_output_tokens,
                    system=system_prompt,
                    messages=[{"role": "user", "content": user_message}],
                )
                # Anthropic API returns content as a list of text blocks
                if resp.content and len(resp.content) > 0:
                    sample_text = resp.content[0].text.strip()
                    sample_texts.append(sample_text)
                return sample_texts
            except Exception as e:
                if "content" in str(e).lower() and "policy" in str(e).lower():
                    sample_texts.append("<FILTERED>")
                    return sample_texts
                else:
                    if try_time == self.max_retry:
                        logger.exception("Unexpected error calling Claude: %s", e)
                        raise e
                    await asyncio.sleep(1)
