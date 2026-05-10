import os
import asyncio
import logging

from .base_model import APILLM


logger = logging.getLogger(__name__)


class OpenaiAPI(APILLM):
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

        from openai import AsyncOpenAI

        api_key = os.environ.get("OPENAI_API_KEY")
        if not api_key:
            raise ValueError(
                "OPENAI_API_KEY environment variable is required for GPT models. "
                "Please set it using: export OPENAI_API_KEY='your-api-key'"
            )
        self.client = AsyncOpenAI(
            api_key=api_key,
        )

    async def async_generate(
        self, system_prompt: str, user_message: str, num_returns: int, **kwargs
    ):
        system_msg = {
            "role": "system",
            "content": [{"type": "text", "text": system_prompt}],
        }
        user_msg = {"role": "user", "content": [{"type": "text", "text": user_message}]}
        for try_time in range(1, self.max_retry + 1):
            try:
                resp = await self.client.chat.completions.create(
                    model=self.model_name_or_path[len("api:openai/") :],
                    messages=[system_msg, user_msg],
                    n=num_returns,
                )
                sample_texts = [
                    choice.message.content.strip() for choice in resp.choices
                ]
                return sample_texts
            except Exception as e:
                if try_time == self.max_retry:
                    logger.exception("Unexpected error calling GPT: %s", e)
                    raise e
                await asyncio.sleep(1)
