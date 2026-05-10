from typing import Any


class BaseLLM:
    def __init__(self, *args, **kwargs) -> None:
        pass


class APILLM(BaseLLM):
    async def async_generate(self, *args, **kwargs) -> Any:
        raise NotImplementedError
