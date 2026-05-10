import torch

from transformers import AutoTokenizer, AutoModelForCausalLM

from .base_model import BaseLLM


class LazyHF(BaseLLM):
    """
    This Huggingface LLM wrapper allows one to load the model when it is actually called.
    """

    def __init__(self, model_name_or_path: str, rank: int, **kwargs):
        super().__init__()
        self.model_name_or_path = model_name_or_path
        self.rank = rank

        self._model = None
        self._tokenizer = None

    def lazy_generate(self, *args, **kwargs):
        if self._model is None:
            self._load_model()
        return self._model.generate(*args, **kwargs)

    def _load_model(self):
        device = "cpu" if not torch.cuda.is_available() else f"cuda:{self.rank}"
        if "pythia-160m" in self.model_name_or_path:
            self._model = AutoModelForCausalLM.from_pretrained(
                self.model_name_or_path,
                return_dict=True,
                trust_remote_code=True,
                torch_dtype=torch.float32,
                device_map=device,
            )
        else:
            self._model = AutoModelForCausalLM.from_pretrained(
                self.model_name_or_path,
                return_dict=True,
                torch_dtype=torch.bfloat16,
                device_map=device,
            )

        self._model.eval()

    @property
    def tokenizer(self):
        if self._tokenizer is None:
            self._tokenizer = AutoTokenizer.from_pretrained(self.model_name_or_path)
        return self._tokenizer
