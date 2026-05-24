import torch
import argparse
import string
import numpy as np

from sentence_transformers import SentenceTransformer

from .base_model import BaseLLM, APILLM  # noqa: F403,F401
from .lazy_hf import LazyHF  # noqa: F403,F401
from .openai_api import OpenaiAPI  # noqa: F403,F401
from .google_api import GoogleAPI  # noqa: F403,F401
from .anthropic_api import AnthropicAPI  # noqa: F403,F401


class GensimEmbeddingAdapter:
    def __init__(self, model_name: str):
        try:
            import gensim.downloader as gensim_downloader
        except ImportError as exc:
            raise ImportError(
                "Embedding models prefixed with 'gensim:' require gensim. "
                "Install the project dependencies again or run `pip install gensim`."
            ) from exc

        self.model_name = model_name
        self.model = gensim_downloader.load(model_name)
        self.vector_size = int(self.model.vector_size)
        self.zero_vector = np.zeros(self.vector_size, dtype=np.float32)

    def eval(self):
        return self

    def _lookup(self, word: str) -> np.ndarray:
        stripped = word.strip().strip(string.punctuation)
        candidates = [
            word,
            word.lower(),
            stripped,
            stripped.lower(),
            stripped.title(),
        ]
        seen = set()
        for candidate in candidates:
            if not candidate or candidate in seen:
                continue
            seen.add(candidate)
            if candidate in self.model:
                return np.asarray(self.model[candidate], dtype=np.float32)
        return self.zero_vector.copy()

    def encode(self, sentences, show_progress_bar: bool = False):
        if isinstance(sentences, str):
            sentences = [sentences]
        return np.vstack([self._lookup(sentence) for sentence in sentences])

    @staticmethod
    def _normalize(embeddings: np.ndarray) -> np.ndarray:
        embeddings = np.asarray(embeddings, dtype=np.float32)
        if embeddings.ndim == 1:
            embeddings = embeddings.reshape(1, -1)
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        norms = np.where(norms == 0, 1.0, norms)
        return embeddings / norms

    def similarity(self, query_embeddings, corpus_embeddings):
        query_embeddings = self._normalize(query_embeddings)
        corpus_embeddings = self._normalize(corpus_embeddings)
        return np.matmul(query_embeddings, corpus_embeddings.T)


def get_model(rank: int, args: argparse.Namespace) -> BaseLLM:
    if args.model_name_or_path.startswith("api:openai/"):
        return OpenaiAPI(model_name_or_path=args.model_name_or_path, rank=rank)
    elif args.model_name_or_path.startswith("api:google/"):
        return GoogleAPI(model_name_or_path=args.model_name_or_path, rank=rank)
    elif args.model_name_or_path.startswith("api:anthropic/"):
        return AnthropicAPI(model_name_or_path=args.model_name_or_path, rank=rank)
    else:
        return LazyHF(model_name_or_path=args.model_name_or_path, rank=rank)


def get_embedder(rank: int, args: argparse.Namespace):
    if args.embedding_model.startswith("gensim:"):
        return GensimEmbeddingAdapter(args.embedding_model.removeprefix("gensim:"))

    device = "cpu" if not torch.cuda.is_available() else f"cuda:{rank}"
    model = SentenceTransformer(args.embedding_model, device=device)
    model.eval()
    return model
