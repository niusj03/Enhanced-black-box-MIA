import torch
import argparse

from sentence_transformers import SentenceTransformer

from .base_model import BaseLLM, APILLM  # noqa: F403,F401
from .lazy_hf import LazyHF  # noqa: F403,F401
from .openai_api import OpenaiAPI  # noqa: F403,F401
from .google_api import GoogleAPI  # noqa: F403,F401
from .anthropic_api import AnthropicAPI  # noqa: F403,F401


def get_model(rank: int, args: argparse.Namespace) -> BaseLLM:
    if args.model_name_or_path.startswith("api:openai/"):
        return OpenaiAPI(model_name_or_path=args.model_name_or_path, rank=rank)
    elif args.model_name_or_path.startswith("api:google/"):
        return GoogleAPI(model_name_or_path=args.model_name_or_path, rank=rank)
    elif args.model_name_or_path.startswith("api:anthropic/"):
        return AnthropicAPI(model_name_or_path=args.model_name_or_path, rank=rank)
    else:
        return LazyHF(model_name_or_path=args.model_name_or_path, rank=rank)


def get_embedder(rank: int, args: argparse.Namespace) -> SentenceTransformer:
    device = "cpu" if not torch.cuda.is_available() else f"cuda:{rank}"
    model = SentenceTransformer(args.embedding_model, device=device)
    model.eval()
    return model
