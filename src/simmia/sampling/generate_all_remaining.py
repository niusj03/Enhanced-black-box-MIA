import argparse
import logging
import torch

from simmia._internal import SAMPLING_METHODS, ASYNC_SAMPLING_METHODS
from simmia.model import LazyHF, APILLM


logger = logging.getLogger(__name__)


def get_prefix(text: str, prefix_ratio: float) -> tuple[str, str]:
    """Split text into prefix and suffix based on ratio."""
    num_words = len(text.split())
    num_prefix_words = int(num_words * prefix_ratio)
    prefix = " ".join(text.split()[:num_prefix_words])
    suffix = " ".join(text.split()[num_prefix_words:])
    return prefix, suffix


@SAMPLING_METHODS.register("generate_all_remaining")
def generate_all_remaining_hf(
    rank: int,
    args: argparse.Namespace,
    model: LazyHF,
    instance: dict,
    prefix_text: str = "",
) -> dict:
    device = f"cuda:{rank}"
    text = instance["input"]

    num_samples = args.num_samples

    with torch.inference_mode():
        # Split text into prefix and reference suffix
        prefix, suffix = get_prefix(text, args.prefix_ratio)
        prefix_tokens = model.tokenizer([prefix], return_tensors="pt", max_length=1024)

        sample_kwargs = {
            "max_length": args.max_length,
            "pad_token_id": model.tokenizer.eos_token_id,
            "output_scores": True,
            "return_dict_in_generate": True,
            "do_sample": True,
            "top_k": args.top_k,
            "top_p": 1.0,
            "temperature": 1.0,
            "use_cache": True,
        }

        batch_size = generate_all_remaining_hf.__dict__.get("batch_size", num_samples)
        sample_texts = []
        while len(sample_texts) < num_samples:
            try:
                sample_generation = model.lazy_generate(
                    input_ids=prefix_tokens["input_ids"].to(device),
                    attention_mask=prefix_tokens["attention_mask"].to(device),
                    num_return_sequences=batch_size,
                    **sample_kwargs,
                )
                _sample_texts = model.tokenizer.batch_decode(
                    sample_generation.sequences[
                        :, prefix_tokens["input_ids"].shape[1] :
                    ],
                    skip_special_tokens=True,
                )
                sample_texts.extend(_sample_texts)
            except RuntimeError as e:
                if "CUDA out of memory." in str(e):
                    logger.info(
                        f"[{device}] Reducing batch_size=[{batch_size} -> {batch_size // 2}]"
                    )
                    batch_size = batch_size // 2
                    if batch_size < 1:
                        raise RuntimeError(
                            "CUDA out of memory, while batch_size reaches 0."
                        )
                    generate_all_remaining_hf.__dict__["batch_size"] = batch_size
                    torch.cuda.empty_cache()
                else:
                    # sometimes OOM may cause other exceptions, try lower `batch_size` manually in these cases
                    # e.g., "RuntimeError: Key and Value must have the same sequence length"
                    raise e

        sample_results = sample_texts[:num_samples]
        sample_results = [prefix + " " + result for result in sample_results]

    return {
        **instance,
        "sample_results": sample_results,
        "prefix": prefix,
        "suffix": suffix,
    }


@ASYNC_SAMPLING_METHODS.register("generate_all_remaining")
async def generate_all_remaining_api(
    rank: int,
    args: argparse.Namespace,
    model: APILLM,
    instance: dict,
    prefix_text: str = "",
) -> dict:
    text = instance["input"]

    # Split text into prefix and reference suffix
    prefix, suffix = get_prefix(text, args.prefix_ratio)

    # System prompt for text continuation
    system_prompt = (
        "You are a pure text-continuation engine. "
        "Given the incomplete text that follows, continue it naturally "
        "and stop when it feels complete. "
        "Do **not** add explanations, do **not** ask questions, "
        "do **not** start a new paragraph unless the continuation demands it."
    )

    num_returns = 5
    num_samples = args.num_samples
    sample_texts = []

    while len(sample_texts) < num_samples:
        results = await model.async_generate(
            system_prompt=system_prompt,
            user_message=prefix,
            num_returns=num_returns,
            temperature=1.0,
            top_p=1.0,
        )
        sample_texts.extend(results)

    sample_results = sample_texts[:num_samples]
    sample_results = [prefix + " " + result for result in sample_results]

    return {
        **instance,
        "sample_results": sample_results,
        "prefix": prefix,
        "suffix": suffix,
    }
