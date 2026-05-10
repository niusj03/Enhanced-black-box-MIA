import argparse
import logging
import torch

from collections import Counter

import transformers

from simmia._internal import SAMPLING_METHODS, ASYNC_SAMPLING_METHODS
from simmia.utils import word_tokenize
from simmia.model import LazyHF, APILLM


logger = logging.getLogger(__name__)


@SAMPLING_METHODS.register("relative_word_by_word")
def relative_word_by_word_hf(
    rank: int,
    args: argparse.Namespace,
    model: LazyHF,
    instance: dict,
) -> dict:
    results = word_by_word_hf(
        rank=rank,
        args=args,
        model=model,
        instance=instance,
    )
    prefix_results = word_by_word_hf(
        rank=rank,
        args=args,
        model=model,
        instance=instance,
        prefix_text=instance["prefix"],
    )

    return {
        **instance,
        "label_results": results["label_results"],
        "sample_results": results["sample_results"],
        "prefix_sample_results": prefix_results["sample_results"],
    }


@SAMPLING_METHODS.register("word_by_word")
def word_by_word_hf(
    rank: int,
    args: argparse.Namespace,
    model: LazyHF,
    instance: dict,
    prefix_text: str = "",
) -> dict:
    device = f"cuda:{rank}"
    text = instance["input"]

    num_samples = args.num_samples

    sample_results = []
    label_results = []
    with torch.inference_mode():
        past_key_values = transformers.DynamicCache()
        sample_kwargs = {
            "max_new_tokens": 3,
            "pad_token_id": model.tokenizer.eos_token_id,
            "output_scores": True,
            "return_dict_in_generate": True,
            "do_sample": True,
            "top_k": args.top_k,
            "use_cache": True,
            "past_key_values": past_key_values,
        }

        words = word_tokenize(text)
        prefix = prefix_text + words[0]
        prefix_tokens = model.tokenizer(prefix, return_tensors="pt")
        for word in words[1:]:
            batch_size = word_by_word_hf.__dict__.get("batch_size", num_samples)
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
                    past_key_values.crop(
                        max(prefix_tokens["input_ids"].shape[1] - 1, 0)
                    )
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
                        past_key_values.crop(
                            max(prefix_tokens["input_ids"].shape[1] - 1, 0)
                        )
                        past_key_values.batch_select_indices([0])
                        past_key_values.batch_repeat_interleave(batch_size)
                        word_by_word_hf.__dict__["batch_size"] = batch_size
                    else:
                        # sometimes OOM may cause other exceptions, try lower `batch_size` manually in these cases
                        # e.g., "RuntimeError: Key and Value must have the same sequence length"
                        raise e
            sample_texts = sample_texts[:num_samples]

            # update prefix & cache
            prefix += word
            new_prefix_tokens = model.tokenizer(prefix, return_tensors="pt")

            valid_cache_len = 0
            for i in range(prefix_tokens["input_ids"].shape[1]):
                if (
                    prefix_tokens["input_ids"][0, i]
                    == new_prefix_tokens["input_ids"][0, i]
                ):
                    valid_cache_len += 1
                else:
                    break
            past_key_values.crop(max(valid_cache_len - 1, 0))

            prefix_tokens = new_prefix_tokens

            # save statistics
            label_results.append(word)
            sample_results.append(Counter(sample_texts).most_common())

    return {
        **instance,
        "label_results": label_results,
        "sample_results": sample_results,
    }


@ASYNC_SAMPLING_METHODS.register("relative_word_by_word")
async def relative_word_by_word_api(
    rank: int,
    args: argparse.Namespace,
    model: APILLM,
    instance: dict,
) -> dict:
    results = await word_by_word_api(
        rank=rank,
        args=args,
        model=model,
        instance=instance,
    )
    prefix_results = await word_by_word_api(
        rank=rank,
        args=args,
        model=model,
        instance=instance,
        prefix_text=instance["prefix"],
    )

    return {
        **instance,
        "label_results": results["label_results"],
        "sample_results": results["sample_results"],
        "prefix_sample_results": prefix_results["sample_results"],
    }


@ASYNC_SAMPLING_METHODS.register("word_by_word")
async def word_by_word_api(
    rank: int,
    args: argparse.Namespace,
    model: APILLM,
    instance: dict,
    prefix_text: str = "",
) -> dict:
    text = instance["input"]

    sample_results = []
    label_results = []
    words = word_tokenize(text)
    prefix = prefix_text + words[0]

    num_returns = 5
    num_samples = args.num_samples
    for word in words[1:]:
        sample_texts = []
        while len(sample_texts) < num_samples:
            results = await model.async_generate(
                system_prompt="Return ONLY the single next token from the text. It can be punctuation or one whole word. No spaces, no quotes, no extra text.",
                user_message=f'Text so far: "{prefix}"',
                num_returns=num_returns,
            )
            sample_texts.extend(results)
        prefix += word
        label_results.append(word)
        sample_results.append(Counter(sample_texts[:num_samples]).most_common())
    return {
        **instance,
        "label_results": label_results,
        "sample_results": sample_results,
    }
