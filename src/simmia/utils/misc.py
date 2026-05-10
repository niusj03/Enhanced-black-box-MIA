import json
import nltk
from typing import List


def load_jsonl(path):
    with open(path, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f]


def save_jsonl(data, path):
    with open(path, "w", encoding="utf-8") as f:
        for item in data:
            f.write(json.dumps(item) + "\n")


def _quote_candidates(suffix: str):
    # order matters: try exact " first if present in text
    cands = []
    if suffix.startswith('"'):
        cands.append('"')
    if suffix.startswith("''"):
        cands.append("''")
    if suffix.startswith("``"):
        cands.append("``")
    if suffix.startswith("’"):
        cands.append("’")
    if suffix.startswith("'"):
        cands.append("'")
    return cands


def word_tokenize(text: str) -> List[str]:
    words = nltk.word_tokenize(text)

    tokens = []
    suffix = text
    prefix = ""
    i = 0

    while i < len(words):
        w = words[i]

        if (
            i + 2 < len(words)
            and w == "$"
            and words[i + 1] == "''"
            and words[i + 2] == "$"
            and suffix.startswith('$"$')
        ):
            tokens.append(prefix + '$"$')
            prefix = ""
            suffix = suffix[len('$"$') :]
            i += 3
            continue

        if w in {"``", "''"}:
            cands = _quote_candidates(suffix)
            matched = None
            for cand in cands:
                if suffix.startswith(cand):
                    matched = cand
                    break
            if matched is not None:
                tokens.append(prefix + matched)
                prefix = ""
                suffix = suffix[len(matched) :]
                i += 1
                continue

        if suffix.startswith(w):
            tokens.append(prefix + w)
            prefix = ""
            suffix = suffix[len(w) :]
            i += 1
            continue

        if w in {"''", "``"} and suffix and suffix[0] == '"':
            tokens.append(prefix + '"')
            prefix = ""
            suffix = suffix[1:]
            i += 1
            continue

        if suffix:
            prefix += suffix[0]
            suffix = suffix[1:]
            continue
        break

    if suffix:
        tokens.append(suffix)

    assert "".join(tokens) == text, f'{words} -> {tokens} != "{text}"'
    assert any([len(token) > 0 for token in tokens]), f"{tokens}"

    return tokens


def extract_first_word(token: str) -> str:
    words = nltk.word_tokenize(token.replace("Ġ", " ").replace("▁", " ").strip())
    if len(words) > 0:
        return words[0]
    else:
        return token


def get_start_idx(len_words, prefix_ratio):
    if prefix_ratio <= 0.0:
        return 0
    else:
        return int(len_words * prefix_ratio)
