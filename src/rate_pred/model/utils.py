from __future__ import annotations

import os
import random
from collections import defaultdict
from importlib.resources import files

import torch
from torch import nn
from torch.nn.utils.rnn import pad_sequence

import jieba
from pypinyin import lazy_pinyin, Style
import math
import re
from typing import List

import unicodedata
from pathlib import Path
import pyphen
from pythainlp.tokenize import syllable_tokenize

from rate_pred.model.jp_syllable import split_syllables as ja_split_syllables
from finnsyll import FinnSyll

def seed_everything(seed=0):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# helpers

def exists(v):
    return v is not None


def default(v, d):
    return v if exists(v) else d


# tensor helpers

def lens_to_mask(t: int["b"], length: int | None = None) -> bool["b n"]:  # noqa: F722 F821
    if not exists(length):
        length = t.amax() # Return the largest sequence length in the batch.

    seq = torch.arange(length, device=t.device)
    return seq[None, :] < t[:, None]

# filter func for dirty data with many repetitions

def repetition_found(text, length=2, tolerance=10):
    pattern_count = defaultdict(int)
    for i in range(len(text) - length + 1):
        pattern = text[i : i + length]
        pattern_count[pattern] += 1
    for pattern, count in pattern_count.items():
        if count > tolerance:
            return True
    return False

PYPHEN_LANG_MAP = {
    "bg": "bg_BG",
    "cs": "cs_CZ",
    "da": "da_DK",
    "de": "de_DE",
    "el": "el_GR",
    "en": "en_US",
    "es": "es_ES",
    "et": "et_EE",
    "fi": "fi_FI",
    "fr": "fr_FR",
    "hr": "hr_HR",
    "hu": "hu_HU",
    "id": "id_ID",
    "it": "it_IT",
    "lt": "lt_LT",
    "lv": "lv_LV",
    "mt": "it_IT",
    "nl": "nl_NL",
    "pl": "pl_PL",
    "pt": "pt_PT",
    "ro": "ro_RO",
    "ru": "ru_RU",
    "sk": "sk_SK",
    "sl": "sl_SI",
    "sv": "sv_SE",
}

_PYPHEN_CACHE = {}

def extract_pyphen_text(text: str) -> str:
    text = unicodedata.normalize("NFKC", text)
    tokens = re.findall(r"[^\W\d_]+(?:['’][^\W\d_]+)*", text, flags=re.UNICODE)
    return " ".join(tokens)


def count_syllables_pure(text: str, lang: str) -> int:
    if not text:
        return 0

    if lang in {"zh", "yue"}:
        return len(re.findall(r"[\u4e00-\u9fff]", text))

    if lang == "ko":
        return len(re.findall(r"[\uac00-\ud7a3]", text))

    if lang == "th":
        clean_text = "".join(
            ch
            for ch in text
            if unicodedata.category(ch)[0] in {"L", "M"} or ch.isspace()
        )
        try:
            tokens = syllable_tokenize(clean_text)
            return len(tokens) if tokens else 0
        except Exception as e:
            print(f"Failed to process {text}\n{e}")
            return 0

    if lang == "ja":
        _, count = ja_split_syllables(text)
        return count

    if lang == "vi":
        text = unicodedata.normalize("NFKC", text)
        tokens = re.findall(r"[^\W\d_]+(?:['’][^\W\d_]+)*", text, flags=re.UNICODE)
        return len(tokens)

    clean_text = extract_pyphen_text(text)
    if not clean_text:
        return 0

    pyphen_lang = PYPHEN_LANG_MAP.get(lang, "en_US")
    if pyphen_lang not in _PYPHEN_CACHE:
        try:
            if lang == "fi":
                _PYPHEN_CACHE[pyphen_lang] = FinnSyll()
            else:
                _PYPHEN_CACHE[pyphen_lang] = pyphen.Pyphen(lang=pyphen_lang)
        except Exception as e:
            print(f"Not support {lang}. Fall back to English.")
            if "en_US" not in _PYPHEN_CACHE:
                _PYPHEN_CACHE["en_US"] = pyphen.Pyphen(lang="en_US")
            pyphen_lang = "en_US"
    dic = _PYPHEN_CACHE[pyphen_lang]
    total = 0
    for word in clean_text.split():
        if word.strip():
            if lang == "fi":
                total += len(dic.syllabify(word)[0].split('.'))
            else:
                total += len(dic.inserted(word).split("-"))
    return total

PUNCT_CHARS = set(',.?!;:。，、！？；：')
def count_punctuations(text):
    punct_syllables = 0
    for char in text:
        if char in PUNCT_CHARS:
            punct_syllables += 1
    return punct_syllables

def count_syllables(text: str, lang: str) -> int:
    return count_syllables_pure(text, lang) + count_punctuations(text)
