# ruff: noqa: F722 F821

from __future__ import annotations

import os
import random
from collections import defaultdict
from importlib.resources import files

import jieba
import torch
from pypinyin import Style, lazy_pinyin
from torch.nn.utils.rnn import pad_sequence

import re

# seed everything


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


def is_package_available(package_name: str) -> bool:
    try:
        import importlib

        package_exists = importlib.util.find_spec(package_name) is not None
        return package_exists
    except Exception:
        return False


# tensor helpers


def lens_to_mask(t: int["b"], length: int | None = None) -> bool["b n"]:
    if not exists(length):
        length = t.amax()

    seq = torch.arange(length, device=t.device)
    return seq[None, :] < t[:, None]


def mask_from_start_end_indices(seq_len: int["b"], start: int["b"], end: int["b"]):
    max_seq_len = seq_len.max().item()
    seq = torch.arange(max_seq_len, device=start.device).long()
    start_mask = seq[None, :] >= start[:, None]
    end_mask = seq[None, :] < end[:, None]
    return start_mask & end_mask


def mask_from_frac_lengths(seq_len: int["b"], frac_lengths: float["b"]):
    lengths = (frac_lengths * seq_len).long()
    max_start = seq_len - lengths

    rand = torch.rand_like(frac_lengths)
    start = (max_start * rand).long().clamp(min=0)
    end = start + lengths

    return mask_from_start_end_indices(seq_len, start, end)


def mask_from_prompt_lens(prompt_lens: int["b"], lens: int["b"]):
    return mask_from_start_end_indices(lens, prompt_lens, lens)


def maybe_masked_mean(t: float["b n d"], mask: bool["b n"] = None) -> float["b d"]:
    if not exists(mask):
        return t.mean(dim=1)

    t = torch.where(mask[:, :, None], t, torch.tensor(0.0, device=t.device))
    num = t.sum(dim=1)
    den = mask.float().sum(dim=1)

    return num / den.clamp(min=1.0)


# simple utf-8 tokenizer, since paper went character based
def list_str_to_tensor(text: list[str], padding_value=-1) -> int["b nt"]:
    list_tensors = [torch.tensor([*bytes(t, "UTF-8")]) for t in text]  # ByT5 style
    text = pad_sequence(list_tensors, padding_value=padding_value, batch_first=True)
    return text


# char tokenizer, based on custom dataset's extracted .txt file
def list_str_to_idx(
    text: list[str] | list[list[str]],
    vocab_char_map: dict[str, int],  # {char: idx}
    padding_value=-1,
) -> int["b nt"]:
    list_idx_tensors = [torch.tensor([vocab_char_map.get(c, 0) for c in t]) for t in text]  # pinyin or char style
    text = pad_sequence(list_idx_tensors, padding_value=padding_value, batch_first=True)
    return text

def get_ipa_id(in_language: str) -> str:
    LANG_MAP = {
        "zh": "cmn",
        "en": "en-us",
        "fr": "fr-fr",
        "pt": "pt-br",
    }
    return LANG_MAP.get(in_language, in_language)

# Version 3 tokenizes by spaces and only splits phonemes from tone markers (123, etc.).
def str_to_list_ipa_v3(phonemized: str) -> List[str]:
    fields = []
    # Split by spaces first.
    words = phonemized.split(" ")
    for i, word in enumerate(words):
        if not word: continue
        processed_tokens = []
        # Then split each word by the phoneme separator "|".
        tokens = word.split("|")
        # Drop empty fields and separate phonemes from tone digits.
        for token in tokens:
            if not token: continue
            match = re.match(r'^([^0-9]+)([0-9]+)$', token)
            if match:
                phoneme_part = match.group(1)
                tone_part = match.group(2)
                processed_tokens.append(phoneme_part)
                processed_tokens.append(tone_part)
            else:
                processed_tokens.append(token)
        fields.extend(processed_tokens)
        if i < len(words) - 1:
            fields.append(" ")
    return fields


TOKEN_EXTRACTOR = re.compile(r"([a-zA-Z\u00C0-\u02AF]+)|(.)")
IPA_NORMALIZATION_MAP = {
    # Normalize unstable symbol combinations.
    # vowel+consonant / consonant+vowel
    "eɪɛ": "eɪ|ɛ", "zeɪɛ": "z|eɪ|ɛ", "teɪɛ": "t|eɪ|ɛ", "əeɪ": "ə|eɪ", "aɪɛ": "aɪ|ɛ", "taɪ": "t|aɪ", 
    "jap": "ja|p", "jud": "ju|d", "jaɛ": "ja|ɛ", "ʃja": "ʃ|ja", "jat": "ja|t", "ɑja": "ɑ|ja", "əlɹ": "əl|ɹ", "əlf": "əl|f", "oʊw": "oʊ|w", 
    "daʊ":"d|aʊ", "meɪ":"m|eɪ", "taʊ":"t|aʊ", "daɪ":"d|aɪ",
}
GARBAGE_CHARS = r'[&ㄜでかすπ%@]'

# First try to match a full alphabetic / extended-Latin chunk as one token.
# If that fails, fall back to matching individual characters.
def str_to_list_ipa_v6(phonemized: str) -> List[str]:
    fields = []
    # Split by spaces first.
    text = re.sub(GARBAGE_CHARS, '', phonemized)
    sorted_map_keys = sorted(IPA_NORMALIZATION_MAP.keys(), key=len, reverse=True)
    for key in sorted_map_keys:
        text = text.replace(key, IPA_NORMALIZATION_MAP[key])
    words = text.split(" ")
    for i, word in enumerate(words):
        if not word: continue
        processed_tokens = []
        # Then split each word by the phoneme separator "|".
        tokens = word.split("|")
        # Drop empty fields and extract normalized tokens.
        for token in tokens:
            if not token: continue
            for m in TOKEN_EXTRACTOR.finditer(token):
                text_chunk = m.group(1) # Alphabetic chunk.
                symbol_chunk = m.group(2) # Single symbol.
                if text_chunk:
                    fields.append(text_chunk)
                elif symbol_chunk:
                    fields.append(symbol_chunk)
        if i < len(words) - 1:
            fields.append(" ")
    return fields

def str_to_list_ipa_all(ipa_string: str, tokenizer, language_id=None) -> List[str]:
    fields = []
    if tokenizer == "ipa_v3":
        # print("Uncomment this for tokenizer debugging in utils.py.")
        fields = str_to_list_ipa_v3(ipa_string)
    elif tokenizer == "ipa_v6":
        fields = str_to_list_ipa_v6(ipa_string)
    else: 
        print(f"{tokenizer} is not a supported  version of ipa.")
    return fields

def list_str_to_idx_ipa(
    text: list[str],
    vocab_char_map: dict[str, int],
    tokenizer: str,
    padding_value: int = -1, 
    language_ids: list[str] | None = None,
) -> torch.Tensor:
    
    unk_idx = vocab_char_map.get("<pad>", 0)
    assert len(text)==len(language_ids), "text and language ids must have the same length"
    batch_indices = []
    fields=[]
    for i, ipa_string in enumerate(text):
        fields = str_to_list_ipa_all(ipa_string, tokenizer, language_ids[i])
        if fields:
            indices=[]
            for token in fields:
                idx = vocab_char_map.get(token, unk_idx)
                if idx == unk_idx and token != "…":
                    print(f"{token} is invalid")
                    token_indices = [vocab_char_map.get(t, unk_idx) for t in token]
                    indices.extend(token_indices)
                else:
                    indices.append(idx) 
            # indices = [vocab_char_map.get(token, unk_idx) for token in fields]
            batch_indices.append(torch.tensor(indices, dtype=torch.long))
        
    padded_batch = pad_sequence(
        batch_indices, 
        batch_first=True, 
        padding_value=padding_value
    )
    
    return padded_batch

def list_list_to_idx(
    text: list[list[str]],
    vocab_char_map: dict[str, int],
    padding_value: int = -1, 
) -> torch.Tensor:
    
    unk_idx = vocab_char_map.get("<pad>", 0)
    batch_indices = []
    for i, ipa_list in enumerate(text):
        if ipa_list:
            indices=[]
            for token in ipa_list:
                idx = vocab_char_map.get(token, unk_idx)
                if idx == unk_idx and token != "…":
                    print(f"{token} is invalid")
                    token_indices = [vocab_char_map.get(t, unk_idx) for t in token]
                    indices.extend(token_indices)
                else:
                    indices.append(idx) 
            batch_indices.append(torch.tensor(indices, dtype=torch.long))
        
    padded_batch = pad_sequence(
        batch_indices, 
        batch_first=True, 
        padding_value=padding_value
    )
    return padded_batch





# Get tokenizer


def get_tokenizer(dataset_name, tokenizer, sft: bool = False):
    """
    tokenizer   - "pinyin" do g2p for only chinese characters, need .txt vocab_file
                - "char" for char-wise tokenizer, need .txt vocab_file
                - "byte" for utf-8 tokenizer
                - "custom" if you're directly passing in a path to the vocab.txt you want to use
    vocab_size  - if use "pinyin", all available pinyin types, common alphabets (also those with accent) and symbols
                - if use "char", derived from unfiltered character & symbol counts of custom dataset
                - if use "byte", set to 256 (unicode byte range)
    """
    if tokenizer in ["pinyin", "char"] or tokenizer.startswith("ipa"):
        tokenizer_path = os.path.join(os.getcwd(), "data", f"{dataset_name}_{tokenizer}/vocab.txt")
        with open(tokenizer_path, "r", encoding="utf-8") as f:
            vocab_char_map = {}
            for i, char in enumerate(f):
                vocab_char_map[char[:-1]] = i
        vocab_size = len(vocab_char_map)

    elif tokenizer == "byte":
        vocab_char_map = None
        vocab_size = 256

    elif tokenizer == "custom":
        with open(dataset_name, "r", encoding="utf-8") as f:
            vocab_char_map = {}
            for i, char in enumerate(f):
                vocab_char_map[char[:-1]] = i
        vocab_size = len(vocab_char_map)

    return vocab_char_map, vocab_size


# convert char to pinyin


def convert_char_to_pinyin(text_list, polyphone=True):
    if jieba.dt.initialized is False:
        jieba.default_logger.setLevel(50)  # CRITICAL
        jieba.initialize()

    final_text_list = []
    custom_trans = str.maketrans(
        {";": ",", "“": '"', "”": '"', "‘": "'", "’": "'"}
    )  # add custom trans here, to address oov

    def is_chinese(c):
        return (
            "\u3100" <= c <= "\u9fff"  # common chinese characters
        )

    for text in text_list:
        char_list = []
        text = text.translate(custom_trans)
        for seg in jieba.cut(text):
            seg_byte_len = len(bytes(seg, "UTF-8"))
            if seg_byte_len == len(seg):  # if pure alphabets and symbols
                if char_list and seg_byte_len > 1 and char_list[-1] not in " :'\"":
                    char_list.append(" ")
                char_list.extend(seg)
            elif polyphone and seg_byte_len == 3 * len(seg):  # if pure east asian characters
                seg_ = lazy_pinyin(seg, style=Style.TONE3, tone_sandhi=True)
                for i, c in enumerate(seg):
                    if is_chinese(c):
                        char_list.append(" ")
                    char_list.append(seg_[i])
            else:  # if mixed characters, alphabets and symbols
                for c in seg:
                    if ord(c) < 256:
                        char_list.extend(c)
                    elif is_chinese(c):
                        char_list.append(" ")
                        char_list.extend(lazy_pinyin(c, style=Style.TONE3, tone_sandhi=True))
                    else:
                        char_list.append(c)
        final_text_list.append(char_list)

    return final_text_list


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


# get the empirically pruned step for sampling


def get_epss_timesteps(n, device, dtype):
    dt = 1 / 32
    predefined_timesteps = {
        5: [0, 2, 4, 8, 16, 32],
        6: [0, 2, 4, 6, 8, 16, 32],
        7: [0, 2, 4, 6, 8, 16, 24, 32],
        10: [0, 2, 4, 6, 8, 12, 16, 20, 24, 28, 32],
        12: [0, 2, 4, 6, 8, 10, 12, 14, 16, 20, 24, 28, 32],
        16: [0, 1, 2, 3, 4, 5, 6, 7, 8, 10, 12, 14, 16, 20, 24, 28, 32],
    }
    t = predefined_timesteps.get(n, [])
    if not t:
        return torch.linspace(0, 1, n + 1, device=device, dtype=dtype)
    return dt * torch.tensor(t, device=device, dtype=dtype)

def prefix_text_padding(
    text: int["b nt"],
    lens: int["b"] | None,
    prompt_lens: int["b"] | None,
    prefix_token_id: int = -1,
    anchor_token_ids: int["na"] | None = None,
) -> int["b nt"]:
    lens = lens.to(device=text.device, dtype=torch.long)
    prompt_lens = prompt_lens.to(device=text.device, dtype=torch.long)
    if exists(anchor_token_ids):
        anchor_token_ids = anchor_token_ids.to(device=text.device, dtype=text.dtype)
    if torch.any(prompt_lens >= lens):
        raise ValueError("prompt_lens must be strictly less than lens")

    padded_text = []
    for i in range(text.shape[0]):
        sample_text = text[i]
        target_text = sample_text[sample_text != -1]
        prompt_len = int(prompt_lens[i].item())
        total_len = int(lens[i].item())
        target_mel_len = total_len - prompt_len

        if prompt_len <= 0 or target_text.numel() == 0:
            padded_text.append(target_text)
            continue

        num_prefix = round(prompt_len / target_mel_len * int(target_text.numel()))
        if num_prefix <= 0:
            padded_text.append(target_text)
            continue

        prefix = torch.full((num_prefix,), prefix_token_id, device=text.device, dtype=text.dtype)
        if exists(anchor_token_ids):
            padded_text.append(torch.cat((prefix, anchor_token_ids, target_text), dim=0))
        else:
            padded_text.append(torch.cat((prefix, target_text), dim=0))

    return pad_sequence(padded_text, batch_first=True, padding_value=-1)


def build_prefixed_language_ids(
    text: torch.Tensor,
    total_lens: torch.Tensor,
    prompt_lens: torch.Tensor,
    language_ids: torch.Tensor,
    anchor_token_ids: torch.Tensor | None = None,
    unknown_lang_id: int | None = None,
) -> torch.Tensor:
    language_ids = language_ids.to(device=text.device, dtype=torch.long)
    total_lens = total_lens.to(device=text.device, dtype=torch.long)
    prompt_lens = prompt_lens.to(device=text.device, dtype=torch.long)
    anchor_len = 0 if anchor_token_ids is None else int(anchor_token_ids.numel())
    padded_language_ids = []

    for i in range(text.shape[0]):
        sample_text = text[i]
        target_text = sample_text[sample_text != -1]
        prompt_len = int(prompt_lens[i].item())
        total_len = int(total_lens[i].item())
        target_mel_len = total_len - prompt_len
        target_text_len = int(target_text.numel())
        target_lang = int(language_ids[i].item())

        if prompt_len <= 0 or target_text_len == 0:
            padded_language_ids.append(
                torch.full((target_text_len,), target_lang, device=text.device, dtype=torch.long)
            )
            continue

        if target_mel_len <= 0:
            raise ValueError("total_lens must be greater than prompt_lens")

        num_prefix = round(prompt_len / target_mel_len * target_text_len)
        if num_prefix <= 0:
            padded_language_ids.append(
                torch.full((target_text_len,), target_lang, device=text.device, dtype=torch.long)
            )
            continue

        prefix_lang = torch.full((num_prefix,), -1, device=text.device, dtype=torch.long)
        target_lang_ids = torch.full((target_text_len,), target_lang, device=text.device, dtype=torch.long)

        if anchor_len > 0:
            anchor_lang_ids = torch.full((anchor_len,), unknown_lang_id, device=text.device, dtype=torch.long)
            padded_language_ids.append(torch.cat((prefix_lang, anchor_lang_ids, target_lang_ids), dim=0))
        else:
            padded_language_ids.append(torch.cat((prefix_lang, target_lang_ids), dim=0))

    return pad_sequence(padded_language_ids, batch_first=True, padding_value=-1)


def build_prefixed_language_ids_tokenwise(
    text: torch.Tensor,
    total_lens: torch.Tensor,
    prompt_lens: torch.Tensor,
    language_ids: torch.Tensor,
    anchor_token_ids: torch.Tensor | None = None,
    unknown_lang_id: int | None = None,
) -> torch.Tensor:
    language_ids = language_ids.to(device=text.device, dtype=torch.long)
    total_lens = total_lens.to(device=text.device, dtype=torch.long)
    prompt_lens = prompt_lens.to(device=text.device, dtype=torch.long)
    anchor_len = 0 if anchor_token_ids is None else int(anchor_token_ids.numel())
    padded_language_ids = []

    for i in range(text.shape[0]):
        sample_text = text[i]
        target_text = sample_text[sample_text != -1]
        target_text_len = int(target_text.numel())

        sample_lang_ids = language_ids[i]
        sample_lang_ids = sample_lang_ids[sample_lang_ids != -1]
        if int(sample_lang_ids.numel()) != target_text_len:
            raise ValueError(
                f"Token-wise language ids length mismatch: got {int(sample_lang_ids.numel())}, expected {target_text_len}"
            )

        prompt_len = int(prompt_lens[i].item())
        total_len = int(total_lens[i].item())
        target_mel_len = total_len - prompt_len

        if prompt_len <= 0 or target_text_len == 0:
            padded_language_ids.append(sample_lang_ids)
            continue

        if target_mel_len <= 0:
            raise ValueError("total_lens must be greater than prompt_lens")

        num_prefix = round(prompt_len / target_mel_len * target_text_len)
        if num_prefix <= 0:
            padded_language_ids.append(sample_lang_ids)
            continue

        prefix_lang = torch.full((num_prefix,), -1, device=text.device, dtype=torch.long)

        if anchor_len > 0:
            anchor_lang_ids = torch.full((anchor_len,), unknown_lang_id, device=text.device, dtype=torch.long)
            padded_language_ids.append(torch.cat((prefix_lang, anchor_lang_ids, sample_lang_ids), dim=0))
        else:
            padded_language_ids.append(torch.cat((prefix_lang, sample_lang_ids), dim=0))

    return pad_sequence(padded_language_ids, batch_first=True, padding_value=-1)
