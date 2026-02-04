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
# 第一版本 按照_分词，不拆分音节，只拆分音节和声调(123等)
def str_to_list_ipa(phonemized: str) -> List[str]:
    fields = []
    # 先按_切分
    words = phonemized.split("_")
    for i, word in enumerate(words):
        if not word: continue
        processed_tokens = []
        # 再按音素分隔符|切分
        tokens = word.split("|")
        # 过滤空字符串、分割音素音调
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
            fields.append("_")
    return fields

# 第二版本，按照空格分词，拆分音节，按照char读词
def str_to_list_ipa_v2(phonemized: str) -> List[str]:
    fields = []
    
    words = phonemized.split(" ")
    
    for i, word in enumerate(words):
        if not word: continue
        processed_tokens = []
        # 再按竖线切分
        tokens = word.split("|")
        for token in tokens:
            if not token: continue
            # 处理带数字声调的情况
            match = re.match(r'^([^0-9]+)([0-9]+)$', token)
            if match:
                phoneme_part = match.group(1) # 'ai' 或 'u:'
                tone_part = match.group(2)    # '2'
                # 强制把音素部分拆成单个字符
                fields.extend(list(phoneme_part)) 
                # 添加声调
                fields.append(tone_part)
            else:
                # 处理不带数字的情况
                # 同样强制拆成单个字符
                fields.extend(list(token))
        if i < len(words) - 1:
            fields.append(" ")
    return fields

# 第三版本 按照空格分词，不拆分音节，只拆分音节和声调(123等)
def str_to_list_ipa_v3(phonemized: str) -> List[str]:
    fields = []
    # 先按_切分
    words = phonemized.split(" ")
    for i, word in enumerate(words):
        if not word: continue
        processed_tokens = []
        # 再按音素分隔符|切分
        tokens = word.split("|")
        # 过滤空字符串、分割音素音调
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


SHARED_SYMBOLS = set(" !%&',-.;:?@[]—…·")
TAGGED_SINGLE_SYMBOLS = {
    'ː', 'ʲ', '̃', 'ʰ', 'ˤ', 'ˠ', 'ˑ', '̆', '^', '`', '̯', '̪', '͡',
    '0', '1', '2', '3', '4', '5', '6', '7', '8', '9',
}
# 核心正则逻辑：
# 1. ([a-zA-Z\u00C0-\u02AF]+) : 匹配连续的字母块（如 iəɜ, bei, jing），这些块会整体打标
# 2. (.) : 匹配任意单个字符（处理标点或修饰符）
TOKEN_EXTRACTOR = re.compile(r"([a-zA-Z\u00C0-\u02AF]+)|(.)")
def str_to_list_ipa_v5(phonemized: str, lang: str) -> List[str]:
    suffix = f"_{lang}"
    fields = []
    # 按空格切词，保留空格位置
    words = phonemized.split(" ")
    for i, word in enumerate(words):
        if not word: continue
        # 内部按 | 切分
        for segment in word.split("|"):
            if not segment: continue
            # 使用正则迭代器，一次性抓取所有的块和符号
            for m in TOKEN_EXTRACTOR.finditer(segment):
                text_chunk = m.group(1) # 字母块
                symbol_chunk = m.group(2) # 单个符号
                if text_chunk:
                    fields.append(text_chunk + suffix)
                elif symbol_chunk:
                    # 判断是否属于共享符号
                    if symbol_chunk in SHARED_SYMBOLS:
                        fields.append(symbol_chunk)
                    else:
                        fields.append(symbol_chunk + suffix)
        # 恢复词间空格
        if i < len(words) - 1:
            fields.append(" ")          
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
        if tokenizer == "ipa":
            fields = str_to_list_ipa(ipa_string)
        elif tokenizer == "ipa_v2":
            fields = str_to_list_ipa_v2(ipa_string)
        elif tokenizer == "ipa_v3":
            fields = str_to_list_ipa_v3(ipa_string)
        elif tokenizer == "ipa_v5":
            fields = str_to_list_ipa_v5(ipa_string, lang=language_ids[i])
        else: 
            print(f"{tokenizer} is not a supported  version of ipa.")
        if fields:
            indices = [vocab_char_map.get(token, unk_idx) for token in fields]
            batch_indices.append(torch.tensor(indices, dtype=torch.long))
        
    padded_batch = pad_sequence(
        batch_indices, 
        batch_first=True, 
        padding_value=padding_value
    )
    
    return padded_batch





# Get tokenizer


def get_tokenizer(dataset_name, tokenizer: str = "pinyin"):
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
        tokenizer_path = os.path.join(files("f5_tts").joinpath("../../data"), f"{dataset_name}_{tokenizer}/vocab.txt")
        with open(tokenizer_path, "r", encoding="utf-8") as f:
            vocab_char_map = {}
            for i, char in enumerate(f):
                vocab_char_map[char[:-1]] = i
        vocab_size = len(vocab_char_map)
        assert vocab_char_map[" "] == 0, "make sure space is of idx 0 in vocab.txt, cuz 0 is used for unknown char"

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
