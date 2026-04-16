import re
import regex
from typing import List, Pattern, Union
import os
import string
import logging

from phonemizer.backend import EspeakBackend
from phonemizer.backend.espeak.language_switch import LanguageSwitch
from phonemizer.backend.espeak.words_mismatch import WordMismatch
from phonemizer.logger import get_logger
from phonemizer.punctuation import Punctuation
from phonemizer.separator import Separator
from x_voice.model.utils import str_to_list_ipa_v6

import jieba
from pypinyin import lazy_pinyin, Style

import pythainlp
from pythainlp.tokenize import word_tokenize
from pythainlp.transliterate import transliterate
from pythainlp.transliterate.ipa import trans_list

import pyopenjtalk
import g2pk

_KOREAN_G2P_MODEL = None
class G2pk:
    """On behalf of g2pk.G2p.

    g2pk.G2p isn't pickalable and it can't be copied to the other processes
    via multiprocessing module.
    As a workaround, g2pk.G2p is instantiated upon calling this class.

    """

    def __init__(
        self,
        descritive=False,
        group_vowels=False,
        to_syl=False,
        no_space=False,
        explicit_space=False,
        space_symbol="<space>",
    ):
        self.descritive = descritive
        self.group_vowels = group_vowels
        self.to_syl = to_syl
        self.no_space = no_space
        self.explicit_space = explicit_space
        self.space_symbol = space_symbol
        self.g2p = None

    def __call__(self, text) -> List[str]:
        global _KOREAN_G2P_MODEL
        if _KOREAN_G2P_MODEL is None:
            print(f"--- Process {os.getpid()} is loading the heavy Korean model... ---")
            _KOREAN_G2P_MODEL = g2pk.G2p()

        phones = list(
            _KOREAN_G2P_MODEL( # Reuse the global instance.
                text,
                descriptive=self.descritive,
                group_vowels=self.group_vowels,
                to_syl=self.to_syl,
            )
        )
        if self.no_space:
            # remove space which represents word serapater
            phones = list(filter(lambda s: s != " ", phones))

        if self.explicit_space:
            # replace space as explicit space symbol
            phones = list(map(lambda s: s if s != " " else self.space_symbol, phones))
        return phones



SYMBOLS_MAPPING = {
    "：": ":", "，": ",", "。": ".", "！": "!", "？": "?",
    "；": ";", "、": ",", "...": "…", "......": "…",
    "‘": "'", "’": "'", "（": "(", "）": ")", "～": "-", "_":"-",
    '"': "'", "«": "'", "»": "'", "”":"'", "“":"'" , "《":"[", "》":"]","「":"[","」":"]","【":"[","】":"]",
    "¿": "?",  "¡": "!","‧":"·",
}
REPLACE_SYMBOL_REGEX = re.compile("|".join(re.escape(p) for p in SYMBOLS_MAPPING.keys()))
IPA_NORMALIZATION_MAP = {
    # Normalize unstable symbols.
    "eɪɛ": "eɪ|ɛ", "zeɪɛ": "z|eɪ|ɛ", "teɪɛ": "t|eɪ|ɛ", "əeɪ": "ə|eɪ", "aɪɛ": "aɪ|ɛ", "taɪ": "t|aɪ", 
    "jap": "ja|p", "jud": "ju|d", "jaɛ": "ja|ɛ", "ʃja": "ʃ|ja", "jat": "ja|t", "ɑja": "ɑ|ja", "əlɹ": "əl|ɹ", "əlf": "əl|f", "oʊw": "oʊ|w", 
    "daʊ":"d|aʊ", "meɪ":"m|eɪ", "taʊ":"t|aʊ", "daɪ":"d|aɪ",
    "nɡ": "ŋ",        # Standalone nasal.
}

def convert_char_to_pinyin(text_list, polyphone=True):
    # Ensure jieba is initialized.
    if not hasattr(jieba, 'dt') or not jieba.dt.initialized:
        jieba.setLogLevel(20)
        jieba.initialize()

    final_text_list = []

    def is_chinese(c):
        return "\u3100" <= c <= "\u9fff"

    for text in text_list:
        text = REPLACE_SYMBOL_REGEX.sub(lambda x: SYMBOLS_MAPPING[x.group()], text)
        text = regex.sub(r"\p{C}|\p{Z}", " ", text)
        sentence_words = [] # Store processed words for the current sentence.
        
        for seg in jieba.cut(text):
            # seg is one segmented token.
            if not seg.strip():
                continue
                
            seg_byte_len = len(bytes(seg, "UTF-8"))
            current_word_parts = [] # Store sub-parts inside the word (pinyin or characters).
            
            # Case A: plain letters / numbers / symbols (English / numbers) -> split by character.
            # "Hello" -> ['H', 'e', 'l', 'l', 'o']
            if seg_byte_len == len(seg): 
                current_word_parts.extend(list(seg))
            
            # Case B: pure Chinese word -> convert to pinyin and keep syllables intact.
            # Example: a Chinese word becomes a list of pinyin syllables.
            elif polyphone and seg_byte_len == 3 * len(seg):
                # style=Style.TONE3 produces tokens like "bei3".
                seg_pinyin = lazy_pinyin(seg, style=Style.TONE3, tone_sandhi=True)
                current_word_parts.extend(seg_pinyin)
            
            # Case C: mixed text -> process character by character.
            # Example: mixed text is processed character by character.
            else:
                for c in seg:
                    if ord(c) < 256:
                        current_word_parts.append(c)
                    elif is_chinese(c):
                        # Convert a single Chinese character to pinyin.
                        p = lazy_pinyin(c, style=Style.TONE3, tone_sandhi=True)[0]
                        current_word_parts.append(p)
                    else:
                        current_word_parts.append(c)
            
            # Join phonemes inside a word with "|".
            if current_word_parts:
                sentence_words.append("|".join(current_word_parts))
        
        # Join words with spaces.
        final_text_list.append(" ".join(sentence_words))

    return final_text_list

class PhonemizeTextTokenizer:
    def __init__(
        self,
        language="en-us",
        backend="espeak",
        separator=Separator(word=" ", syllable="", phone="|"),
        preserve_punctuation=True,
        punctuation_marks: Union[str, Pattern] = Punctuation.default_marks(),
        with_stress: bool = True,
        tie: Union[bool, str] = False,
        language_switch: LanguageSwitch = "remove-flags",
        words_mismatch: WordMismatch = "ignore",
        mapping_map=IPA_NORMALIZATION_MAP,
        symbol_map=SYMBOLS_MAPPING,
    ) -> None:
        if language == 'ko':
            self.g2p = G2pk(no_space=False) 
        else:
            self.g2p = None
        if backend == "espeak":
            phonemizer = EspeakBackend(
                language,
                punctuation_marks=punctuation_marks,
                preserve_punctuation=preserve_punctuation,
                with_stress=with_stress,
                tie=tie,
                language_switch=language_switch,
                words_mismatch=words_mismatch,
                logger=logging.basicConfig(level=logging.ERROR)
            )
        else:
            raise NotImplementedError(f"{backend}")

        self.backend = phonemizer
        self.separator = separator
        self.language = language
        
        self.mapping = mapping_map
        sorted_keys = sorted(mapping_map.keys(), key=len, reverse=True)
        pattern_str = "|".join(map(re.escape, sorted_keys))
        self.pattern = re.compile(pattern_str)
        
        self.symbol_mapping = symbol_map
        self.symbol_pattern = re.compile("|".join(re.escape(p) for p in symbol_map.keys()))

    def clean_text(self, text):
        text = text.strip()
        text = regex.sub(r"\p{C}|\p{Z}", " ", text)
        text = self.symbol_pattern.sub(lambda x: self.symbol_mapping[x.group()], text)
        text = re.sub(r'([,.:;!?…—])', r' \1 ', text)
        text = re.sub(r'\s+', ' ', text)
        return text

    def post_clean_ipa(self, ipa_string: str, mapping: bool = False) -> str:
        ipa_string = re.sub(r'[\[\](){}]', ' ', ipa_string)
        ipa_string = regex.sub(r",(?!\s)", 'ˌ', ipa_string) # Replace commas without trailing spaces with the secondary stress marker.
        ipa_string = re.sub(r'[&ㄜでかすπ%@]', '', ipa_string) # Remove garbage annotation characters.
        if mapping:
            ipa_string = self.pattern.sub(lambda match: self.mapping[match.group(0)], ipa_string)
        ipa_string = re.sub(r'\|+', '|', ipa_string) # || -> |
        ipa_string = re.sub(r' \| ', ' ', ipa_string) # Do not keep "|" around spaces.
        ipa_string = re.sub(r'^\||\|$', '', ipa_string) # Strip leading and trailing "|".
        ipa_string = re.sub(r'\s+', ' ', ipa_string) # Collapse repeated spaces.
        return ipa_string.strip()

    def __call__(self, text, strip=True) -> str:
        if isinstance(text, str):
            text = [self.clean_text(text)]
        elif isinstance(text, list):
            text = [self.clean_text(t) for t in text]
        else:
            return ""

        input_text = text[0]

        # Chinese
        if self.language == 'cmn' or self.language == 'zh':
            res = convert_char_to_pinyin([input_text])
            cleaned_ipa = self.post_clean_ipa(res[0], mapping=False)

        # Thai
        elif self.language == 'th':
            words = word_tokenize(input_text, engine="newmm")
            ipa_words=[]
            for word in words:
                # Skip pure whitespace or empty strings.
                if not word.strip():
                    continue
                #raw_ipa = transliterate(word, engine="ipa")
                raw_ipa = trans_list(word)
                if raw_ipa:
                    formatted_ipa = "|".join(raw_ipa)
                    ipa_words.append(formatted_ipa)
            cleaned_ipa = self.post_clean_ipa(" ".join(ipa_words), mapping=True)

        elif self.language == 'ja':
            raw_phones = pyopenjtalk.g2p(input_text)
            phones_list = raw_phones.split(" ")
            processed_phones = []
            for p in phones_list:
                # Filter pyopenjtalk silence/pause markers or convert them into punctuation.
                processed_phones.append(p.lower())
            # Join phoneme units with "|".
            res_ipa = ""
            if processed_phones:
                # Keep the final result free of repeated commas.
                res_ipa = "|".join(processed_phones)
                #res_ipa = re.sub(r'(\|?,\|?)+', '|', res_ipa).strip('|')
            cleaned_ipa = self.post_clean_ipa(res_ipa, mapping=True)
        # Korean: first convert text into pronunciation with g2pK.
        elif self.language == 'ko':
            assert self.g2p is not None 
            pronounced_hangul_list = self.g2p(input_text)
            pronounced_hangul = "".join(pronounced_hangul_list)
            phonemized = self.backend.phonemize([pronounced_hangul], separator=self.separator, strip=strip, njobs=1)
            cleaned_ipa = self.post_clean_ipa(phonemized[0], mapping=True)
            # print(input_text)
            # print(pronounced_hangul)
        # Other languages: fall back to generic eSpeak phonemization.
        else:
            phonemized = self.backend.phonemize([input_text], separator=self.separator, strip=strip, njobs=1)
            cleaned_ipa = self.post_clean_ipa(phonemized[0], mapping=True)
        return cleaned_ipa

def tokenize_text(tokenizer: PhonemizeTextTokenizer, text: str) -> str:
    phonemes = tokenizer([text.strip()])
    return phonemes

def get_ipa_id(in_language: str) -> str:
    LANG_MAP = {
        "zh": "cmn", "en": "en-us", "fr": "fr-fr", 
    }
    return LANG_MAP.get(in_language, in_language)

def run_test():
    TEST_CASES = [
        # Add ad-hoc multilingual examples here when debugging tokenizer behavior.
    ]
    
    print(f"{'Lang':<5} | {'Original Text':<25}")
    print("-" * 80)
    for lang_code, text in TEST_CASES:
        espeak_code = get_ipa_id(lang_code)
        try:
            tokenizer = PhonemizeTextTokenizer(language=espeak_code)
            ipa_string = tokenize_text(tokenizer, text)
            tokens = str_to_list_ipa_v6(ipa_string)
            
            print(f"[{lang_code}] Input: {text}")
            print(f"     -> IPA String: {ipa_string}")
            print(f"     -> Token List: {tokens}")
            print("-" * 80)
            
        except Exception as e:
            print(f"[{lang_code}] Error: {e}")
            import traceback
            traceback.print_exc()

if __name__ == "__main__":
    run_test()
# python src/x_voice/train/datasets/ipa_v6_tokenizer.py
