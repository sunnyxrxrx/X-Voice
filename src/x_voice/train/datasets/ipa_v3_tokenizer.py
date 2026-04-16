import re
import regex
from typing import List, Pattern, Union
import os
import string

from phonemizer.backend import EspeakBackend
from phonemizer.backend.espeak.language_switch import LanguageSwitch
from phonemizer.backend.espeak.words_mismatch import WordMismatch
from phonemizer.punctuation import Punctuation
from phonemizer.separator import Separator
from x_voice.model.utils import str_to_list_ipa_v3 

import jieba
from pypinyin import lazy_pinyin, Style

import pythainlp
from pythainlp.tokenize import word_tokenize
from pythainlp.transliterate import transliterate



SYMBOLS_MAPPING = {
    "：": ":", "，": ",", "。": ".", "！": "!", "？": "?",
    "；": ";", "、": ",", "...": "…", "......": "…",
    "‘": "'", "’": "'", "（": "(", "）": ")", "～": "-", "_":"-",
    '"': "'", "«": "'", "»": "'", "”":"'", "“":"'" , "《":"[", "》":"]","「":"[","」":"]","【":"[","】":"]",
    "¿": "?",  "¡": "!","‧":"·",
}
REPLACE_SYMBOL_REGEX = re.compile("|".join(re.escape(p) for p in SYMBOLS_MAPPING.keys()))

def get_ipa_id(in_language: str) -> str:
    LANG_MAP = {
        "zh": "cmn", "en": "en-us", "fr": "fr-fr",
    }
    return LANG_MAP.get(in_language, in_language)

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
        with_stress: bool = False,
        tie: Union[bool, str] = False,
        language_switch: LanguageSwitch = "remove-flags",
        words_mismatch: WordMismatch = "ignore",
    ) -> None:
        if backend == "espeak":
            phonemizer = EspeakBackend(
                language,
                punctuation_marks=punctuation_marks,
                preserve_punctuation=preserve_punctuation,
                with_stress=with_stress,
                tie=tie,
                language_switch=language_switch,
                words_mismatch=words_mismatch,
            )
        else:
            raise NotImplementedError(f"{backend}")

        self.backend = phonemizer
        self.separator = separator
        self.language = language

    def clean_text(self, text):
        text = text.strip()
        text = regex.sub(r"\p{C}|\p{Z}", " ", text)
        text = REPLACE_SYMBOL_REGEX.sub(lambda x: SYMBOLS_MAPPING[x.group()], text)
        text = re.sub(r'([,.:;!?…—])', r' \1 ', text)
        text = re.sub(r'\s+', ' ', text)
        return text

    def post_clean_ipa(self, ipa_string: str) -> str:
        ipa_string = re.sub(r'[\[\](){}]', '', ipa_string)
        # Clean up duplicated separators.
        ipa_string = regex.sub(r",(?!\s)", "ˌ", ipa_string)
        ipa_string = re.sub(r'\|+', '|', ipa_string) # || -> |
        ipa_string = re.sub(r' \| ', ' ', ipa_string) # Do not keep "|" around spaces.
        ipa_string = re.sub(r'^\||\|$', '', ipa_string) # Strip leading and trailing "|".
        ipa_string = re.sub(r'\s+', ' ', ipa_string)
        
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
            return res[0]

        # Thai
        elif self.language == 'th':
            words = word_tokenize(input_text, engine="newmm")
            ipa_words=[]
            for word in words:
                # Skip pure whitespace or empty strings.
                if not word.strip():
                    continue
                raw_ipa = transliterate(word, engine="ipa")
                # list(raw_ipa) splits the string into single characters, including tone markers.
                # "pʰɯ̂ːt" -> ['p', 'ʰ', 'ɯ', '̂', 'ː', 't']
                if raw_ipa:
                    raw_ipa = raw_ipa.replace("͡", "")
                    raw_ipa = raw_ipa.replace("-", "")
                    formatted_ipa = "|".join(list(raw_ipa))
                    ipa_words.append(formatted_ipa)
            return " ".join(ipa_words)
        # Other languages: fall back to generic eSpeak phonemization.
        phonemized = self.backend.phonemize([input_text], separator=self.separator, strip=strip, njobs=1)
        cleaned_ipa = self.post_clean_ipa(phonemized[0])
        return cleaned_ipa

def tokenize_text(tokenizer: PhonemizeTextTokenizer, text: str) -> str:
    phonemes = tokenizer([text.strip()])
    return phonemes

def run_test():
    TEST_CASES = [
    ("vi", "là odybulin vào năm hai nghìn lẻ ba công nghệ này"),
    ("vi", "người ta đưa tôi đi bs đông y và tây y"),
    ("vi", "gần một nghìn bốn trăm tỷ đồng theo quy định khu công"),
    ("vi", "được cái là là lòng đường là sáum vỉa hè"),
    ("vi", "du học đông âu vào những thập niên tám mươi"),
    ("vi", "youtube được nhiều các bạn quan tâm và"),
    ("vi", "hơn năm mươi tuổi rồi đồng nghiệp nữ lòng dạ"),
    ("vi", "năm trăm đức phật và cũng từng được nghe pháp"),
    ("vi", "chín giờ đến mười một giờ ngày mùng sáu tháng năm năm"),
    ("vi", "cũng quay về ngày mười hai tháng mười hai khi vào"),
    ]
    
    print(f"{'Lang':<5} | {'Original Text':<25}")
    print("-" * 80)
    for lang_code, text in TEST_CASES:
        espeak_code = get_ipa_id(lang_code)
        try:
            tokenizer = PhonemizeTextTokenizer(language=espeak_code)
            ipa_string = tokenize_text(tokenizer, text)
            tokens = str_to_list_ipa_v3(ipa_string)
            
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
# python src/x_voice/train/datasets/ipa_v3_tokenizer.py
