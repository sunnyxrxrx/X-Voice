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
from f5_tts.model.utils import str_to_list_ipa

SYMBOLS_MAPPING = {
    "：": ":", "，": ",", "。": ".", "！": "!", "？": "?",
    "；": ";", "、": ",", "...": "…", "......": "…",
    "‘": "'", "’": "'", "（": "(", "）": ")", "～": "—", "-": "—", 
    '"': "'", "«": "'", "»": "'", "”":"'", "“":"'", "《":"[", "》":"]","「":"[","」":"]","【":"[","】":"]",
}
REPLACE_SYMBOL_REGEX = re.compile("|".join(re.escape(p) for p in SYMBOLS_MAPPING.keys()))

def get_ipa_id(in_language: str) -> str:
    LANG_MAP = {
        "zh": "cmn",
        "en": "en-us",
        "bg": "bg",
        "nl": "nl",
        "ko": "ko",
        "fr": "fr-fr"
    }
    return LANG_MAP.get(in_language, in_language)



    
class PhonemizeTextTokenizer:
    """Phonemize Text."""

    def __init__(
        self,
        language="en-us",
        backend="espeak",
        separator=Separator(word="_", syllable="-", phone="|"),
        preserve_punctuation=True,
        punctuation_marks: Union[str, Pattern] = Punctuation.default_marks(),  # default_marks = ';:,.!?¡¿—…"«»“”(){}[]'
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
    
    def clean_text(self, text):
        # Clean the text
        text = text.strip()
        text = regex.sub(r"\p{C}|\p{Z}", " ", text)  # ignore \n \t \v \f \r \u2028
        # 做符号的代换
        text = REPLACE_SYMBOL_REGEX.sub(lambda x: SYMBOLS_MAPPING[x.group()], text)
        # 给符号前后加空格
        text = re.sub(r'([,.:;!?…])', r' \1 ', text)
        #print(f"now text {text}")
        #合并多余空格
        text = re.sub(r'\s+', ' ', text)
        #print(text)
        return text


    
    def post_clean_ipa(self, ipa_string: str) -> str:
        #print(ipa_string)
        ipa_string = re.sub(r'[\[\](){}]', '', ipa_string)
        # 将所有连字符统一替换为单词分隔符_
        ipa_string = ipa_string.replace("—", "_")
        # 合并多余的分隔符
        ipa_string = re.sub(r'__+', '_', ipa_string)
        # 移除首位的三种符号
        ipa_string = ipa_string.strip('_| ')
        
        return ipa_string

    def __call__(self, text, strip=True) -> str:
        # try:
        if isinstance(text, str):
            text = [self.clean_text(text)]
        elif isinstance(text, list):
            text = [self.clean_text(t) for t in text]
        else:
            print("Only support text_list input and str input")
        phonemized = self.backend.phonemize(text, separator=self.separator, strip=strip, njobs=1)
        cleaned_ipa = [self.post_clean_ipa(ipa) for ipa in phonemized]
        return cleaned_ipa[0]


def tokenize_text(tokenizer: PhonemizeTextTokenizer, text: str) -> str:
    phonemes = tokenizer([text.strip()])
    return phonemes  # k2symbols


def run_test():
    TEST_CASES = [
    ("zh", "冥王星是柯伊伯带中较大。"),
    ("zh", "冥王姓是刻意博带中教大。"), 
    ("zh", "通过创新技术成为“该部门发放咨询的唯一渠道。"),
    ("th", "พืชผลิตก๊าซออกซิเจนที่มนุษย์ใช้หายใจและรับเอาคาร์บอนไดออกไซด์ที่มนุษย์ขับออกมา"),
    ("vi", "Hiển nhiên, nếu bạn biết một ngôn ngữ La Mã, bạn sẽ dễ dàng học Tiếng Bồ Đào Nha."),
    ("ko", "안녕하세요, 이것은 테스트입니다."),
    ("en", "I have 20$."),
]

    
    print(f"{'Lang':<5} | {'Original Text':<25} | {'IPA Output (Raw String)'}")
    print("-" * 80)
    for lang_code, text in TEST_CASES:
        espeak_code = get_ipa_id(lang_code)
        try:
            tokenizer = PhonemizeTextTokenizer(language=espeak_code)
            ipa_string = tokenize_text(tokenizer, text)
            tokens = str_to_list_ipa(ipa_string)
            
            print(f"[{lang_code}] Input: {text}")
            print(f"     -> IPA String: {ipa_string}")
            print(f"     -> Token List: {tokens}")
            
            print("-" * 80)
            
        except Exception as e:
            print(f"[{lang_code}] Error: {e}")
            print("-" * 80)

# if __name__ == "__main__":
#     run_test()
    
# python src/f5_tts/train/datasets/ipa_tokenizer.py