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
from f5_tts.model.utils import str_to_list_ipa_v2

import jieba
from pypinyin import lazy_pinyin, Style

import pythainlp
from pythainlp.tokenize import word_tokenize
from pythainlp.transliterate import transliterate


SYMBOLS_MAPPING = {
    "：": ":", "，": ",", "。": ".", "！": "!", "？": "?",
    "；": ";", "、": ",", "...": "…", "......": "…",
    "‘": "'", "’": "'", "（": "(", "）": ")", "～": "-", "-": "-", "_":"-",
    '"': "'", "«": "'", "»": "'", "”":"'", "“":"'" , "《":"[", "》":"]","「":"[","」":"]","【":"[","】":"]",
}
REPLACE_SYMBOL_REGEX = re.compile("|".join(re.escape(p) for p in SYMBOLS_MAPPING.keys()))

def get_ipa_id(in_language: str) -> str:
    LANG_MAP = {
        "zh": "cmn", "en": "en-us", "fr": "fr-fr",
    }
    return LANG_MAP.get(in_language, in_language)


def convert_char_to_pinyin(text_list, polyphone=True):
    # 确保 jieba 初始化
    if not hasattr(jieba, 'dt') or not jieba.dt.initialized:
        jieba.setLogLevel(20)
        jieba.initialize()

    final_text_list = []
    # 标点映射
    #custom_trans = str.maketrans({";": ",", "“": '"', "”": '"', "‘": "'", "’": "'"})

    def is_chinese(c):
        return "\u3100" <= c <= "\u9fff"

    for text in text_list:
        text = REPLACE_SYMBOL_REGEX.sub(lambda x: SYMBOLS_MAPPING[x.group()], text)
        text = regex.sub(r"\p{C}|\p{Z}", " ", text)
        sentence_words = [] # 存放这一句里的所有词
        
        # 按词处理，而不是按字处理
        for seg in jieba.cut(text):
            # seg 是一个词，例如 "北京" 或 "20" 或 ","
            seg_byte_len = len(bytes(seg, "UTF-8"))
            current_word_tokens = [] # 存放当前这个词内部的音素
            
            # 纯字母/数字/符号
            if seg_byte_len == len(seg): 
                # 把字母拆开，用 | 连接
                # 例如 "20" -> "2|0"
                # 如果 seg 是空格，要保留吗？通常 jieba 会把空格切出来
                if seg.strip():
                    current_word_tokens.append("|".join(list(seg)))
            
            # 纯中文
            elif polyphone and seg_byte_len == 3 * len(seg):
                # 转拼音: ['bei3', 'jing1']
                seg_pinyin = lazy_pinyin(seg, style=Style.TONE3, tone_sandhi=True)
                
                # 把每个字的拼音拆开，并用 | 连接
                for p in seg_pinyin:
                    current_word_tokens.append("|".join(list(p)))
                # 现在的 current_word_tokens 是 ['b|e|i|3', 'j|i|n|g|1']
                # 我们需要在词内部，把这两个字也用 | 连起来！
                # 变成: "b|e|i|3|j|i|n|g|1"
            # 3. 混合 (Mixed)
            else:
                for c in seg:
                    if ord(c) < 256:
                        current_word_tokens.append(c)
                    elif is_chinese(c):
                        p = lazy_pinyin(c, style=Style.TONE3, tone_sandhi=True)[0]
                        current_word_tokens.append("|".join(list(p)))
                    else:
                        current_word_tokens.append(c)
            
            # 如果这个词解析出了内容
            if current_word_tokens:
                # 把词内部的音素全部用 | 连起来
                # 例子: Word="北京" -> tokens=["b|e|i|3", "j|i|n|g|1"] -> joined="b|e|i|3|j|i|n|g|1"
                full_word_str = "|".join(current_word_tokens)
                sentence_words.append(full_word_str)
        
        # === 最后：词与词之间用空格连接 ===
        # 例子: "我" "爱" "北京" -> "w|o|3 a|i|4 b|e|i|3|j|i|n|g|1"
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
        text = re.sub(r'([,.:;!?…])', r' \1 ', text)
        text = re.sub(r'\s+', ' ', text)
        return text

    def post_clean_ipa(self, ipa_string: str) -> str:
        ipa_string = re.sub(r'[\[\](){}]', '', ipa_string)
        # 清理多余的分隔符
        ipa_string = re.sub(r'\|+', '|', ipa_string) # || -> |
        ipa_string = re.sub(r' \| ', ' ', ipa_string) # 空格两边不需要 |
        ipa_string = re.sub(r'^\||\|$', '', ipa_string) # 去除首尾 |
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

        # === Case 1: 中文 (Pinyin + |) ===
        if self.language == 'cmn' or self.language == 'zh':
            res = convert_char_to_pinyin([input_text])
            return res[0]

        # === Case 2: 泰语 (加空格 + |) ===
        elif self.language == 'th':
            words = word_tokenize(input_text, engine="newmm")
            ipa_words=[]
            for word in words:
                # 过滤掉纯空格或空字符串
                if not word.strip():
                    continue
                raw_ipa = transliterate(word, engine="ipa")
                # list(raw_ipa) 会把字符串拆成单个字符，包括声调符号
                # "pʰɯ̂ːt" -> ['p', 'ʰ', 'ɯ', '̂', 'ː', 't']
                # 用 | 连接 -> "p|ʰ|ɯ|̂|ː|t"
                if raw_ipa:
                    raw_ipa = raw_ipa.replace("͡", "")
                    raw_ipa = raw_ipa.replace("-", "")
                    formatted_ipa = "|".join(list(raw_ipa))
                    ipa_words.append(formatted_ipa)
            return " ".join(ipa_words)
        # === Case 3: 通用 eSpeak ===
        phonemized = self.backend.phonemize([input_text], separator=self.separator, strip=strip, njobs=1)
        cleaned_ipa = self.post_clean_ipa(phonemized[0])
        return cleaned_ipa

def tokenize_text(tokenizer: PhonemizeTextTokenizer, text: str) -> str:
    phonemes = tokenizer([text.strip()])
    return phonemes

def run_test():
    TEST_CASES = [
    ("zh", "我爱北京天安门，今天的天气真不错，气温大概是二十五度。"), # Chinese (测试: 拼音/数字/声调)
    ("th", "พืชผลิตก๊าซออกซิเจนที่มนุษย์ใช้หายใจและรับเอาคาร์บอนไดออกไซด์ที่มนุษย์ขับออกมา"), # Thai (测试: 无空格长句/分词/声调)
    ("vi", "Hiển nhiên, nếu bạn biết một ngôn ngữ La Mã, bạn sẽ dễ dàng học Tiếng Bồ Đào Nha."), # Vietnamese (测试: 声调符号/空格)
    ("ko", "안녕하세요, 이것은 테스트입니다. 오늘 날씨가 정말 좋네요."), # Korean (测试: 谚文/音变)
    ("id", "Saya suka makan nasi goreng dan sate ayam di malam hari."), # Indonesian (测试: 简单拉丁/无声调)

    # === 欧洲主要语言 (各语族代表) ===
    ("en", "I have $20 in my pocket, and I plan to buy 3 apples."), # English (测试: NSW正则化/重音)
    ("de", "Über den sieben Bergen, bei den sieben Zwergen, ist es wunderschön."), # German (测试: 变音符号 äöü/复合词)
    ("fr", "C'est la vie, mon ami. J'aime manger des croissants le matin."), # French (测试: 连读/省略符)
    ("es", "El perro corre rápidamente por el parque verde."), # Spanish (测试: 颤音 R/重音)
    ("it", "La pizza napoletana è famosa in tutto il mondo."), # Italian (测试: 节奏/双辅音)
    ("pt", "O tempo está muito bom para ir à praia hoje."), # Portuguese (测试: 鼻音/变音)
    ("nl", "Hij fietst elke dag naar zijn werk in Amsterdam."), # Dutch (测试: 双元音/G发音)
    
    # === 北欧与东欧语言 (特殊字符与辅音簇) ===
    ("sv", "Jag älskar att fika med mina vänner på helgerna."), # Swedish (测试: 音调重音/åäö)
    ("da", "Rødgrød med fløde er en traditionel dansk dessert."), # Danish (测试: 软D/øæå/发音吞音)
    ("fi", "Suomi on tuhansien järvien maa, ja sauna on tärkeä."), # Finnish (测试: 双元音/双辅音/元音和谐)
    ("et", "Tere hommikust! Eestis on palju ilusaid metsi."), # Estonian (测试: 长短音/与芬兰语对比)
    ("pl", "W Szczebrzeszynie chrząszcz brzmi w trzcinie."), # Polish (测试: 极难辅音簇/cz/sz)
    ("cs", "Příliš žluťoučký kůň úpěl ďábelské ódy."), # Czech (测试:  ř/复杂的变音符号)
    ("sk", "Dnes je pekný deň na prechádzku v horách."), # Slovak (测试: 与捷克语对比/长音)
    ("hu", "Az élet szép, de néha nehéz döntéseket kell hozni."), # Hungarian (测试: 元音和谐/长双元音)
    ("ro", "Bună ziua, ce mai faci? Vremea este foarte frumoasă."), # Romanian (测试: ̆/î/â 特殊字符)
    ("bg", "Здравейте, как сте? Днес е прекрасен ден за разходка."), # Bulgarian (测试: 西里尔字母转写)
    ("hr", "Dobar dan, kako ste? Danas je lijep dan."), # Croatian (测试: 塞尔维亚-克罗地亚语系)
    ("sl", "Ljubljana je zelo lepo in zeleno mesto."), # Slovenian (测试: 双数/重音)
    ("lt", "Labas rytas, Lietuva yra graži šalis."), # Lithuanian (测试: 音调重音/šžč)
    ("lv", "Sveiki, kā jums klājas? Šodien ir jauka diena."), # Latvian (测试: 长音符号/šž)
    ("el", "Καλημέρα, τι κάνετε; Ο καιρός είναι fantastikós."), # Greek (测试: 希腊字母转写/重音)
    ("mt", "Il-bniedem li jaf jitkellem tajjeb għandu vantaġġ kbir.") # Maltese (测试: 闪含语系特征/ħ/għ)
    ]
    
    print(f"{'Lang':<5} | {'Original Text':<25}")
    print("-" * 80)
    for lang_code, text in TEST_CASES:
        espeak_code = get_ipa_id(lang_code)
        try:
            tokenizer = PhonemizeTextTokenizer(language=espeak_code)
            ipa_string = tokenize_text(tokenizer, text)
            tokens = str_to_list_ipa_v2(ipa_string)
            
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
# python src/f5_tts/train/datasets/ipa_v2_tokenizer.py