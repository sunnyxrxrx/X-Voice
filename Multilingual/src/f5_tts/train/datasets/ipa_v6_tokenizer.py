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
from f5_tts.model.utils import str_to_list_ipa_v6

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
            print(f"--- 进程 {os.getpid()} 正在加载重型韩语模型... ---")
            _KOREAN_G2P_MODEL = g2pk.G2p()

        phones = list(
            _KOREAN_G2P_MODEL( # 使用全局实例
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
    # 符号统一化
    "eɪɛ": "eɪ|ɛ", "zeɪɛ": "z|eɪ|ɛ", "teɪɛ": "t|eɪ|ɛ", "əeɪ": "ə|eɪ", "aɪɛ": "aɪ|ɛ", "taɪ": "t|aɪ", 
    "jap": "ja|p", "jud": "ju|d", "jaɛ": "ja|ɛ", "ʃja": "ʃ|ja", "jat": "ja|t", "ɑja": "ɑ|ja", "əlɹ": "əl|ɹ", "əlf": "əl|f", "oʊw": "oʊ|w", 
    "daʊ":"d|aʊ", "meɪ":"m|eɪ", "taʊ":"t|aʊ", "daɪ":"d|aɪ",
    "nɡ": "ŋ",        # 独立的鼻音
}

def convert_char_to_pinyin(text_list, polyphone=True):
    # 确保 jieba 初始化
    if not hasattr(jieba, 'dt') or not jieba.dt.initialized:
        jieba.setLogLevel(20)
        jieba.initialize()

    final_text_list = []

    def is_chinese(c):
        return "\u3100" <= c <= "\u9fff"

    for text in text_list:
        text = REPLACE_SYMBOL_REGEX.sub(lambda x: SYMBOLS_MAPPING[x.group()], text)
        text = regex.sub(r"\p{C}|\p{Z}", " ", text)
        sentence_words = [] # 存放当前句子中处理好的词
        
        for seg in jieba.cut(text):
            # seg 是一个分词结果
            if not seg.strip():
                continue
                
            seg_byte_len = len(bytes(seg, "UTF-8"))
            current_word_parts = [] # 存放词内部的成分 (拼音或字符)
            
            # Case A: 纯字母/数字/符号 (English/Numbers) -> 按字符拆分
            # "Hello" -> ['H', 'e', 'l', 'l', 'o']
            if seg_byte_len == len(seg): 
                current_word_parts.extend(list(seg))
            
            # Case B: 纯中文 (Chinese Word) -> 转拼音，保持音节整体
            # "北京" -> ['bei3', 'jing1']
            elif polyphone and seg_byte_len == 3 * len(seg):
                # style=Style.TONE3 生成 "bei3" 格式
                seg_pinyin = lazy_pinyin(seg, style=Style.TONE3, tone_sandhi=True)
                current_word_parts.extend(seg_pinyin)
            
            # Case C: 混合文本 (Mixed) -> 逐字处理
            # "U盘" -> ['U', 'pan2']
            else:
                for c in seg:
                    if ord(c) < 256:
                        current_word_parts.append(c)
                    elif is_chinese(c):
                        # 单字转拼音
                        p = lazy_pinyin(c, style=Style.TONE3, tone_sandhi=True)[0]
                        current_word_parts.append(p)
                    else:
                        current_word_parts.append(c)
            
            # 词内部用 | 连接
            if current_word_parts:
                sentence_words.append("|".join(current_word_parts))
        
        # 词之间用空格连接
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
        ipa_string = regex.sub(r",(?!\s)", 'ˌ', ipa_string) # 后面没有空格的逗号，替换为次重音符号
        ipa_string = re.sub(r'[&ㄜでかすπ%@]', '', ipa_string) # 删除标注中的垃圾字符
        if mapping:
            ipa_string = self.pattern.sub(lambda match: self.mapping[match.group(0)], ipa_string)
        ipa_string = re.sub(r'\|+', '|', ipa_string) # || -> |
        ipa_string = re.sub(r' \| ', ' ', ipa_string) # 空格两边不需要 |
        ipa_string = re.sub(r'^\||\|$', '', ipa_string) # 去除首尾 |
        ipa_string = re.sub(r'\s+', ' ', ipa_string) # 多个连续空格变为一个
        return ipa_string.strip()

    def __call__(self, text, strip=True) -> str:
        if isinstance(text, str):
            text = [self.clean_text(text)]
        elif isinstance(text, list):
            text = [self.clean_text(t) for t in text]
        else:
            return ""

        input_text = text[0]

        # 中文
        if self.language == 'cmn' or self.language == 'zh':
            res = convert_char_to_pinyin([input_text])
            cleaned_ipa = self.post_clean_ipa(res[0], mapping=False)

        # 泰语
        elif self.language == 'th':
            words = word_tokenize(input_text, engine="newmm")
            ipa_words=[]
            for word in words:
                # 过滤掉纯空格或空字符串
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
                # 过滤掉 pyopenjtalk 的静音/停顿标识符，或者转换成标点
                processed_phones.append(p.lower())
            # 用 "|" 将音素单元连接起来
            res_ipa = ""
            if processed_phones:
                # 过滤掉连续的逗号，保持结果整洁
                res_ipa = "|".join(processed_phones)
                #res_ipa = re.sub(r'(\|?,\|?)+', '|', res_ipa).strip('|')
            cleaned_ipa = self.post_clean_ipa(res_ipa, mapping=True)
        # 韩语，先用g2pK转为发音形式的文字
        elif self.language == 'ko':
            assert self.g2p is not None 
            pronounced_hangul_list = self.g2p(input_text)
            pronounced_hangul = "".join(pronounced_hangul_list)
            phonemized = self.backend.phonemize([pronounced_hangul], separator=self.separator, strip=strip, njobs=1)
            cleaned_ipa = self.post_clean_ipa(phonemized[0], mapping=True)
            # print(input_text)
            # print(pronounced_hangul)
        # 其他语言：通用eSpeak
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
    # ("el",". Πώς δεν είν' αστείο, βέβαια είναι αστείο· πάμε λοιπόν!"),
    # ("ko","안경"),
    # ("th","องค์กรเพื่อเสรีภาพของสัตว์และ(ราชสมาคมเพื่อการป้องกันการท)ารุณกรรมสัตว์  เรียกร้องอีกครั้งให้มีการบังคับติดตั้งกล้องวงจรปิดในโรงฆ่าสัตว์ทุกแห่งในออสเตรเลีย"),
    # ("zh", "我爱北京[天安门]，今天的天气真不错，气温大概是——二十五度。"), # Chinese (测试: 拼音/数字/声调)
    # ("de", "die dazu auffordert ein angebot bzw vorschläge"), 
    # ("vi", "“Ai tuyên truyền,  gieo rắc mê tín, tà tín ,tà kiến”."), # Vietnamese (测试: 声调符号/空格)
    # ("ja", "。。。 こんにちは、今日は天気がとても良いです。漢字とカタカナ、ひらがなが混ざっています。"), # Japanese (测试: 假名/汉字混合/长音/促音/拗音)
    # ("ja", "東京タワーへ行こう！コーヒーを飲みながら話そう。"), # Japanese (测试: 促音ッ/长音ー/感叹符/片假名外来语)
    # ("ru", "Привет, как дела? Сегодня очень хороша́я погода!"), # Russian (测试: 重音符号/软音符号/感叹符/表情符号)
    # ("ru", "Москва́ – столица Росси́и. В ча́шке чай с мёдом и лимо́ном."), # Russian (测试: 软音符号ь/硬音符号ъ/连字符/重音/空格)
    # ("ru", "Же́лтый ле́тний дождь шёл целы́ми дня́ми, мокро́ и хо́лодно."), # Russian (测试: 复杂辅音组合/й音/长句/逗号分隔)
    # ("ko", "안녕하세요, 이것은 테스트입니다. 오늘 날씨가 정말 좋네요."), # Korean (测试: 谚文/音变)
    # ("ko", "안녕하세요, 이거슨 테스트임니다. 오늘 랄씨가 정말 존네요.")
    # ("id", "Saya suka makan nasi goreng dan sate ayam di malam hari."), # Indonesian (测试: 简单拉丁/无声调)
    # ("en","the mmrs can pick up"),
    # ("en", "I have $20 in my pocket,daddy I felt good."), # English (测试: NSW正则化/重音)
    # ("de", "Über den sieben Bergen, bei den sieben Zwergen, ist es wunderschön."), # German (测试: 变音符号 äöü/复合词)
    # ("fr", "C'est la vie, mon ami. J'aime manger des croissants le matin."), # French (测试: 连读/省略符)
    # ("es", "El perro corre rápidamente por el parque verde."), # Spanish (测试: 颤音 R/重音)
    # ("it", "La pizza napoletana è famosa in tutto il mondo."), # Italian (测试: 节奏/双辅音)
    # ("pt", "introduzidas em 2004 e 2015,"), # Portuguese (测试: 鼻音/变音)
    # ("nl", "Hij fietst elke dag naar zijn werk in Amsterdam."), # Dutch (测试: 双元音/G发音)
    # ("en", "I felt cold."), # 错误 Token: ['lt', 'ld']
    # ("en", "Hello,world!This is-a test."), # 预期: [..., 'o', ' ', ',', ' ', 'w', ...] (必须有空格!)
    # ("sv", "Jag älskar att fika med mina vänner på helgerna."), # Swedish (测试: 音调重音/åäö)
    # ("da", "Rødgrød med fløde er en traditionel dansk dessert."), # Danish (测试: 软D/øæå/发音吞音)
    # ("fi", "Suomi on tuhansien järvien maa, ja sauna on tärkeä."), # Finnish (测试: 双元音/双辅音/元音和谐)
    # ("et", "Tere hommikust! Eestis on palju ilusaid metsi."), # Estonian (测试: 长短音/与芬兰语对比)
    # ("pl", "W Szczebrzeszynie chrząszcz brzmi w trzcinie."), # Polish (测试: 极难辅音簇/cz/sz)
    # ("cs", "Příliš žluťoučký kůň úpěl ďábelské ódy."), # Czech (测试:  ř/复杂的变音符号)
    # ("sk", "Dnes je pekný deň na prechádzku v horách."), # Slovak (测试: 与捷克语对比/长音)
    # ("hu", "Az élet szép, de néha nehéz döntéseket kell hozni."), # Hungarian (测试: 元音和谐/长双元音)
    # ("ro", "Bună ziua, ce mai faci? Vremea este foarte frumoasă."), # Romanian (测试: ̆/î/â 特殊字符)
    # ("bg", "Здравейте, как сте? Днес е прекрасен ден за разходка."), # Bulgarian (测试: 西里尔字母转写)
    # ("hr", "Dobar dan, kako ste? Danas je lijep dan."), # Croatian (测试: 塞尔维亚-克罗地亚语系)
    # ("sl", "Ljubljana je zelo lepo in zeleno mesto."), # Slovenian (测试: 双数/重音)
    # ("lt", "Labas rytas, Lietuva yra graži šalis."), # Lithuanian (测试: 音调重音/šžč)
    # ("lv", "Sveiki, kā jums klājas? Šodien ir jauka diena."), # Latvian (测试: 长音符号/šž)
    # ("el", "Καλημέρα, τι κάνετε; Ο καιρός είναι fantastikós."), # Greek (测试: 希腊字母转写/重音)
    # ("mt", "Il-bniedem li jaf jitkellem tajjeb għandu vantaġġ kbir.") # Maltese (测试: 闪含语系特征/ħ/għ)
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
# python src/f5_tts/train/datasets/ipa_v6_tokenizer.py