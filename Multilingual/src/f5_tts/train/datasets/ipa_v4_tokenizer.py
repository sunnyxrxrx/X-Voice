import re
import regex
from typing import List, Pattern, Union, Dict
import os
import string

from spellchecker import SpellChecker
from lingua import Language, LanguageDetectorBuilder

from phonemizer.backend import EspeakBackend
from phonemizer.backend.espeak.language_switch import LanguageSwitch
from phonemizer.backend.espeak.words_mismatch import WordMismatch
from phonemizer.punctuation import Punctuation
from phonemizer.separator import Separator
from f5_tts.model.utils import str_to_list_ipa_v3 

import jieba
from pypinyin import lazy_pinyin, Style
import pythainlp
from pythainlp.tokenize import word_tokenize
from pythainlp.transliterate import transliterate


# 1. 主语言代码 -> SpellChecker 代码
MAP_TO_SPELLCHECKER = {
    "en-us": "en", "en": "en",
    "de": "de", "fr-fr": "fr", "fr": "fr",
    "es": "es", "pt": "pt", "it": "it"
}

# 2. Lingua 语言对象 -> Espeak 代码
MAP_LINGUA_TO_ESPEAK = {
    Language.ENGLISH: "en-us",
    Language.GERMAN: "de",
    Language.FRENCH: "fr-fr",
    Language.SPANISH: "es",
    Language.PORTUGUESE: "pt",
    Language.ITALIAN: "it",
}

ALLOWED_SWITCHES = {
    "de": ["en-us", "fr-fr"],
    "es": ["en-us"],          # 西班牙语只借英语，绝不借葡语
    "pt": ["en-us"],          # 葡萄牙语只借英语
    "it": ["en-us"],          # 意大利语只借英语
    "fr-fr": ["en-us"],       # 法语只借英语
    "ru": ["en-us"],
}


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
        sentence_words = [] # 存放当前句子中处理好的“词”
        
        # === 核心逻辑：按词处理 ===
        for seg in jieba.cut(text):
            # seg 是一个分词结果，例如 "北京" 或 "Hello" 或 "，"
            
            # 过滤掉纯空格 (除非你想保留原文本的空格作为停顿)
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
            
            # === 词内部用 | 连接 ===
            # "北京" -> "bei3|jing1"
            # "Hello" -> "H|e|l|l|o"
            if current_word_parts:
                sentence_words.append("|".join(current_word_parts))
        
        # === 词之间用空格连接 ===
        # 结果: "wo3 ai4 bei3|jing1 H|e|l|l|o"
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
        self.main_language_code = language
        self.separator = separator
        
        # espeak 参数
        self.espeak_kwargs = {
            "punctuation_marks": punctuation_marks,
            "preserve_punctuation": preserve_punctuation,
            "with_stress": with_stress,
            "tie": tie,
            "language_switch": language_switch,
            "words_mismatch": words_mismatch
        }
        self.allowed_targets = ALLOWED_SWITCHES.get(language, []) # 获取当前语言的白名单

        if backend == "espeak":
            # 1. 初始化 espeak 后端池
            self.backends: Dict[str, EspeakBackend] = {}
            self.backends[language] = EspeakBackend(language, **self.espeak_kwargs)
            
            # 2. 初始化检测工具 (仅针对拉丁语系)
            self.spell_checker = None
            self.lingua_detector = None
            
            # 判断是否需要 Code-switching (主语言在我们的支持列表里)
            spell_lang_code = MAP_TO_SPELLCHECKER.get(language, None)
            
            if spell_lang_code:
                try:
                    # A. 加载 SpellChecker (词典)
                    self.spell_checker = SpellChecker(language=spell_lang_code)
                    
                    # B. 加载 Lingua Detector (AI模型)
                    # 我们只关注这几种常用语言之间的混淆，减少计算量
                    languages_to_detect = [
                        Language.ENGLISH, Language.GERMAN, 
                        Language.FRENCH, Language.SPANISH, 
                        Language.PORTUGUESE, Language.ITALIAN
                    ]
                    self.lingua_detector = LanguageDetectorBuilder.from_languages(*languages_to_detect).build()
                    
                    print(f"[Init] Code-switching enabled for {language}. Loaded SpellChecker & Lingua.")
                    
                    # 预加载常用 espeak backend
                    if language != "en-us":
                        self.backends["en-us"] = EspeakBackend("en-us", **self.espeak_kwargs)
                        
                except Exception as e:
                    print(f"[Warn] Failed to init code-switching tools: {e}")
        else:
            raise NotImplementedError(f"{backend}")

    def get_backend(self, lang_code):
        """Lazy load espeak backend"""
        if lang_code in self.backends:
            return self.backends[lang_code]
        try:
            bk = EspeakBackend(lang_code, **self.espeak_kwargs)
            self.backends[lang_code] = bk
            return bk
        except:
            return self.backends[self.main_language_code]

    def clean_text(self, text):
        text = text.strip()
        text = regex.sub(r"\p{C}|\p{Z}", " ", text)
        text = REPLACE_SYMBOL_REGEX.sub(lambda x: SYMBOLS_MAPPING[x.group()], text)
        text = re.sub(r'([,.:;!?…—])', r' \1 ', text)
        text = re.sub(r'\s+', ' ', text)
        return text

    def post_clean_ipa(self, ipa_string: str) -> str:
        ipa_string = re.sub(r'[\[\](){}]', '', ipa_string)
        ipa_string = regex.sub(r",(?!\s)", "ˌ", ipa_string)
        ipa_string = re.sub(r'\|+', '|', ipa_string)
        ipa_string = re.sub(r' \| ', ' ', ipa_string)
        ipa_string = re.sub(r'^\||\|$', '', ipa_string)
        ipa_string = re.sub(r'\s+', ' ', ipa_string)
        return ipa_string.strip()

    def get_word_g2p_language(self, word: str) -> str:
        """
        决定一个词应该用什么语言的 G2P。
        返回 espeak 的 language code (如 'en-us', 'de')。
        如果返回 None，使用主语言。
        """
        # 1. 基础过滤：太短、非字母 -> 不处理，用主语言
        if len(word) < 4 or not word.isalpha():
            return None
        
        # 2. 【第一道防线】查主语言字典
        # 如果词在主语言字典里，绝对不切语言 (防止 die, gift, art 等同形词误判)
        if self.spell_checker:
            # 检查原词和小写形式
            if word in self.spell_checker or word.lower() in self.spell_checker:
                return None # 是主语言单词
        #print(f"find unmatched word: {word}")
        # 3. 【第二道防线】Lingua 智能检测
        # 能走到这一步，说明这个词不在主语言字典里 (可能是借词，也可能是拼写错误)
        if self.lingua_detector:
            try:
                # detect_language_of 返回置信度最高的语言
                detected_lang = self.lingua_detector.detect_language_of(word)
                
                if detected_lang:
                    
                    # 将 Lingua 对象映射回 espeak 代码
                    target_espeak_code = MAP_LINGUA_TO_ESPEAK.get(detected_lang)
                    print(f"{word} will be changed to {target_espeak_code}")
                    # 只有当检测结果不是主语言时才切换
                    if target_espeak_code and target_espeak_code in self.allowed_targets:
                        return target_espeak_code
                    
            except:
                pass
                
        return None

    def __call__(self, text, strip=True) -> str:
        if isinstance(text, str):
            text = [self.clean_text(text)]
        elif isinstance(text, list):
            text = [self.clean_text(t) for t in text]
        else:
            return ""

        input_text = text[0]

        # Case 1: 中文/泰语 (直接返回)
        if self.main_language_code in ['cmn', 'zh']:
            return convert_char_to_pinyin([input_text])[0]
        elif self.main_language_code == 'th':
            words = word_tokenize(input_text, engine="newmm")
            ipa_words=[]
            for word in words:
                if not word.strip(): continue
                raw_ipa = transliterate(word, engine="ipa")
                if raw_ipa:
                    raw_ipa = raw_ipa.replace("͡", "").replace("-", "")
                    formatted_ipa = "|".join(list(raw_ipa))
                    ipa_words.append(formatted_ipa)
            return " ".join(ipa_words)

        # 其他语言
        else:
            # 如果没有初始化检测工具，回退到普通模式
            if not self.spell_checker or not self.lingua_detector:
                ph = self.backends[self.main_language_code].phonemize([input_text], separator=self.separator, strip=strip, njobs=1)[0]
                return self.post_clean_ipa(ph)

            # 正常 Code-switching 流程
            words = input_text.split()
            final_ipa_list = []
            
            for word in words:
                # 标点符号直接处理
                if re.match(r'^[,.:;!?…—]+$', word):
                    ph = self.backends[self.main_language_code].phonemize([word], separator=self.separator, strip=strip, njobs=1)[0]
                    final_ipa_list.append(ph.strip())
                    continue

                # 判定语言
                target_lang = self.get_word_g2p_language(word)
                # 选择后端
                backend = self.backends[self.main_language_code]
                if target_lang:
                    backend = self.get_backend(target_lang)
                    print(f"DEBUG: Switching '{word}' to {target_lang}") 
                ph = backend.phonemize([word], separator=self.separator, strip=strip, njobs=1)[0]
                final_ipa_list.append(ph.strip())

            return self.post_clean_ipa(" ".join(final_ipa_list))
def tokenize_text(tokenizer: PhonemizeTextTokenizer, text: str) -> str:
    phonemes = tokenizer([text.strip()])
    return phonemes

def run_test():
    TEST_CASES = [
    # --- 🇩🇪 German (德语) ---
    # [Trap] "Die" 是定冠词，不能读成英语的 "死" (/daɪ/)
    ("de", "Die Katze schläft auf dem Sofa."),
    # [Trap] "Art" 是种类，不能读成英语的 "艺术" (/ɑːt/)
    ("de", "Das ist eine besondere Art von Kunst."),
    # [Trap] "Gift" 是毒药，不能读成英语的 "礼物" (/gɪft/)
    ("de", "Vorsicht, das ist Gift!"),
    # [Trap] "Rat" 是建议，不能读成英语的 "老鼠" (/ræt/)
    ("de", "Ich brauche deinen Rat."),
    # [Loanword] "Wireframe": W应读/w/, i应读/ai/ (德语G2P会读 /v/ /i:/)
    ("de", "Das Design basiert auf einem Wireframe."),
    # [Loanword] "Manager", "Know-how": g的发音, ow的发音
    ("de", "Der Manager hat viel Know-how."),
    # [Loanword] "Recycling": y在德语里通常读/y/(郁)，这里必须读/ai/
    ("de", "Recycling ist gut für die Umwelt."),
    # [Loanword] "Browser", "Update": ow和u的发音
    ("de", "Bitte mach ein Update für deinen Browser."),

    # --- 🇮🇹 Italian (意大利语) ---
    # [Trap] "Burro" 是黄油，不是西班牙语的驴，也不是英语借词
    ("it", "Mi piace il pane con il burro."),
    # [Trap] "Mare" 是海，不是英语的母马 (/mɛə/)
    ("it", "Andiamo al mare domani."),
    # [Loanword] "Management": g不发音，a发梅花音
    ("it", "Il management ha deciso di cambiare strategia."),
    # [Loanword] "Budget": u发/ʌ/，g发/dʒ/ (意语会读 /bud-dʒet/)
    ("it", "Abbiamo superato il budget previsto."),
    # [Loanword] "Smartphone", "Touchscreen": 复合词检测
    ("it", "Il mio smartphone ha un touchscreen rotto."),
    # [Loanword] "Privacy": i发/ai/ (意语会读 /pri-va-tʃi/)
    ("it", "La privacy è molto importante."),

    # --- 🇵🇹 Portuguese (葡萄牙语) ---
    # [Trap] "Pé" (脚) vs English "Pie" (派) - 虽然拼写不同，但测试短词逻辑
    ("pt", "Eu machuquei o meu pé."),
    # [Trap] "Real" (货币/真实的) vs English "Real"
    ("pt", "Isso não parece real."),
    # [Loanword] "Software", "Hardware": 葡语R是小舌音或H音，这里需要卷舌
    ("pt", "Preciso atualizar o software e o hardware."),
    # [Loanword] "Layout": ay读/ei/
    ("pt", "O layout do site ficou excelente."),
    # [Loanword] "Feedback": ee读/i:/
    ("pt", "O cliente nos deu um ótimo feedback."),
    # [Loanword] "Hollywood": H在葡语不发音，这里必须发/h/
    ("pt", "Ele sonha em ir para Hollywood."),

    # --- 🇪🇸 Spanish (西班牙语) ---
    # [Trap] "Pie" 是脚，不能读成英语的 "派" (/paɪ/)
    ("es", "Me duele mucho el pie."),
    # [Trap] "Red" 是网络，不能读成英语的 "红色" (/rɛd/)
    ("es", "La red social es muy popular."),
    # [Trap] "Once" 是十一，不能读成英语的 "曾经" (/wʌns/)
    ("es", "Tengo once gatos."),
    # [Loanword] "Marketing": ing结尾
    ("es", "Trabajo en el departamento de marketing."),
    # [Loanword] "Show", "Business": sh发音
    ("es", "El show business es complicado."),
    # [Loanword] "Wifi", "Online": i的发音
    ("es", "No hay señal de Wifi aqui."),
    
    # --- 🇫🇷 French (法语) ---
    # [Trap] "Pain" 是面包，不能读成英语的 "痛苦" (/peɪn/)
    ("fr", "J'aime le pain frais."),
    # [Trap] "Coin" 是角落，不能读成英语的 "硬币" (/kɔɪn/)
    ("fr", "Il est dans le coin."),
    # [Loanword] "Weekend": w在法语通常读/v/，这里要读/w/
    ("fr", "Bon weekend à tous."),
    # [Loanword] "Parking", "Shopping": ing在法语有时会鼻化，用英语G2P更准
    ("fr", "Le parking est complet pour le shopping."),
    # [Loanword] "Brainstorming": 极难读对的词
    ("fr", "Nous avons fait un brainstorming."),
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
# python src/f5_tts/train/datasets/ipa_v4_tokenizer.py