from nemo_text_processing.text_normalization.normalize import Normalizer as nemo_normalizer
from num2words import num2words
import re
from functools import partial
try:
    from xphonebr import normalizer as pt_processor
except ImportError:
    pt_processor = None
try:
    from tts_preprocess_et.convert import convert_sentence as et_convert_sentence
except ImportError:
    et_convert_sentence = None
# import debugpy
# debugpy.listen(('localhost', 5678))
# print("Waiting for debugger attach")
# debugpy.wait_for_client()

class TextNormalizer:
    def __init__(self, language):
        self.language = language
        self.processor = self._init_language_processor()
        print(f"Successfully created text normalizer for {language}.")

    def _init_language_processor(self):
        # if self.language == "de":
        #     from german_transliterate.core import GermanTransliterate # de
        #     return = GermanTransliterate(
        #         replace={';': ',', ':': ' ','bzw.': 'beziehungsweise', 'u.a.': 'unter anderem','etc.': 'et cetera'},
        #         sep_abbreviation=' ',
        #         make_lowercase=True,
        #         transliterate_ops=[
        #             'acronym_phoneme',    # 缩写转音素友好型口语（如ABC→ah beh zee）
        #             'accent_peculiarity', # 清洗特殊Unicode字符，转为ASCII兼容格式
        #             'amount_money',       # 货币转口语（如250€→zweihundertfünfzig euro）
        #             'time_of_day',        # 时间转口语（如13:15h→dreizehn uhr fünfzehn）
        #             'ordinal',            # 序数词转口语（如1.→erste）
        #             'special',            # 特殊格式转口语（如8/10→acht von zehn）
        #         ]
        #     )
        if self.language in ["zh","yue"]:
            from tn.chinese.normalizer import Normalizer as zh_normalizer # zh
            return zh_normalizer()
        elif self.language == "bg":
            from bg_text_normalizer import BulgarianTextNormalizer
            return BulgarianTextNormalizer()

        else:
            try:
                return nemo_normalizer(input_case='cased', lang=self.language)
            except Exception as e:
                print(e)
                return None
                
            
    def split_german_sentence(self, text):
        if not text:
            return ""
        words = text.split()
        processed_words = []
        for word in words:
            # 只尝试拆分长度大于 5 的词
            if len(word) > 5:
                try:
                    # 获取拆分建议
                    splitting_results = char_split.split_compound(word)
                    # 如果得分超过阈值0.5，则进行拆分
                    if splitting_results and splitting_results[0][0] > 0.5:
                        _, part1, part2 = splitting_results[0]
                        # 递归处理
                        processed_words.append(self.split_german_sentence(part1))
                        processed_words.append(self.split_german_sentence(part2))
                    else:
                        processed_words.append(word)
                except Exception as e: 
                    print(e)
                    processed_words.append(word)
            else:
                processed_words.append(word)
                
        return " ".join(processed_words).strip()

    def clean_text_for_tts(self, text):
        text = re.sub(r'\s+', ' ', text).strip() # 合并空格
        text = text.lower()
        # 特殊字符替换
        text = text.replace('!!', '!').replace('¡¡', '¡')
        text = text.replace('!,', '!').replace('!.', '!')
        text = text.replace('?,', '?').replace('?.', '?')
        text = text.replace('-', ' ')
        quote_pattern = r'[„“»«”]' 
        text= re.sub(quote_pattern, '"', text)
        text = text.replace('#', '').replace('*',' ') # 德语可能会有*在两个单词中间，换为空格
        text = re.sub(r'\s+([.,!?;:])', r'\1', text) # 标点符号前的空格
        text = re.sub(r'([¡¿])\s+', r'\1', text) # 倒标点必须紧跟文本
        # text = re.sub(r'([a-z]+)([A-Z])', r'\1 \2', text) # 驼峰式命名的拆分
        if text:
            text = text[0].upper() + text[1:]
        return text
    
    def _replace_num(self, match, to=None):
        number = match.group()
        try:
            val = float(number)
            lang = self.language
            if val.is_integer():
                val = int(val)
                if to:
                    return num2words(val, lang=lang, to=to)
                return num2words(val, lang=lang)
            else:
                if to:
                    return num2words(val, lang=lang, to=to)
                return num2words(val, lang=lang)

        except Exception as e:
            logger.error(f"num2words failed for [{number}] in {self.language}: {e}")
            return number

    def normalize(self, text: str, post: bool=False, to=None) -> str:
        if not isinstance(text, str) or len(text.strip()) == 0:
            print("Text is not a string, please check.")
            return ""  
        text_clean = self.clean_text_for_tts(text)
        
        if self.language == "pt":
            text_normalized = pt_processor(text_clean)
        elif self.language == "et":
            from tts_preprocess_et.convert import convert_sentence
            text_normalized = convert_sentence(text_clean)
        elif self.processor and self.language in ["zh", "bg", "yue"]: # "de"
            text_normalized = self.processor.normalize(text_clean)
        elif self.processor:
            text_normalized = self.processor.normalize(text_clean, verbose=False, punct_post_process=True)
        else:
            replace_num_with_to = partial(self._replace_num, to=to)
            text_normalized = re.sub(r'\d+(\.\d+)?', replace_num_with_to, text_clean)
        # if self.language == "de":
        #     if post:
        #         from compound_split import char_split # de
        #         text_normalized = self.split_german_sentence(text_normalized)
        #         text_normalized = text_normalized.lower()
        #     else:
        #         text_normalized = text_normalized
            
        return text_normalized



if __name__ == "__main__":
    
    normalizer = TextNormalizer(language="et")    
    test_cases = [
        "Tere hommikust! Eestis on palju ilusaid metsi 2026..."
        # "Đó cũng là lý do glucose gọi là \"đường huyết\"."
        # "kommen fünf vertriebsgemeinkosten zu schlag"
        # "Die Rechtsschutzversicherungsgesellschaften prüfen das Arbeiterunfallversicherungsgesetz.",
        # "Ich habe das neue Update für den Webbrowser heruntergeladen, das Interface ist jetzt viel cooler.",
        # "Er sagte: „Das ist (vielleicht) die beste Lösung!“ – oder auch nicht.",
        # "Alle Lehrer*innen und Schüler:innen sind im Klassenzimmer.",
    ]

    for idx, test_text in enumerate(test_cases, 1):
        print(f"原始文本：{test_text}")
        normalized_text = normalizer.normalize(test_text,post=True) #, to="cardinal")
        print(f"归一化后：{normalized_text}\n")
        
# python src/f5_tts/eval/text_normalizer.py