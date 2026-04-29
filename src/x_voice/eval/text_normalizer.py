from nemo_text_processing.text_normalization.normalize import Normalizer as nemo_normalizer
from num2words import num2words
import re
from functools import partial



class TextNormalizer:
    def __init__(self, language):
        self.language = language
        self.processor = self._init_language_processor()
        print(f"Successfully created text normalizer for {language}.")

    def _init_language_processor(self):
        try:
            if self.language == "zh":
                from tn.chinese.normalizer import Normalizer as zh_normalizer # zh
                return zh_normalizer()
            elif self.language == "bg":
                from bg_text_normalizer import BulgarianTextNormalizer
                return BulgarianTextNormalizer()
            else:
                return nemo_normalizer(input_case='cased', lang=self.language)
        except Exception as e:
            print(e)
            return None

    def clean_text_for_tts(self, text):
        text = re.sub(r'\s+', ' ', text).strip() # Collapse repeated whitespace.
        text = text.lower()
        # Normalize punctuation and special symbols.
        text = text.replace('!!', '!').replace('¡¡', '¡')
        text = text.replace('!,', '!').replace('!.', '!')
        text = text.replace('?,', '?').replace('?.', '?')
        text = text.replace('-', ' ')
        quote_pattern = r'[„“»«”]' 
        text= re.sub(quote_pattern, '"', text)
        text = text.replace('#', '').replace('*',' ') # German text may use * between words; convert it to a space.
        text = re.sub(r'\s+([.,!?;:])', r'\1', text) # Remove spaces before punctuation.
        text = re.sub(r'([¡¿])\s+', r'\1', text) # Keep inverted punctuation attached to the text.
        # text = re.sub(r'([a-z]+)([A-Z])', r'\1 \2', text) # Split camelCase words.
        if text:
            text = text[0].upper() + text[1:]
        return text
    
    def _replace_num(self, match):
        number = match.group()
        try:
            val = float(number)
            lang = self.language
            if val.is_integer():
                val = int(val)
            return num2words(val, lang=lang)

        except Exception as e:
            print(e)
            return number

    def normalize(self, text: str) -> str:
        if not isinstance(text, str) or len(text.strip()) == 0:
            print("Text is not a string, please check.")
            return ""  
        text_clean = self.clean_text_for_tts(text)
        
        if self.language == "pt":
            try:
                from xphonebr import normalizer as pt_processor
                text_normalized = pt_processor(text_clean)
            except Exception as e:
                print(e)
                text_normalized = text_clean
        elif self.language == "et":
            try:
                from tts_preprocess_et.convert import convert_sentence
                text_normalized = convert_sentence(text_clean)
            except Exception as e:
                print(e)
                text_normalized = text_clean
        elif self.processor and self.language in ["zh", "bg"]:
            text_normalized = self.processor.normalize(text_clean)
        elif self.processor:
            text_normalized = self.processor.normalize(text_clean, verbose=False, punct_post_process=True)
        else:
            text_normalized = re.sub(r'\d+(\.\d+)?', self._replace_num, text_clean)

            
        return text_normalized



if __name__ == "__main__":
    
    normalizer = TextNormalizer(language="zh")    
    test_cases = [
        "7 12 57 0 5 40"
    ]

    for idx, test_text in enumerate(test_cases, 1):
        print(f"Original text: {test_text}")
        normalized_text = normalizer.normalize(test_text)
        print(f"Normalized text: {normalized_text}\n")
        
# python src/x_voice/eval/text_normalizer.py
