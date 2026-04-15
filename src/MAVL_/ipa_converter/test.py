from epitran_utils import get_valid_epitran_mappings_list, get_epitran
from language_processors import (
    english,
    spanish,
    french,
    korean,
    japanese,
)


def transliterate_example_languages():
    lang_map = {
        "en": ("eng-Latn", english),
        "es": ("spa-Latn", spanish),
        "fr": ("fra-Latn", french),
        "ko": ("kor-Hang", korean),
        "ja": ("jpn-Hrgn", japanese),
    }

    examples = {
        "en": "Hello, how are you today?",
        "es": "Hola, ¿cómo estás hoy?",
        "fr": "Bonjour, comment ça va aujourd'hui ?",
        "ko": "안녕하세요, 오늘 어떻게 지내세요? English Test",
        "ja": "Konnichiwa、今日はどうですか？ English 今日はどうですか",
    }

    valid_epitran_mappings = get_valid_epitran_mappings_list()
    eng_epi = get_epitran("eng-Latn")

    for lang_code, (epitran_mapping, processor) in lang_map.items():
        # if epitran_mapping in valid_epitran_mappings:
        print(f"Processing language: {lang_code} ({epitran_mapping})")
        try:
            epi = get_epitran(epitran_mapping)
            example_text = examples[lang_code]
            example_text = processor.process_text(example_text)
            transliteration = processor.transliterate(
                example_text, epi=epi, eng_epi=eng_epi
            )
            # transliteration = epi.transliterate(example_text)
            print(f"Original text: {example_text}")
            print(f"Transliteration (IPA): {transliteration}")
        except Exception as e:
            print(f"Error processing {lang_code} ({epitran_mapping}): {e}")
        # else:
        #     print(f"Mapping {epitran_mapping} not found in valid mappings!")


if __name__ == "__main__":
    transliterate_example_languages()
