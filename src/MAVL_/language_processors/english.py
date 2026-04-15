import re


def normalize_text(text):
    """Normalizes all types of whitespace characters to regular spaces."""
    return re.sub(r"\s+", " ", text).strip()


def process_text(text):
    try:
        from num2words import num2words
    except ImportError:
        raise ImportError(
            "num2words package is required. Please run 'pip install num2words'"
        )

    text = normalize_text(text)

    # Convert numbers to English words (add spaces before and after)
    text = re.sub(r"\d+", lambda m: f" {num2words(int(m.group()), lang='en')} ", text)

    # Unify apostrophes
    text = re.sub(r"['＇']", "'", text)

    # Convert special characters to spaces (preserve Roman alphabets and all types of apostrophes)
    text = re.sub(r"[^a-zA-ZÀ-ÿĀ-ſƀ-ƿǀ-ɏ'\" ]+", " ", text)

    # Unify consecutive spaces into one
    text = re.sub(r"\s+", " ", text).strip()

    return text


def transliterate(sentence, epi, eng_epi=None):
    sentence = process_text(sentence)
    return epi.transliterate(sentence)