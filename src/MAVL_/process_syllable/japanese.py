import re
import os
import sys

sys.path.append(os.getcwd())

from language_processors.japanese import process_text as ja_processor
from process_syllable.english import split_syllables as en_split_syllables

# 1) Regular expression to remove all characters except hiragana (ぁ-ん) and the long vowel mark (ー)
FILTER_PATTERN = re.compile(r"[^ぁ-んー]")

# 2) "Combined patterns" (if an exact match occurs without the long vowel 'ー', treat it as a single chunk)
COMBINED_PATTERNS = [
    # (お|こ|そ|と|の|ほ|も|よ|ろ|ご|ぞ|ど|ぼ|ぽ)(う|ぅ)
    re.compile(r"^(?:お|こ|そ|と|の|ほ|も|よ|ろ|ご|ぞ|ど|ぼ|ぽ)(?:ー)?(?:う|ぅ)"),
    # (い|き|し|ち|に|ひ|み|り|ぎ|じ|ぢ|び|ぴ)(ゆ|よ|ゅ|ょ)(ー)?(?:う|ぅ)?
    re.compile(
        r"^(?:い|き|し|ち|に|ひ|み|り|ぎ|じ|ぢ|び|ぴ)(?:ゆ|よ|ゅ|ょ)(?:ー)?(?:う|ぅ)?"
    ),
    # (い|き|し|ち|に|ひ|み|り|ぎ|じ|ぢ|び|ぴ)(や|ゃ)(ー)?(?:あ|ぁ)?
    re.compile(
        r"^(?:い|き|し|ち|に|ひ|み|り|ぎ|じ|ぢ|び|ぴ)(?:や|ゃ)(?:ー)?(?:あ|ぁ)?"
    ),
]


def split_syllables(text: str):
    """
    From the given text:
      - Keep only hiragana (ぁ-ん) and 'ー'
      - If a given pattern matches (o+う, i+yu/yo/ya etc.), treat it as a single chunk.
      - 'ー' is included in the preceding syllable (unlike before, it's not separated into a new syllable).
      - みんな → ['みん', 'な'] ('ん' is attached to the preceding syllable, and a new syllable starts after it)
      - 'っ' is treated as a separate syllable (chunk).
      - Otherwise, basically split by single characters.
    Returns a list.
    """
    text = ja_processor(text)

    syllables = []
    i = 0
    length = len(text)
    current_eng_word = ""

    while i < length:
        c = text[i]

        # Check if it's an English character
        if c.isascii() and (c.isalpha() or c.isspace()):
            if current_eng_word or c.isalpha():
                current_eng_word += c
            i += 1
            continue

        # If there was an English word, process it
        if current_eng_word:
            eng_words = current_eng_word.strip().split()
            for word in eng_words:
                if word:
                    eng_syllables, eng_count = en_split_syllables(word)
                    syllables.extend(eng_syllables)
            current_eng_word = ""

        # If it's 'ー', append to the previous syllable
        if c == "ー":
            if len(syllables) > 0:
                syllables[-1] = syllables[-1] + "ー"
            else:
                syllables.append("ー")
            i += 1
            continue

        # Process 'ん'
        if c == "ん":
            if len(syllables) > 0:
                syllables[-1] = syllables[-1] + "ん"
            else:
                syllables.append("ん")
            i += 1
            continue

        if c == "っ":
            if len(syllables) > 0:
                syllables[-1] = syllables[-1] + "っ"
            else:
                syllables.append("っ")
            i += 1
            continue

        # Pattern matching part
        substring_matched = False
        for pattern in COMBINED_PATTERNS:
            match = pattern.match(text[i:])
            if match:
                matched_str = match.group(0)
                syllables.append(
                    matched_str
                )  # Treat the whole match as one syllable, regardless of long vowel presence
                i += len(matched_str)
                substring_matched = True
                break

        if substring_matched:
            continue

        # if hiragana
        if "ぁ" <= c <= "ん":
            syllables.append(c)
        i += 1

    # Process the last English word
    if current_eng_word:
        eng_words = current_eng_word.strip().split()
        for word in eng_words:
            if word:
                eng_syllables, eng_count = en_split_syllables(word)
                syllables.extend(eng_syllables)

    # # Append characters other than hiragana, long vowels, and English to the preceding syllable
    # final_syllables = []
    # for syllable in syllables:
    #     if FILTER_PATTERN.match(syllable):
    #         if final_syllables:
    #             final_syllables[-1] += syllable
    #         else:
    #             final_syllables.append(syllable)
    #     else:
    #         final_syllables.append(syllable)

    return syllables, len(syllables)
