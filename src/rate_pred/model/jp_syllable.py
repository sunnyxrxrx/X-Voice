from __future__ import annotations

import re
import unicodedata

import pykakasi


_KKS = pykakasi.kakasi()

_COMBINED_PATTERNS = [
    re.compile(r"^(?:お|こ|そ|と|の|ほ|も|よ|ろ|ご|ぞ|ど|ぼ|ぽ)(?:ー)?(?:う|ぅ)"),
    re.compile(r"^(?:い|き|し|ち|に|ひ|み|り|ぎ|じ|ぢ|び|ぴ)(?:ゆ|よ|ゅ|ょ)(?:ー)?(?:う|ぅ)?"),
    re.compile(r"^(?:い|き|し|ち|に|ひ|み|り|ぎ|じ|ぢ|び|ぴ)(?:や|ゃ)(?:ー)?(?:あ|ぁ)?"),
]

_ROMAJI_DICT = {
    ".": "ten",
    "0": "zero",
    "1": "ichi",
    "2": "ni",
    "3": "san",
    "4": "yon",
    "5": "go",
    "6": "roku",
    "7": "nana",
    "8": "hachi",
    "9": "kyuu",
    "10": "juu",
    "100": "hyaku",
    "1000": "sen",
    "10000": "man",
    "100000000": "oku",
    "300": "sanbyaku",
    "600": "roppyaku",
    "800": "happyaku",
    "3000": "sanzen",
    "8000": "hassen",
    "01000": "issen",
}

_KANJI_DICT = {
    ".": "点",
    "0": "零",
    "1": "一",
    "2": "二",
    "3": "三",
    "4": "四",
    "5": "五",
    "6": "六",
    "7": "七",
    "8": "八",
    "9": "九",
    "10": "十",
    "100": "百",
    "1000": "千",
    "10000": "万",
    "100000000": "億",
    "300": "三百",
    "600": "六百",
    "800": "八百",
    "3000": "三千",
    "8000": "八千",
    "01000": "一千",
}

_HIRAGANA_DICT = {
    ".": "てん",
    "0": "ゼロ",
    "1": "いち",
    "2": "に",
    "3": "さん",
    "4": "よん",
    "5": "ご",
    "6": "ろく",
    "7": "なな",
    "8": "はち",
    "9": "きゅう",
    "10": "じゅう",
    "100": "ひゃく",
    "1000": "せん",
    "10000": "まん",
    "100000000": "おく",
    "300": "さんびゃく",
    "600": "ろっぴゃく",
    "800": "はっぴゃく",
    "3000": "さんぜん",
    "8000": "はっせん",
    "01000": "いっせん",
}

_KEY_DICT = {"kanji": _KANJI_DICT, "hiragana": _HIRAGANA_DICT, "romaji": _ROMAJI_DICT}

_EN_SUB_SYLLABLES = [
    "cial", "tia", "cius", "cious", "uiet", "gious", "geous", "priest", "giu", "dge", "ion", "iou", "sia$",
    ".che$", ".ched$", ".abe$", ".ace$", ".ade$", ".age$", ".aged$", ".ake$", ".ale$", ".aled$", ".ales$",
    ".ane$", ".ame$", ".ape$", ".are$", ".ase$", ".ashed$", ".asque$", ".ate$", ".ave$", ".azed$", ".awe$",
    ".aze$", ".aped$", ".athe$", ".athes$", ".ece$", ".ese$", ".esque$", ".esques$", ".eze$", ".gue$",
    ".ibe$", ".ice$", ".ide$", ".ife$", ".ike$", ".ile$", ".ime$", ".ine$", ".ipe$", ".iped$", ".ire$",
    ".ise$", ".ished$", ".ite$", ".ive$", ".ize$", ".obe$", ".ode$", ".oke$", ".ole$", ".ome$", ".one$",
    ".ope$", ".oque$", ".ore$", ".ose$", ".osque$", ".osques$", ".ote$", ".ove$", ".pped$", ".sse$",
    ".ssed$", ".ste$", ".ube$", ".uce$", ".ude$", ".uge$", ".uke$", ".ule$", ".ules$", ".uled$", ".ume$",
    ".une$", ".upe$", ".ure$", ".use$", ".ushed$", ".ute$", ".ved$", ".we$", ".wes$", ".wed$", ".yse$",
    ".yze$", ".rse$", ".red$", ".rce$", ".rde$", ".ily$", ".ely$", ".des$", ".gged$", ".kes$", ".ced$",
    ".ked$", ".med$", ".mes$", ".ned$", ".[sz]ed$", ".nce$", ".rles$", ".nes$", ".pes$", ".tes$", ".res$",
    ".ves$", "ere$",
]
_EN_ADD_SYLLABLES = [
    "ia", "riet", "dien", "ien", "iet", "iu", "iest", "io", "ii", "ily", ".oala$", ".iara$", ".ying$",
    ".earest", ".arer", ".aress", ".eate$", ".eation$", "[aeiouym]bl$", "[aeiou]{3}", "^mc", "ism", "^mc",
    "asm", "([^aeiouy])1l$", "[^l]lien", "^coa[dglx].", "[^gq]ua[^auieo]", "dnt$",
]
_EN_RE_SUB = [re.compile(pattern) for pattern in _EN_SUB_SYLLABLES]
_EN_RE_ADD = [re.compile(pattern) for pattern in _EN_ADD_SYLLABLES]


def _normalize_text(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def _len_one(convert_num: str, requested_dict: dict[str, str]) -> str:
    return requested_dict[convert_num]


def _len_two(convert_num: str, requested_dict: dict[str, str]) -> str:
    if convert_num[0] == "0":
        return _len_one(convert_num[1], requested_dict)
    if convert_num == "10":
        return requested_dict["10"]
    if convert_num[0] == "1":
        return requested_dict["10"] + " " + _len_one(convert_num[1], requested_dict)
    if convert_num[1] == "0":
        return _len_one(convert_num[0], requested_dict) + " " + requested_dict["10"]

    num_list = [requested_dict[x] for x in convert_num]
    num_list.insert(1, requested_dict["10"])
    return " ".join(num_list)


def _len_three(convert_num: str, requested_dict: dict[str, str]) -> str:
    num_list = []
    if convert_num[0] == "1":
        num_list.append(requested_dict["100"])
    elif convert_num[0] == "3":
        num_list.append(requested_dict["300"])
    elif convert_num[0] == "6":
        num_list.append(requested_dict["600"])
    elif convert_num[0] == "8":
        num_list.append(requested_dict["800"])
    else:
        num_list.append(requested_dict[convert_num[0]])
        num_list.append(requested_dict["100"])

    if convert_num[1:] != "00" or len(convert_num) != 3:
        if convert_num[1] == "0":
            num_list.append(requested_dict[convert_num[2]])
        else:
            num_list.append(_len_two(convert_num[1:], requested_dict))

    return " ".join(num_list)


def _len_four(convert_num: str, requested_dict: dict[str, str], stand_alone: bool) -> str:
    num_list = []
    if convert_num == "0000":
        return ""
    while convert_num[0] == "0":
        convert_num = convert_num[1:]
    if len(convert_num) == 1:
        return _len_one(convert_num, requested_dict)
    if len(convert_num) == 2:
        return _len_two(convert_num, requested_dict)
    if len(convert_num) == 3:
        return _len_three(convert_num, requested_dict)

    if convert_num[0] == "1" and stand_alone:
        num_list.append(requested_dict["1000"])
    elif convert_num[0] == "1":
        num_list.append(requested_dict["01000"])
    elif convert_num[0] == "3":
        num_list.append(requested_dict["3000"])
    elif convert_num[0] == "8":
        num_list.append(requested_dict["8000"])
    else:
        num_list.append(requested_dict[convert_num[0]])
        num_list.append(requested_dict["1000"])

    if convert_num[1:] != "000" or len(convert_num) != 4:
        if convert_num[1] == "0":
            num_list.append(_len_two(convert_num[2:], requested_dict))
        else:
            num_list.append(_len_three(convert_num[1:], requested_dict))

    return " ".join(num_list)


def _len_x(convert_num: str, requested_dict: dict[str, str]) -> str:
    num_list = []
    prefix = convert_num[0:-4]
    if len(prefix) == 1:
        num_list.extend([requested_dict[prefix], requested_dict["10000"]])
    elif len(prefix) == 2:
        num_list.extend([_len_two(convert_num[0:2], requested_dict), requested_dict["10000"]])
    elif len(prefix) == 3:
        num_list.extend([_len_three(convert_num[0:3], requested_dict), requested_dict["10000"]])
    elif len(prefix) == 4:
        num_list.extend([_len_four(convert_num[0:4], requested_dict, False), requested_dict["10000"]])
    elif len(prefix) == 5:
        num_list.extend([requested_dict[convert_num[0]], requested_dict["100000000"], _len_four(convert_num[1:5], requested_dict, False)])
        if convert_num[1:5] != "0000":
            num_list.append(requested_dict["10000"])
    else:
        return "Not yet implemented, please choose a lower number."

    num_list.append(_len_four(convert_num[-4:], requested_dict, False))
    return " ".join(num_list)


def _remove_spaces(convert_result: str) -> str:
    return convert_result.replace(" ", "")


def _do_convert(convert_num: str, requested_dict: dict[str, str]) -> str:
    if len(convert_num) == 1:
        return _len_one(convert_num, requested_dict)
    if len(convert_num) == 2:
        return _len_two(convert_num, requested_dict)
    if len(convert_num) == 3:
        return _len_three(convert_num, requested_dict)
    if len(convert_num) == 4:
        return _len_four(convert_num, requested_dict, True)
    return _len_x(convert_num, requested_dict)


def _split_point(split_num: str, dict_choice: str) -> str:
    split_num_a, split_num_b = split_num.split(".")
    split_num_b_end = " " + " ".join(_len_one(x, _KEY_DICT[dict_choice]) for x in split_num_b) + " "
    if split_num_a[-1] == "0" and split_num_a[-2] != "0" and dict_choice == "hiragana":
        small_tsu = _convert_number(split_num_a, dict_choice)
        small_tsu = small_tsu[0:-1] + "っ"
        return small_tsu + _KEY_DICT[dict_choice]["."] + split_num_b_end
    if split_num_a[-1] == "0" and split_num_a[-2] != "0" and dict_choice == "romaji":
        small_tsu = _convert_number(split_num_a, dict_choice)
        small_tsu = small_tsu[0:-1] + "t"
        return small_tsu + _KEY_DICT[dict_choice]["."] + split_num_b_end
    return _convert_number(split_num_a, dict_choice) + " " + _KEY_DICT[dict_choice]["."] + split_num_b_end


def _convert_number(convert_num: str | int, dict_choice: str) -> str:
    convert_num = str(convert_num).replace(",", "")
    dict_choice = dict_choice.lower()
    dictionary = _KEY_DICT[dict_choice]

    while convert_num[0] == "0" and len(convert_num) > 1:
        convert_num = convert_num[1:]

    if "." in convert_num:
        result = _split_point(convert_num, dict_choice)
    else:
        result = _do_convert(convert_num, dictionary)

    if dictionary != _ROMAJI_DICT:
        result = _remove_spaces(result)
    return result


def _convert_full_to_half_width(text: str) -> str:
    full_width = "１２３４５６７８９０"
    half_width = "1234567890"
    return text.translate(str.maketrans(full_width, half_width))


def _kanji_to_hiragana(text: str) -> str:
    result = _KKS.convert(text)
    return "".join(item["hira"] for item in result)


def _number_to_hiragana(text: str) -> str:
    text = re.sub(r"\d+", lambda match: f" {_convert_number(match.group(), 'hiragana')} ", text)
    return re.sub(r"\s+", " ", text).strip()


def _english_process_text(text: str) -> str:
    text = _normalize_text(text)
    text = re.sub(r"\d+", lambda match: f" {match.group()} ", text)
    text = re.sub(r"['＇']", "'", text)
    text = re.sub(r"[^a-zA-ZÀ-ÿĀ-ſƀ-ƿǀ-ɏ'\" ]+", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def _english_num_syllables(candidate: str):
    lower_word = candidate.lower()
    pattern = r"[^aeiouy]*[aeiouy]+[^aeiouy]*"
    chunks = re.findall(pattern, lower_word)
    if not chunks:
        return [lower_word], 1

    syllables = len([part for part in re.split(r"[^aeiouy]+", lower_word) if part])
    for pattern_obj in _EN_RE_SUB:
        if pattern_obj.match(lower_word):
            syllables -= 1
    for pattern_obj in _EN_RE_ADD:
        if pattern_obj.match(lower_word):
            syllables += 1
    syllables = max(syllables, 1)
    return chunks, syllables


def _split_english_word(word: str) -> list[str]:
    processed = _english_process_text(word)
    if not processed:
        return []
    chunks, _ = _english_num_syllables(processed.split()[0])
    return chunks


def _process_text(text: str) -> str:
    text = _normalize_text(text)
    text = _convert_full_to_half_width(text)
    text = _kanji_to_hiragana(text)
    text = _number_to_hiragana(text)
    return text


def split_syllables(text: str) -> tuple[list[str], int]:
    text = _process_text(text)

    syllables: list[str] = []
    current_eng_word = ""
    i = 0
    length = len(text)

    while i < length:
        char = text[i]

        if char.isascii() and (char.isalpha() or char.isspace()):
            if current_eng_word or char.isalpha():
                current_eng_word += char
            i += 1
            continue

        if current_eng_word:
            for word in current_eng_word.strip().split():
                if word:
                    syllables.extend(_split_english_word(word))
            current_eng_word = ""

        if char == "ー":
            if syllables:
                syllables[-1] += "ー"
            else:
                syllables.append("ー")
            i += 1
            continue

        if char == "ん":
            if syllables:
                syllables[-1] += "ん"
            else:
                syllables.append("ん")
            i += 1
            continue

        if char == "っ":
            if syllables:
                syllables[-1] += "っ"
            else:
                syllables.append("っ")
            i += 1
            continue

        substring_matched = False
        for pattern in _COMBINED_PATTERNS:
            match = pattern.match(text[i:])
            if match:
                matched_str = match.group(0)
                syllables.append(matched_str)
                i += len(matched_str)
                substring_matched = True
                break
        if substring_matched:
            continue

        if "ぁ" <= char <= "ん":
            syllables.append(char)
        i += 1

    if current_eng_word:
        for word in current_eng_word.strip().split():
            if word:
                syllables.extend(_split_english_word(word))

    return syllables, len(syllables)
