import re
import sys
from typing import Optional, Tuple
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    from language_processors.english import process_text as eng_processor
except ModuleNotFoundError:
    from ipa_converter.language_processors.english import process_text as eng_processor

from .syllabifier import cmuparser3
from .syllabifier.syllable3 import generate_syllables

cmu_dict = cmuparser3.CMUDictionary()


def generate(candidate: str) -> Optional[map]:
    phoneme_str = cmu_dict.get_first(candidate)
    if phoneme_str:
        return generate_syllables(phoneme_str)
    else:
        return None


def num_syllables(candidate: str) -> Optional[Tuple[map, int]]:
    # print(f"{candidate}")
    syl_map = generate(candidate)
    if syl_map is not None:
        return syl_map, len(syl_map)
    return None


# Patterns for estimating syllable count
sub_syllables = [
    "cial",
    "tia",
    "cius",
    "cious",
    "uiet",
    "gious",
    "geous",
    "priest",
    "giu",
    "dge",
    "ion",
    "iou",
    "sia$",
    ".che$",
    ".ched$",
    ".abe$",
    ".ace$",
    ".ade$",
    ".age$",
    ".aged$",
    ".ake$",
    ".ale$",
    ".aled$",
    ".ales$",
    ".ane$",
    ".ame$",
    ".ape$",
    ".are$",
    ".ase$",
    ".ashed$",
    ".asque$",
    ".ate$",
    ".ave$",
    ".azed$",
    ".awe$",
    ".aze$",
    ".aped$",
    ".athe$",
    ".athes$",
    ".ece$",
    ".ese$",
    ".esque$",
    ".esques$",
    ".eze$",
    ".gue$",
    ".ibe$",
    ".ice$",
    ".ide$",
    ".ife$",
    ".ike$",
    ".ile$",
    ".ime$",
    ".ine$",
    ".ipe$",
    ".iped$",
    ".ire$",
    ".ise$",
    ".ished$",
    ".ite$",
    ".ive$",
    ".ize$",
    ".obe$",
    ".ode$",
    ".oke$",
    ".ole$",
    ".ome$",
    ".one$",
    ".ope$",
    ".oque$",
    ".ore$",
    ".ose$",
    ".osque$",
    ".osques$",
    ".ote$",
    ".ove$",
    ".pped$",
    ".sse$",
    ".ssed$",
    ".ste$",
    ".ube$",
    ".uce$",
    ".ude$",
    ".uge$",
    ".uke$",
    ".ule$",
    ".ules$",
    ".uled$",
    ".ume$",
    ".une$",
    ".upe$",
    ".ure$",
    ".use$",
    ".ushed$",
    ".ute$",
    ".ved$",
    ".we$",
    ".wes$",
    ".wed$",
    ".yse$",
    ".yze$",
    ".rse$",
    ".red$",
    ".rce$",
    ".rde$",
    ".ily$",
    ".ely$",
    ".des$",
    ".gged$",
    ".kes$",
    ".ced$",
    ".ked$",
    ".med$",
    ".mes$",
    ".ned$",
    ".[sz]ed$",
    ".nce$",
    ".rles$",
    ".nes$",
    ".pes$",
    ".tes$",
    ".res$",
    ".ves$",
    "ere$",
]
add_syllables = [
    "ia",
    "riet",
    "dien",
    "ien",
    "iet",
    "iu",
    "iest",
    "io",
    "ii",
    "ily",
    ".oala$",
    ".iara$",
    ".ying$",
    ".earest",
    ".arer",
    ".aress",
    ".eate$",
    ".eation$",
    "[aeiouym]bl$",
    "[aeiou]{3}",
    "^mc",
    "ism",
    "^mc",
    "asm",
    "([^aeiouy])1l$",
    "[^l]lien",
    "^coa[dglx].",
    "[^gq]ua[^auieo]",
    "dnt$",
]

re_sub_syllables = [re.compile(s) for s in sub_syllables]
re_add_syllables = [re.compile(s) for s in add_syllables]


def estimate(word):
    """Estimates the number of syllables in an English-language word

    Parameters
    ----------
    word : str
        The English-language word to estimate syllables for

    Returns
    -------
    int
        The estimated number of syllables in the word
    """
    # Convert to lowercase
    lower_word = word.lower()

    # Split by non-vowel (aeiouy) characters
    parts = re.split(r"[^aeiouy]+", lower_word)
    valid_parts = [p for p in parts if p != ""]

    # Default syllable count: number of vowel groups
    syllables = len(valid_parts)

    # Adjust syllable count based on special patterns (subtraction)
    for pattern in re_sub_syllables:
        if pattern.match(lower_word):
            syllables -= 1

    # Adjust syllable count based on special patterns (addition)
    for pattern in re_add_syllables:
        if pattern.match(lower_word):
            syllables += 1

    # Ensure at least 1 syllable
    if syllables < 1:
        syllables = 1

    return syllables


def syllabify(word):
    """
    Splits an English word into 'syllable-like' parts and returns them as a list.
    (Splits based on vowel groups, including surrounding consonants, according to the syllable estimation logic in this code.)

    Example) "princess" -> ["prin", "cess"]
    """
    lower_word = word.lower()

    # Group and split vowel clusters and their surrounding consonants
    #   0 or more consonants -> [^aeiouy]*
    #   1 or more vowels -> [aeiouy]+
    #   0 or more consonants -> [^aeiouy]*
    pattern = r"[^aeiouy]*[aeiouy]+[^aeiouy]*"

    # Use findall to get all matching syllable chunks as a list
    chunks = re.findall(pattern, lower_word)

    # If there are no vowels (e.g., "zzz"), return the word as is.
    # In this case, the syllable count according to this code's logic is 1,
    # so the split is also treated as a single chunk ["zzz"].
    if not chunks:
        return [lower_word], 1

    return chunks, len(chunks)


def split_syllables(sentence: str) -> tuple[list[str], int]:
    sentence = eng_processor(sentence)
    words = sentence.split()
    syllable_chunks = []
    total_syllables = 0

    for word in words:
        chunks_and_count = num_syllables(word)
        if chunks_and_count is not None:
            chunks, count = chunks_and_count
            syllable_chunks.extend(chunks)
            total_syllables += count
        else:
            chunks, count = syllabify(word)
            syllable_chunks.extend(chunks)
            total_syllables += count

    return syllable_chunks, total_syllables
