""" 
Data types for syllabification
"""

from typing import Optional, List, Type


from .phoneme_types import *


class Phoneme:
    def __init__(self, phoneme: str):
        self.phoneme = phoneme

    @property
    def is_approximate(self):
        return self.phoneme in APPROXIMANTS

    @property
    def is_obstruent(self):
        return self.phoneme in OBSTRUENTS

    @property
    def is_voiced_obstruent(self):
        return self.phoneme in VOICED_OBSTRUENTS

    def __eq__(self, other):
        if type(other) == str:
            return self.phoneme == other
        return self.phoneme == other.phoneme

    def __repr__(self):
        return self.phoneme


class Vowel(Phoneme):
    """ Represents an individual phoneme that has been classified as a vowel """


class Consonant(Phoneme):
    """ Represents an individual phoneme that has been classified as a consonant """


class Cluster:
    """Represents groups of phonemes. Clusters contain either Vowels, or Consonants - never both"""

    def __init__(self, phoneme: Optional[Phoneme] = None):
        self.phoneme_list: List[Phoneme] = []
        if phoneme:
            self.phoneme_list.append(phoneme)

    def append(self, rest):
        self.phoneme_list.append(rest)

    def extend(self, rest):
        if type(rest) == Cluster:
            self.phoneme_list.extend(rest.phoneme_list)
        else:
            self.phoneme_list.extend(rest)

    @property
    def is_complex(self):
        """
        A cluster is complex if it is composed of consonants, and it has more than one consonant
        """
        return len(self.phoneme_list) > 1 and type(self.phoneme_list[0]) == Consonant

    def find_first(self, phoneme: str):
        """
        Return the index of first occurrence of `phoneme` in `self.phoneme_list`
        """

        if not self.phoneme_list:
            raise ValueError(f"'{phoneme}' is not in phoneme_list")
        if type(self.phoneme_list[0]) == Consonant:
            to_search = Consonant(phoneme)
        else:
            raise ValueError(f"Don't know how to find '{phoneme}' in phoneme_list")
        return self.phoneme_list.index(to_search)

    @property
    def first(self):
        """
        Returns the first phoneme
        """
        return self.phoneme_list[0]

    @property
    def second(self):
        """
        Returns the second phoneme
        :return:
        """
        return self.phoneme_list[1]

    @property
    def type(self) -> Optional[Type[Phoneme]]:
        """returns the type of the phoneme cluster: either Vowel, or Consonant"""
        if not self.phoneme_list:
            return None
        else:
            return type(self.phoneme_list[-1])

    def can_cluster_with(self, next_phoneme: Phoneme):
        """only consonants can cluster, if it's not NG"""

        if self.type is None:
            return True

        return (
            self.type == Consonant
            and type(next_phoneme) == Consonant
            and NG not in self.phoneme_list
        )

    def __eq__(self, other):
        if not hasattr(other, "phoneme_list"):
            return False
        return self.phoneme_list == other.phoneme_list

    def __bool__(self):
        return self.phoneme_list != []

    def __str__(self):
        return "".join([ph.phoneme for ph in self.phoneme_list])

    def __contains__(self, item):
        if not self.phoneme_list:
            return False
        if type(item) == str:
            if self.phoneme_list and type(self.phoneme_list[0]) == Consonant:
                item = Consonant(item)
        return item in self.phoneme_list


class Syllable:
    """
    Represents an English syllable with an onset, nucleus, and coda, some or all of
    which may be empty.
    """

    def __init__(
        self,
        onset: Optional[Cluster] = None,
        nucleus: Optional[Cluster] = None,
        coda: Optional[Cluster] = None,
    ):
        self.onset = onset
        self.nucleus = nucleus
        self.coda = coda

    @property
    def is_empty(self):
        return all([cl is None for cl in [self.onset, self.nucleus, self.coda]])

    def __repr__(self):
        return f"<Syllable -- onset={self.onset}, nucleus={self.nucleus}, coda={self.coda}>"

    def __str__(self):
        return f"<o:{self.onset}|n:{self.nucleus}|c:{self.coda}>"
