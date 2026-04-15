import string
import sys
from typing import Optional, List

from . import cmuparser3
from .phoneme_types import *
from .syllable_types3 import (
    Cluster,
    Consonant,
    Vowel,
    Syllable,
    Phoneme,
)

CList = List[Cluster]
SList = List[Syllable]

# https://ipfs.io/ipfs/bafykbzacecizbpwbwfzejh2ynyfvxbyhuuyqcw54sfy3h3kaiqrrhxbggoatu?filename=%28The%20Language%20Library%29%20Heidi%20Harley%20-%20English%20Words_%20A%20Linguistic%20Introduction-Wiley-Blackwell%20%282006%29.pdf:w


def parse_phonemes(phoneme_string: str) -> Phoneme:
    """creates a Vowel or Consonant from the single phoneme represented by `phoneme_string`"""

    remove_digits = str.maketrans("", "", string.digits)
    phoneme_string = phoneme_string.translate(remove_digits)
    if phoneme_string in VOWELS:
        return Vowel(phoneme_string)
    elif phoneme_string in CONSONANTS:
        return Consonant(phoneme_string)
    raise ValueError(f"Don't recognize phoneme {phoneme_string}")


def cluster_phonemes(clusters: CList, next_phoneme: Phoneme) -> CList:
    """consonants must be grouped together into clusters"""

    if not clusters:
        # first phoneme, so cannot cluster with anything
        return [Cluster(next_phoneme)]

    if clusters[-1].can_cluster_with(next_phoneme):
        # if we can cluster, just update the last cluster on the list with the new phoneme
        clusters[-1].append(next_phoneme)
    else:
        # otherwise, make a new cluster and add it to the list
        clusters.append(Cluster(next_phoneme))

    return clusters


def syllabify_clusters(syllables: SList, cluster: Cluster) -> Optional[SList]:
    """
    decides whether the next cluster is part of current syllable or a new syllable
    """

    if cluster.type not in [Vowel, Consonant]:
        raise AttributeError(f"Unknown phoneme cluster type {type(cluster)}")

    if not syllables:
        syllables = [Syllable()]

    last_syl = syllables[-1]

    if cluster.type == Vowel:
        # Vowel clusters can only be part of current syllable if it doesn't
        # already have a nucleus. Otherwise it's a new syllable
        if last_syl.nucleus:
            # this cluster becomes the nucleus of a new syllable
            new_syllable = Syllable(nucleus=cluster)
            syllables.append(new_syllable)
        else:
            # syllable doesn't have nucleus so this cluster becomes the nucleus on the current syllable
            last_syl.nucleus = cluster
        return syllables

    # cluster is a consonant below this point

    if last_syl.is_empty:
        # no onset, nucleus, or coda, so this cluster should be the onset
        last_syl.onset = cluster
        return syllables

    if last_syl.coda:
        # previous syllable already has a coda, so this cluster is the onset of the next syllable
        new_syllable = Syllable(onset=cluster)
        syllables.append(new_syllable)
        return syllables

    # previous syllable doesn't have a coda, so this cluster might actually be
    # partially the previous coda and partially this onset
    new_coda, new_onset = onset_rules(cluster)

    if new_coda:
        last_syl.coda = new_coda

    if new_onset:
        # some of the consonants were left in this onset, so create a new syllable
        new_syllable = Syllable(onset=new_onset)
        syllables.append(new_syllable)

    return syllables


def check_last_syllable(syllables: SList) -> SList:
    """
    The syllable algorithm may assign a consonant cluster to a syllable that does not have
    a nucleus - this is not allowed in the English language.
    """

    last = syllables[-1]
    if last.nucleus:
        # there is no violation, a nucleus is present
        return syllables

    penultimate = syllables[-2]

    if last.onset:
        if penultimate.coda:
            penultimate.coda.extend(last.onset)
        else:
            penultimate.coda = last.onset
        return syllables[:-1]

    raise AttributeError(f"Couldn't fix last syllable for {syllables}")


def generate_syllables(phoneme_str: str):
    # `phoneme_str` is a string of phonemes e.g.'B IH0 K AH0 Z'

    # group phonemes into clusters
    phonemes = [parse_phonemes(ph) for ph in phoneme_str.split()]
    clusters = []
    for ph in phonemes:
        clusters = cluster_phonemes(clusters, ph)

    # group clusters into syllables
    syllables = []
    for cl in clusters:
        syllables = syllabify_clusters(syllables, cl)

    # Validate last syllable
    syllables = check_last_syllable(syllables)

    # Convert syllables to string representation
    return [str(syl) for syl in syllables]


def onset_rules(onset: Cluster):
    """
    Given a proposed onset, checks whether any of the consonants are actually
    part of the previous syllable's coda instead, and returns a tuple of (coda, onset)
    """

    # parts of the proposed onset that should really be part of the previous coda instead
    coda = Cluster()

    def move_n_to_coda(n: int = 1) -> None:
        """
        Move the first `n` (default 1) consonants from the proposed onset into the coda instead
        """
        nonlocal onset, coda
        to_move = onset.phoneme_list[:n]
        onset.phoneme_list = onset.phoneme_list[n:]
        coda.phoneme_list.extend(to_move)

    def split_on(phoneme: str) -> None:
        """
        Find `phoneme` and move it and previous to the coda, out of the onset
        """
        nonlocal onset
        move_n_to_coda(onset.find_first(phoneme) + 1)

    def split_before(phoneme: str) -> None:
        """
        Find `phoneme` and move anything before it to the coda, out of the onset.
        `phoneme` stays in the onset.
        """
        nonlocal onset
        move_n_to_coda(onset.find_first(phoneme))

    # Harley Phonotactic Rule 3: The velar nasal /NG/ never occurs in the onset of
    # a syllable.
    # -> if /NG/ is found anywhere in the onset, it and any consonants before it
    # must belong to the previous coda instead.
    # Test case: ringing
    if NG in onset:
        split_on(NG)

    # Harley Phonotactic Rule 4: The glottal fricative /HH/ never occurs in the coda
    # of a syllable.
    # -> if /HH/ occurs, it must be in the onset, so any unhandled consonants
    # before it must be in the previous coda
    if HH in onset:
        split_before(HH)

    # Harley Phonotactic Rule 5: The affricates /CH/ and /JH/, and the glottal
    # fricative /HH/ do not occur in complex onsets.
    # -> if any of these consonants occur, they and anything before them goes
    # into the previous coda instead
    if onset.is_complex:
        for ph in AFFRICATES | {HH}:
            if ph in onset.phoneme_list:
                split_on(ph)

    # Harley Phonotactic Rule 6: The first consonant in a two-consonant onset
    # must be an obstruent.
    # -> if the first consonant IS NOT an obstruent, it must belong to the
    # coda instead of to the onset
    if onset.is_complex and not onset.first.is_obstruent:
        split_on(onset.first)

    # Harley Phonotactic Rule 7: The second consonant in a two-consonant onset must
    # not be a voiced obstruent.
    # -> if second consonant IS a voiced obstruent, both the first and second consonants
    # must belong to the previous coda
    # Test case: 'amused': [/Z/, /D/] are part of the coda
    if onset.is_complex and onset.second.is_voiced_obstruent:
        split_on(onset.second)

    # Harley Phonotactic Rule 8: If the first consonant of a two-consonant onset
    # is not an /S/, the second consonant must be a liquid or a glide (i.e., it must
    # be one of {/L/, /R/, /W/, /Y/}
    # -> if the first consonant is not an /S/, and the second consonant IS NOT an
    # approximate, then both the first and second consonants must belong to the previous
    # coda
    if onset.is_complex and onset.first != S and onset.second.is_approximate:
        split_on(onset.second)

    return coda, onset
