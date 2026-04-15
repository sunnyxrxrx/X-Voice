import os
import re
from collections import defaultdict
from typing import List, Dict, Optional

CMU_PATTERN = re.compile(
    r"(?P<Word>'?\w+[^()]*)(?P<Alt>\(\d+\))?\s\s(?P<Phoneme>[^\n]+)"
)

CMU_DIR = "CMU_dictionary"
VERSION = "cmudict.0.7a"
FOLDER_ROOT = os.path.dirname(os.path.abspath(__file__))
DICT_PATH = os.path.join(FOLDER_ROOT, CMU_DIR, VERSION)


class CMUDictionary:
    def __init__(self):
        if not os.path.exists(DICT_PATH):
            raise IOError(f"Could not read in {DICT_PATH}")

        self._cmudict: Dict[str, List] = defaultdict(list)
        with open(DICT_PATH) as dict_file:
            for line in dict_file.readlines():
                match = re.match(CMU_PATTERN, line)
                if not match or not match.group("Word"):
                    continue
                self._cmudict[match.group("Word")].append(match.group("Phoneme"))

    def get(self, key, default=None) -> List[str]:
        return self._cmudict.get(key.upper(), default)

    def get_first(self, key, default=None) -> Optional[str]:
        phonemes = self._cmudict.get(key.upper(), default)
        if phonemes:
            return phonemes[0]
        return phonemes

    def __getitem__(self, key):
        try:
            return self._cmudict[key.upper()]
        except (KeyError, UnicodeDecodeError):
            return None
