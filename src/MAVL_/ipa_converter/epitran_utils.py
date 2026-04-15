import os
import epitran
import langcodes
from pathlib import Path


def get_valid_epitran_mappings_list():
    map_path = Path(epitran.__path__[0]) / "data" / "map"
    map_files = map_path.glob("*.*")
    valid_mappings = [map_file.stem for map_file in map_files]
    valid_mappings.append("cmn-Hans")
    valid_mappings.append("cmn-Hant")

    problem_mappings = [
        "generic-Latn",
        "tur-Latn-bab",
        "ood-Latn-sax",
        "vie-Latn-so",
        "vie-Latn-ce",
        "vie-Latn-no",
        "kaz-Cyrl-bab",
    ]

    return [mapping for mapping in valid_mappings if mapping not in problem_mappings]


def get_epitran(selected_mapping):
    if selected_mapping in {"cmn-Hans", "cmn-Hant"}:
        data_dir = os.path.expanduser("~/epitran_data/")
        if not os.path.exists(data_dir):
            print("Chinese requires a special dictionary. Downloading now")
            epitran.download.cedict()

    return epitran.Epitran(selected_mapping)
