import os
import csv

AO = "AO"
UW = "UW"
EH = "EH"
AH = "AH"
AA = "AA"
IY = "IY"
IH = "IH"
UH = "UH"
AE = "AE"
AW = "AW"
AY = "AY"
ER = "ER"
EY = "EY"
OW = "OW"
OY = "OY"

VOWELS = [
    AO,
    UW,
    EH,
    AH,
    AA,
    IY,
    IH,
    UH,
    AE,
    AW,
    AY,
    ER,
    EY,
    OW,
    OY,
]
""" consonant phonemes """

CH = "CH"
DH = "DH"
HH = "HH"
JH = "JH"
NG = "NG"
SH = "SH"
TH = "TH"
ZH = "ZH"
Z = "Z"
S = "S"
P = "P"
R = "R"
K = "K"
L = "L"
M = "M"
N = "N"
F = "F"
G = "G"
D = "D"
B = "B"
T = "T"
V = "V"
W = "W"
Y = "Y"

CONSONANTS = [
    CH,
    DH,
    HH,
    JH,
    NG,
    SH,
    TH,
    ZH,
    Z,
    S,
    P,
    R,
    K,
    L,
    M,
    N,
    F,
    G,
    D,
    B,
    T,
    V,
    W,
    Y,
]

PHONEME_FILE_NAME = "arpa_phonemes.csv"
FOLDER_ROOT = os.path.dirname(os.path.abspath(__file__))
DICT_PATH = "CMU_dictionary"
PHONEME_PATH = os.path.join(FOLDER_ROOT, DICT_PATH, PHONEME_FILE_NAME)

arpa = {}
with open(PHONEME_PATH, "r") as ph_file:
    header = ph_file.readline().strip().split(",")
    reader = csv.DictReader(ph_file, header)
    for row in reader:
        arpa[row["PHONEME"]] = row


def make_set(key, value):
    return {ph for ph, data in arpa.items() if value.upper() in data[key.upper()]}


AFFRICATES = make_set("airstream mechanism", "affricate")
PLOSIVES = make_set("airstream mechanism", "plosive")
FRICATIVES = make_set("airstream mechanism", "fricative")

OBSTRUENTS = PLOSIVES | FRICATIVES
VOICED = make_set("voice", "voiced")
VOICED_OBSTRUENTS = VOICED & OBSTRUENTS

APPROXIMANTS = make_set("airstream mechanism", "approximant")
LIQUIDS = make_set("class", "liquid")
GLIDES = APPROXIMANTS - LIQUIDS
