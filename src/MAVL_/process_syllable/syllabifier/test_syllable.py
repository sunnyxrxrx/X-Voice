import os
import unittest
import random
from typing import List

from . import syllable3

ROOT = os.path.dirname(os.path.abspath(__file__))
TEST_CASE_PATH = os.path.join(ROOT, "test_cases.csv")


class TestSyllables(unittest.TestCase):
    test_cases = []

    def test_one_syllable_correct(self):
        words = [
            "once",
            "twice",
            "there",
            "their",
            "Cap",
            "Caps",
            "Poop",
            "Fuck",
            "texts",
            "tree",
            "trick",
            "tricked",
            "plan",
            "planned",
            "quay",
            "queue",
            "rang",
            "pinged",
            "stream",
            "stew",
            "sprawl",
            "splat",
            "scream",
            "can't",
            "through",
            "spleen",
            "lamp",
            "svelte",
            "sphinx",
            "owl",
            "our",
        ]

        for word in words:
            with self.subTest(word=word):
                self.assertEqual(syllable3.num_syllables(word), 1)

    def test_one_syllable_wrong(self):
        wrong_words = ["always", "Fucking"]
        for word in wrong_words:
            with self.subTest(word=word):
                self.assertNotEqual(syllable3.num_syllables(word), 1)

    def test_two_syllables_correct(self):
        words = [
            "digging",
            "boogie",
            "happy",
            "ringing",
            "tricking",
            "describe",
            "attract",
            "playground",
            "amused",
            "tingle",
            "rhythm",
            "sputum",
            "squirrel",
            "asthma",
            "hatchet",
            "lawful",
            "papa",
            "doghouse",
            "behave",
            "inhale",
            "attempt",
            "hangman",
            "lamprey",
            "complex",
            "describe",
            "gewgaw",
            "guava",
            "little",
            "belfry",
            "hello",
            "onion",
            "endless",
            "undress",
            "heartbreak",
            "toothbrush",
            "handbag",
            "handling",
        ]

        for word in words:
            with self.subTest(word=word):
                self.assertEqual(syllable3.num_syllables(word), 2, f"Failed on {word}")

    def test_two_syllables_incorrect(self):
        words = ["abominable", "testosterone"]
        for word in words:
            with self.subTest(word=word):
                self.assertNotEqual(word, 2)

    def test_three_syllables_correct(self):
        words = ["sclerosis", "textual", "grandmother", "resources"]

        for word in words:
            with self.subTest(word=word):
                self.assertEqual(
                    act := syllable3.num_syllables(word),
                    3,
                    f"Expected {word} to have 3 syllables, but had {act}",
                )

    def test_four_syllables_correct(self):
        words = ["therapeutic"]

        for word in words:
            with self.subTest(word=word):
                self.assertEqual(
                    act := syllable3.num_syllables(word),
                    4,
                    f"Expected {word} to have 4 syllables, but had {act}",
                )

    def x_random_words(self, x: int) -> List[str]:
        if not self.test_cases:
            with open(TEST_CASE_PATH, "r") as csv_file:
                for line in csv_file.readlines():
                    word, syllables = line.strip().split(",")
                    syllables = int(syllables)
                    self.test_cases.append((word, syllables))
        random.shuffle(self.test_cases)
        return self.test_cases[:x]

    def assert_word_has_syllables(self, word: str, syllables: int):
        self.assertEqual(
            act := syllable3.num_syllables(word),
            syllables,
            f"Expected {word} to have {syllables} syllables but found {act}",
        )

    def test_random_words(self):
        lines = self.x_random_words(5000)
        for word, syllables in lines:
            with self.subTest(word=word, syllables=syllables):
                self.assert_word_has_syllables(word, syllables)


if __name__ == "__main__":
    unittest.main()
