"""Test for word tokenizers"""
import unittest

import pytest

from tiny_tokenizer.tiny_tokenizer_token import Token
from tiny_tokenizer.word_tokenizer import WordTokenizer

SENTENCE1 = "吾輩は猫である"
SENTENCE2 = "医薬品安全管理責任者"


class SudachiTokenizerSegmentationTest(unittest.TestCase):
    def test_word_tokenize_with_sudachi_mode_a(self):
        """Test Sudachi tokenizer."""
        try:
            tokenizer = WordTokenizer(tokenizer="Sudachi", mode="A")
        except ModuleNotFoundError:
            pytest.skip("skip sudachi")

        expect = [Token(surface=w) for w in "医薬 品 安全 管理 責任 者".split(" ")]
        result = tokenizer.tokenize(SENTENCE2)
        self.assertEqual(expect, result)

    def test_word_tokenize_with_sudachi_mode_b(self):
        """Test Sudachi tokenizer."""
        try:
            tokenizer = WordTokenizer(tokenizer="Sudachi", mode="B")
        except ModuleNotFoundError:
            pytest.skip("skip sudachi")

        expect = [Token(surface=w) for w in "医薬品 安全 管理 責任者".split(" ")]
        result = tokenizer.tokenize(SENTENCE2)
        self.assertEqual(expect, result)

    def test_word_tokenize_with_sudachi_mode_c(self):
        """Test Sudachi tokenizer."""
        try:
            tokenizer = WordTokenizer(tokenizer="Sudachi", mode="C")
        except ModuleNotFoundError:
            pytest.skip("skip sudachi")

        expect = [Token(surface=w) for w in "医薬品安全管理責任者".split(" ")]
        result = tokenizer.tokenize(SENTENCE2)
        self.assertEqual(expect, result)


class SudachiTokenizerPostaggingTest(unittest.TestCase):
    """Test ordinal word tokenizer."""

    def test_word_tokenize_with_sudachi_mode_a(self):
        """Test Sudachi tokenizer."""
        try:
            tokenizer = WordTokenizer(
                tokenizer="sudachi", mode="A", with_postag=True)
        except ModuleNotFoundError:
            pytest.skip("skip sudachi")

        words = "医薬 品 安全 管理 責任 者".split(" ")  # NOQA
        postags = "名詞 接尾辞 名詞 名詞 名詞 接尾辞".split(" ")

        expect = [Token(surface=w, postag=p) for w, p in zip(words, postags)]
        result = tokenizer.tokenize(SENTENCE2)
        import IPython; IPython.embed()
        self.assertEqual(expect, result)
