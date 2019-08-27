from tiny_tokenizer.tiny_tokenizer_token import Token
from tiny_tokenizer.word_tokenizers.tokenizer import BaseTokenizer


class SudachiTokenizer(BaseTokenizer):
    """Wrapper class for SudachiPy."""

    def __init__(self, mode: str, with_postag: bool, **kwargs):
        """
        Initializer for SudachiTokenizer

        Parameters
        ---
        mode (str)
            Splitting mode which controls a granuality oftiny_tokenizer.token.
            (mode should be `A`, `B` or `C`)
            For more information, see following links.
            - document: https://github.com/WorksApplications/Sudachi#the-modes-of-splitting  # NOQA
            - paper: http://www.lrec-conf.org/proceedings/lrec2018/summaries/8884.html  # NOQA
        with_postag (bool=False)
            flag determines iftiny_tokenizer.tokenizer include pos tags.
        **kwargs
            others.
        """
        super(SudachiTokenizer, self).__init__(f"sudachi ({mode})")
        try:
            from sudachipy import tokenizer
            from sudachipy import dictionary
        except ModuleNotFoundError:
            raise ModuleNotFoundError("sudachipy is not installed")
        try:
            self.tokenizer = dictionary.Dictionary().create()
        except KeyError:
            msg = "please install dictionary"
            msg += " ( see https://github.com/WorksApplications/SudachiPy#install-dict-packages )"  # NOQA
            raise KeyError(msg)

        _mode = mode.capitalize()
        if _mode == "A":
            self.mode = tokenizer.Tokenizer.SplitMode.A
        elif _mode == "B":
            self.mode = tokenizer.Tokenizer.SplitMode.B
        elif _mode == "C":
            self.mode = tokenizer.Tokenizer.SplitMode.C
        else:
            msg = "Invalid mode is specified. Mode should be 'A', 'B' or 'C'"
            raise ValueError(msg)

        self.with_postag = with_postag

    def tokenize(self, text: str):
        """Tokenize."""
        result = []
        for token in self.tokenizer.tokenize(text, self.mode):
            _token = Token(token.surface())
            if self.with_postag:
                _token.postag, *_postag2 = token.part_of_speech()
                _token.postag2 = "\t".join(_postag2)
            result.append(_token)
        return result
