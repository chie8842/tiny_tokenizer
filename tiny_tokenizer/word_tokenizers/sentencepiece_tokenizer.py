from tiny_tokenizer.tiny_tokenizer_token import Token
from tiny_tokenizer.word_tokenizers.tokenizer import BaseTokenizer


class SentencepieceTokenizer(BaseTokenizer):
    """Wrapper class forSentencepiece"""

    def __init__(self, model_path: str, **kwargs):
        """
        Initializer for SentencepieceTokenizer.

        Parameters
        ---
        model_path (str)
            path to sentencepiece model.
        **kwargs
            others.
        """
        super(SentencepieceTokenizer, self).__init__("sentencepiece")
        try:
            import sentencepiece
        except ModuleNotFoundError:
            raise ModuleNotFoundError("sentencepiece is not installed")

        self.tokenizer = sentencepiece.SentencePieceProcessor()
        self.tokenizer.load(model_path)

    def tokenize(self, text: str):
        result = []
        for subword in self.tokenizer.EncodeAsPieces(text):
            token = Token(surface=subword)
            result.append(token)
        return result
