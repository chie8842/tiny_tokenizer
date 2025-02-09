from tiny_tokenizer import SentenceTokenizer
from tiny_tokenizer import WordTokenizer


if __name__ == "__main__":
    sentence_tokenizer = SentenceTokenizer()
    tokenizers = ["MeCab", "KyTea", "Character"]
    tokenizers_support_postag = ["MeCab", "KyTea"]

    word_tokenizers = []
    for tokenizer in tokenizers:
        try:
            _tokenizer = WordTokenizer(tokenizer)
            word_tokenizers.append(_tokenizer)

            if tokenizer in tokenizers_support_postag:
                _tokenizer = WordTokenizer(tokenizer, with_postag=True)
                word_tokenizers.append(_tokenizer)

        except ModuleNotFoundError:
            print("Skip: ", tokenizer)

    try:
        _tokenizer = WordTokenizer("Sentencepiece", model_path="./data/model.spm")  # NOQA
        word_tokenizers.append(_tokenizer)

    except ModuleNotFoundError:
        print("Skip: ", "Sentencepiece")

    try:
        _tokenizer = WordTokenizer("Sudachi", mode="A", with_postag=True)
        word_tokenizers.append(_tokenizer)

    except ModuleNotFoundError:
        print("Skip: ", "Sudachi")

    print("Finish creating word tokenizers")
    print()

    # ref: https://ja.wikipedia.org/wiki/東京特許許可局
    document = "東京特許許可局（とうきょうとっきょきょかきょく） 日本語の早口言葉。"  # NOQA
    document += "なお実際に特許に関する行政を行うのは特許庁であり、過去にこのような役所が存在したことは一度も無い。"  # NOQA
    print(f"Given document: {document}")

    sentences = sentence_tokenizer.tokenize(document)
    for sentence_id, sentence in enumerate(sentences):
        print(f"#{sentence_id}: {sentence}")

        for tokenizer in word_tokenizers:
            print(f"Tokenizer: {tokenizer.name}")
            result = tokenizer.tokenize(sentence)
            result = [str(r) for r in result]
            print(' '.join(result))

        print()
