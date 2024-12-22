from .LanguageModel import LanguageModel
class UniformLM(LanguageModel):
    """
    A uniform language model that assigns equal probability to all words in the vocabulary.
    """
    def __init__(self, vocab):
        super().__init__(vocab, 1)

    def probability(self, word, *history):
        return 1.0 / len(self.vocab) if word in self.vocab else 0.0