from .LanguageModel import LanguageModel
class AdjustedLaplaceLM(LanguageModel):
    """
    A language model with adjusted Laplace smoothing.

    Args:
        base_lm: The base language model (e.g., n-gram model).
        alpha: The smoothing parameter.
    """
    def __init__(self, base_lm, alpha):
        super().__init__(base_lm.vocab, base_lm.order)
        self.base_lm = base_lm
        self.alpha = alpha
        self.eps = 1e-6  

    def counts(self, word_and_history):
        history = word_and_history[1:]
        word = word_and_history[0]
        if word not in self.vocab:
            return 0.0
        base_count = self.base_lm.counts[(word,) + history]
        norm = self.base_lm.norms[history] + self.alpha * len(self.vocab)
        return (base_count + self.alpha) / (norm + self.eps)

    def norm(self, history):
        return self.base_lm.norms[history] + self.eps 

    def probability(self, word, *history):
        """Calculate the smoothed probability of a word given history."""
        word_and_history = (word,) + tuple(history)
        return self.counts(word_and_history)
