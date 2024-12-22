from math import exp, log
class LanguageModel:
    """
    Base class for language models.

    Args:
        vocab: The vocabulary for the model.
        order: The n-gram order.
    """
    def __init__(self, vocab, order):
        self.vocab = vocab
        self.order = order

    def perplexity(self, data):
        """Calculate the perplexity of the model on a dataset."""
        log_prob = 0.0
        for i in range(self.order - 1, len(data)):
            history = data[i - self.order + 1 : i]
            word = data[i]
            p = self.probability(word, *history)
            log_prob += log(p) if p > 0 else float("-inf")
        return exp(-log_prob / (len(data) - self.order + 1))
    
    def probability(self, word, *history):
        """Calculate the probability of a word given history."""
        raise NotImplementedError("This method must be implemented by subclasses")