from .LanguageModel import LanguageModel
from collections import defaultdict
class NGramLM(LanguageModel):
    """
    A n-gram language model using word counts.

    Args:
        train_data: Training data as a list of tokens.
        order: The n-gram order.
    """
    def __init__(self, train, order):
        super().__init__(set(train), order)
        self.counts = defaultdict(float)  
        self.norms = defaultdict(float)  

        for i in range(order - 1, len(train)):
            history = tuple(train[i - order + 1 : i])
            word = train[i]
            self.counts[(word,) + history] += 1.0
            self.norms[history] += 1.0

    def probability(self, word, *history):
        history = tuple(history[-(self.order - 1):])  # Trim history to match n-gram order
        return self.counts[(word,) + history] / self.norms[history] if self.norms[history] > 0 else 0.0