from .LanguageModel import LanguageModel
class InterpolatedLM(LanguageModel):
    def __init__(self, main, backoff, alpha):
        super().__init__(main.vocab, main.order)
        self.main = main
        self.backoff = backoff
        self.alpha = alpha
        self.vocab = main.vocab
        self.order = main.order
    def probability(self, word, *history):
        return self.alpha * self.main.probability(word,*history) + \
               (1.0 - self.alpha) * self.backoff.probability(word,*history)
    
    # this is for making sure that the probabilites are well-formed
    def sum_probabilities(self):
        probs = sorted([(word,self.probability(word,*())) for word in self.vocab], key=lambda x:x[1], reverse=True)
        probabilities = [prob for _, prob in probs]
        return sum(probabilities)
