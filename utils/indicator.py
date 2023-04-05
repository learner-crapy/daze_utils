class Metrics:
    def __init__(self, trues, preds):
        self.trues = trues
        self.preds = preds

    def accuracy(self):
        return sum([1 for i in range(len(self.trues)) if self.trues[i] == self.preds[i]]) / len(self.trues)

    def precision(self):
        tp = sum([1 for i in range(len(self.trues)) if self.trues[i] == 1 and self.preds[i] == 1])
        fp = sum([1 for i in range(len(self.trues)) if self.trues[i] == 0 and self.preds[i] == 1])
        return tp / (tp + fp)

    def recall(self):
        tp = sum([1 for i in range(len(self.trues)) if self.trues[i] == 1 and self.preds[i] == 1])
        fn = sum([1 for i in range(len(self.trues)) if self.trues[i] == 1 and self.preds[i] == 0])
        return tp / (tp + fn)

    def f1(self):
        p = self.precision()
        r = self.recall()
        return 2 * p * r / (p + r)

# example
# trues = [1, 0, 1, 0, 1, 0, 0]
# preds = [1, 1, 0, 0, 1, 0, 0]
