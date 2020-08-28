import numpy as np

class PriorProbability():
    def __init__(self):
        self.most_common_class = None

    def fit(self, features, targets):
        ones=0
        zeros=0
        for n in targets:
            if (n==1):
                ones+=1
            elif (n==0):
                zeros+=1
        if (ones>zeros):
            self.most_common_class=1
        else:
            self.most_common_class=0

    def predict(self, data):
        answer=np.array([])
        for n in data:
            answer=np.append(answer, self.most_common_class)
        return answer
