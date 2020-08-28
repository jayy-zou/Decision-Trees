from .decision_tree import DecisionTree
from .prior_probability import PriorProbability
from .metrics import precision_and_recall, confusion_matrix, f1_measure, accuracy
from .data import load_data, train_test_split

def run(data_path, learner_type, fraction):
    path=data_path

    features, targets, attribute_names=load_data(path)

    if(learner_type=="decision_tree"):
        model=DecisionTree(attribute_names)
    else:
        model=PriorProbability()

    train_features, train_targets, test_features, test_targets = train_test_split(features, targets, fraction)

    model.fit(train_features, train_targets)

    predictions=model.predict(test_features)

    myprecision, myrecall= precision_and_recall(test_targets, predictions)
    confusionmatrix=confusion_matrix(test_targets, predictions)
    myaccuracy= accuracy(test_targets, predictions)
    f1measure=f1_measure(test_targets, predictions)

    return confusionmatrix, myaccuracy, myprecision, myrecall, f1measure

