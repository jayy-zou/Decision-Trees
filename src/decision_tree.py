import numpy as np

def most_common(targets):
    ones = 0
    zeros = 0
    for n in targets:
        if (n == 1):
            ones += 1
        elif (n == 0):
            zeros += 1
    if (ones > zeros):
        return 1
    else:
        return 0


def entropy(targets):
    positives = 0
    negatives = 0
    for example in targets:
        if (example == 1):
            positives += 1
        else:
            negatives += 1
    negatives = np.size(targets) - positives
    if (positives == 0 or negatives == 0):
        return 0
    else:
        return -positives / (positives + negatives) * np.log2(positives / (positives + negatives)) - negatives / (
                positives + negatives) * np.log2(negatives / (positives + negatives))


def trim(features, index):
    return np.delete(features, index, axis=1)


class Tree():
    def __init__(self, value=None, attribute_name="root", attribute_index=None, branches=None):
        self.branches = [] if branches is None else branches
        self.attribute_name = attribute_name
        self.attribute_index = attribute_index
        self.value = value


class DecisionTree():
    def __init__(self, attribute_names):
        self.attribute_names = attribute_names
        self.tree = None
        return

    def _check_input(self, features):
        if features.shape[1] != len(self.attribute_names):
            raise ValueError(
                "Number of features and number of attribute names must match!", self.attribute_names
            )


    def ID3(self, features, targets, attribute_names, new_tree):

        root=Tree()

        class_count = np.size(targets)
        positives = 0
        negatives = 0
        for i in targets:
            if (i > 0):
                positives += 1
            else:
                negatives+=1

        if (positives==class_count):
            root.value = 1
            return root

        elif (negatives==class_count):
            root.value = 0
            return root

        elif (len(attribute_names) == 0):
            root.value = most_common(targets)
            return root

        else:
            left_tree=Tree()
            right_tree=Tree()

            infogain = []

            for i in range(len(attribute_names)):
                infogain.append(information_gain(features, i, targets))


            winner_index = np.argmax(infogain)

            new_tree.attribute_name=attribute_names[winner_index]
            new_tree.attribute_index=self.attribute_names.index(attribute_names[winner_index])

            attributes = attribute_names.copy()
            attributes.remove(attribute_names[winner_index])

            new_tree.branches.append(self.ID3(trim(features, winner_index)[np.nonzero(features[:, winner_index] == 0), :][0], targets[np.nonzero(features[:, winner_index] == 0)], attributes, left_tree))
            new_tree.branches.append(self.ID3(trim(features, winner_index)[np.nonzero(features[:, winner_index] == 1), :][0], targets[np.nonzero(features[:, winner_index] == 1)], attributes, right_tree))

            return new_tree


    def fit(self, features, targets):
        self._check_input(features)

        self.tree = Tree()
        self.tree = self.ID3(features, targets, self.attribute_names, self.tree)
        # self.visualize(self.tree)

    def predict(self, features):
        self._check_input(features)

        prediction = np.empty([])

        for row in features:
            prediction = np.append(prediction, self.eval(row, self.tree))
        prediction = np.delete(prediction, 0)
        return prediction

    def eval(self, feature, tree):
        if (len(tree.branches) == 0):
            return tree.value
        else:
            if (feature[tree.attribute_index] == 0):
                return self.eval(feature, tree.branches[0])
            else:
                return self.eval(feature, tree.branches[1])

    def _visualize_helper(self, tree, level):
        """
        Helper function for visualize a decision tree at a given level of recursion.
        """
        tab_level = "  " * level
        val = tree.value if tree.value is not None else 0
        print("%d: %s%s == %f" % (level, tab_level, tree.attribute_name, val))

    def visualize(self, branch=None, level=0):
        """
        Visualization of a decision tree. Implemented for you to check your work and to
        use as an example of how to use the given classes to implement your decision
        tree.
        """
        if not branch:
            branch = self.tree
        self._visualize_helper(branch, level)

        for branch in branch.branches:
            self.visualize(branch, level + 1)


def information_gain(features, attribute_index, targets):
    S_before = entropy(targets)
    positives = 0
    negatives = 0

    true_targets = []
    false_targets = []

    index = 0

    for row in features:
        if (row[attribute_index] == 1):
            true_targets.append(targets[index])
            positives += 1
        else:
            false_targets.append(targets[index])
            negatives += 1
        index += 1
    if (negatives==0 or positives==0):
        info_gain=0
    else:
        info_gain = S_before - ((positives / (positives + negatives)) * entropy(true_targets) + (
                negatives / (positives + negatives)) * entropy(false_targets))
    return info_gain


if __name__ == '__main__':
    attribute_names = ['larry', 'curly', 'moe']
    decision_tree = DecisionTree(attribute_names=attribute_names)
    while len(attribute_names) > 0:
        attribute_name = attribute_names[0]
        if not decision_tree.tree:
            decision_tree.tree = Tree(
                attribute_name=attribute_name,
                attribute_index=decision_tree.attribute_names.index(attribute_name),
                value=0,
                branches=[]
            )
        else:
            decision_tree.tree.branches.append(
                Tree(
                    attribute_name=attribute_name,
                    attribute_index=decision_tree.attribute_names.index(attribute_name),
                    value=0,
                    branches=[]
                )
            )
        attribute_names.remove(attribute_name)
    decision_tree.visualize()
