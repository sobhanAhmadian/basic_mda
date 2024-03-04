import numpy as np
from sklearn.metrics import (
    f1_score,
    precision_score,
    recall_score,
    accuracy_score,
    matthews_corrcoef,
    auc,
    roc_curve,
    precision_recall_curve,
)


class Result:
    def __init__(self) -> None:
        self.auc = 0
        self.acc = 0
        self.f1 = 0
        self.aupr = 0
        self.loss = 0

    def get_result(self):
        return {
            "AUC": self.auc,
            "ACC": self.acc,
            "F1 Score": self.f1,
            "AUPR": self.aupr,
            "Loss": self.loss,
        }

    def add(self, result):
        self.auc = self.auc + result.auc
        self.acc = self.acc + result.acc
        self.f1 = self.f1 + result.f1
        self.aupr = self.aupr + result.aupr
        self.loss = self.loss + result.loss

    def divide(self, k):
        self.auc = self.auc / k
        self.acc = self.acc / k
        self.f1 = self.f1 / k
        self.aupr = self.aupr / k
        self.loss = self.loss / k


def get_prediction_results(y_test, y_predict, threshold):
    result = Result()

    binary_predict = np.where(np.array(y_predict) >= threshold, 1, 0).tolist()

    result.acc = accuracy_score(y_test, binary_predict)
    result.f1 = f1_score(y_test, binary_predict, average="macro", zero_division=1)
    result.recall = recall_score(
        y_test, binary_predict, average="macro", zero_division=1
    )
    result.precision = precision_score(
        y_test, binary_predict, average="macro", zero_division=1
    )
    result.mcc = matthews_corrcoef(y_test, binary_predict)

    precision, recall, thresholds = precision_recall_curve(y_test, y_predict)
    numerator = 2 * recall * precision
    denominator = recall + precision
    f1_scores = np.divide(
        numerator, denominator, out=np.zeros_like(denominator), where=(denominator != 0)
    )
    result.max_f1 = np.max(f1_scores)

    fpr, tpr, thresholds = roc_curve(y_test, y_predict, pos_label=1)
    result.auc = auc(fpr, tpr)
    result.tpr = tpr
    result.fpr = fpr

    return result
