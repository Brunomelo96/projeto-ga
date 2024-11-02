def get_metrics(labels, pred):
    TP = ((pred == 1) & (labels == 1)).sum().item()
    FP = ((pred == 1) & (labels == 0)).sum().item()
    FN = ((pred == 0) & (labels == 1)).sum().item()

    precision = TP / (TP + FP) if TP + FP > 0 else 0
    recall = TP / (TP + FN) if TP + FN > 0 else 0
    f1 = 2 * (precision * recall) / (precision +
                                     recall) if (precision + recall) > 0 else 0

    print(precision, recall, f1, 'metrics')

    return precision, recall, f1
