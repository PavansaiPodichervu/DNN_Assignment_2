import numpy as np
from sklearn.metrics import accuracy_score

cm = np.loadtxt(
    "test_confusion_matrix.csv",
    delimiter=","
)

print("Confusion matrix shape:", cm.shape)
print("Total samples:", cm.sum())
print("Correct predictions:", np.trace(cm))
print("Accuracy from matrix:", np.trace(cm) / cm.sum())

assert cm.shape == (6, 6)
assert cm.sum() == 2947

print("Confusion matrix verification passed.")