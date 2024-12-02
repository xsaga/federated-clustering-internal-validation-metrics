import numpy as np
from scipy.spatial import distance
from sklearn.preprocessing import LabelEncoder
from typing import List

def davies_bouldin_score_centralized(X: np.ndarray, labels: np.ndarray, debug=False) -> float:
    """Davies-Bouldin score, notation from wikipedia."""
    le = LabelEncoder()
    labels = le.fit_transform(labels)
    n_labels = len(le.classes_)

    S = []
    A = []
    for k in range(n_labels):
        X_k = X[labels == k]
        center_k = np.mean(X_k, axis=0)
        A.append(center_k)
        T_i = X_k.shape[0]
        S.append(1/T_i * np.sum(distance.cdist(X_k,
                                               center_k.reshape(1,-1),
                                               metric="euclidean")))
    DB = 0
    for i in range(n_labels):
        D_i = max([(S[i] + S[j])/distance.euclidean(A[i],
                                                    A[j])
                   for j in range(n_labels) if j != i])
        DB += D_i
    return DB/n_labels
