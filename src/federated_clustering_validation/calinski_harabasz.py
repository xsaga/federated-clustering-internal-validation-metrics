import numpy as np
from scipy.spatial import distance
from sklearn.preprocessing import LabelEncoder
from typing import List

def calinski_harabasz_score_centralized(X: np.ndarray, labels: np.ndarray, debug=False) -> float:
    n_samples = X.shape[0]

    le = LabelEncoder()
    labels = le.fit_transform(labels)
    n_labels = len(le.classes_)

    center = np.mean(X, axis=0)
    W = 0
    B = 0

    if debug:
        print("Number of total samples: ", n_samples)
        print("Number of total labels: ", n_labels)
        print("Center of total dataset: ", center)

    for k in range(n_labels):
        X_k = X[labels == k]
        center_k = np.mean(X_k, axis=0)
        W += np.sum(distance.cdist(X_k,
                                  center_k.reshape(1,-1),
                                  metric="sqeuclidean"))
        B += (X_k.shape[0]) * distance.sqeuclidean(center_k, center)

        if debug:
            print(f"Cluster {k}: points in cluster = {X_k.shape[0]}; W = {W}; B = {B}")

    return 1.0 if W == 0 else (B / W) * ((n_samples - n_labels) / (n_labels - 1))


def calinski_harabasz_score_federated(X_clients: List[np.ndarray], labels_clients: List[np.ndarray], global_centers: np.ndarray, debug=False) -> float:
    """Compute the Calinski-Harabasz score in a Federated network.
    X_clients is local to each client, the server does not have this data.
    labels_clients is local to each client, the server does not have this data.
    global_centers is known to clients and the central server (output of federated clustering).
    """
    assert len(X_clients) == len(labels_clients)
    n_clients = len(X_clients)

    # server already knows total number of labels (clusters)
    n_labels = np.unique(np.concatenate(labels_clients)).shape[0]

    # compute total number of data samples
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    _n_samples_clients = []
    for c in range(n_clients):
        # local client computation
        _n_samples_c = X_clients[c].shape[0]
        # send to server
        _n_samples_clients.append(_n_samples_c)

    # server computation
    n_samples = sum(_n_samples_clients)

    # compute distributed dataset center
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    _sum_clients = []
    for c in range(n_clients):
        # local client computation
        _sum_c = np.sum(X_clients[c], axis=0)
        # send to server
        _sum_clients.append(_sum_c)

    # server computation
    center = np.sum(np.array(_sum_clients), axis=0) / n_samples

    W = 0
    B = 0

    if debug:
        print("Number of total samples: ", n_samples)
        print("Number of total labels: ", n_labels)
        print("Center of total dataset: ", center)

    for k in range(n_labels):
        _W_clients = []
        _n_cluster = []
        # local computation
        for c in range(n_clients):
            X_k_c = X_clients[c][labels_clients[c] == k]
            # use the global center for this cluster, do not recompute this value
            # because for each client it will be different.
            center_k = global_centers[k, :]
            _W_clients.append(np.sum(distance.cdist(X_k_c,
                                                    center_k.reshape(1, -1),
                                                    metric="sqeuclidean")))
            _n_cluster.append(X_k_c.shape[0])

        # server computation
        W += sum(_W_clients)
        B += (sum(_n_cluster)) * distance.sqeuclidean(global_centers[k, :], center)

        if debug:
            print(f"Cluster {k}: points in cluster = {sum(_n_cluster)}; W = {W}; B = {B}")

    return 1.0 if W == 0 else (B / W) * ((n_samples - n_labels) / (n_labels - 1))
