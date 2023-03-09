import numpy as np

from sklearn.datasets import make_blobs
from sklearn.cluster import KMeans
from sklearn.metrics import calinski_harabasz_score, davies_bouldin_score

from matplotlib import pyplot as plt
import seaborn as sns

from federated_clustering_validation import calinski_harabasz as fed_ch


c1_centers = np.array([[0.0, 0.0],
                       [6.0, 2.0],
                       [3.0, 9.0]])
c2_centers = np.array([[6.5, 2.5],
                       [12.0, 12.0]])
all_centers = np.concatenate((c1_centers, c2_centers))

c1_x, c1_y = make_blobs(n_samples=600, n_features=2, centers=c1_centers)
c2_x, c2_y = make_blobs(n_samples=200, n_features=2, centers=c2_centers)
client_X = [c1_x, c2_x]
all_x = np.concatenate(client_X)

fig, ax = plt.subplots()
sns.scatterplot(x=c1_x[:,0], y=c1_x[:,1], linewidth=0, ax=ax)
sns.scatterplot(x=c2_x[:,0], y=c2_x[:,1], linewidth=0, ax=ax)
ax.scatter(all_centers[:,0], all_centers[:,1], color="k", marker="x")
fig.show()

scores_ch_scikit = []
scores_ch_this = []
for kg in range(2, 10):
    _km = KMeans(n_clusters=kg).fit(all_x)
    scores_ch_scikit.append(calinski_harabasz_score(all_x, _km.labels_))
    scores_ch_this.append(fed_ch.calinski_harabasz_score_centralized(all_x, _km.labels_))

assert np.allclose(np.array(scores_ch_scikit), np.array(scores_ch_this))

fig, ax = plt.subplots()
sns.lineplot(x=list(range(2, 10)), y=scores_ch_this, markers=True, ax=ax)
fig.show()


# Simulate a global clustering result
kmeans_glob = KMeans(n_clusters=4).fit(all_x)
global_cluster_centers = kmeans_glob.cluster_centers_
global_cluster_labels = kmeans_glob.labels_
print(calinski_harabasz_score(all_x, global_cluster_labels))

fig, ax = plt.subplots()
sns.scatterplot(x=c1_x[:,0], y=c1_x[:,1], linewidth=0, ax=ax)
sns.scatterplot(x=c2_x[:,0], y=c2_x[:,1], linewidth=0, ax=ax)
ax.scatter(all_centers[:,0], all_centers[:,1], color="k", marker="x")
ax.scatter(global_cluster_centers[:,0], global_cluster_centers[:,1], color="r", marker="*")
for k in range(4):
    ax.text(global_cluster_centers[k,0], global_cluster_centers[k,1], s=f"C-{k}", color="red")
fig.show()

# Share the global clustering to the clients
client_labels = []
for client in client_X:
    c_km = KMeans(n_clusters=4, init=global_cluster_centers, n_init=1).fit(global_cluster_centers)
    client_labels.append(c_km.predict(client))

fed_ch.calinski_harabasz_score_federated(client_X, client_labels, global_cluster_centers, True)
fed_ch.calinski_harabasz_score_centralized(all_x, global_cluster_labels, True)

assert np.isclose(fed_ch.calinski_harabasz_score_federated(client_X, client_labels, global_cluster_centers), fed_ch.calinski_harabasz_score_centralized(all_x, global_cluster_labels))
