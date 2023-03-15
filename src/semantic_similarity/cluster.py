from collections import defaultdict

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn import cluster
from sklearn.decomposition import PCA


def get_embedding_clusters(category_mean_embed, num_clusters=None, distance_threshold=None):
    mean_matrix = np.asarray(list(category_mean_embed.values()))
    # cluster_model = cluster.AgglomerativeClustering(
    #     n_clusters=num_clusters, compute_distances=True, distance_threshold=distance_threshold
    # )
    cluster_model = cluster.KMeans(n_clusters=num_clusters)
    clustering = cluster_model.fit(mean_matrix)
    return clustering


def get_pca_embeddings(category_mean_embed):
    mean_matrix = np.asarray(list(category_mean_embed.values()))
    embeddings_pca = PCA(n_components=2).fit_transform(mean_matrix)
    return embeddings_pca


def save_pca_plot(embedding_pca, cluster_labels, run_label):
    df = combine_pca_labels_as_df(embedding_pca, cluster_labels)
    unique_labels = np.unique(cluster_labels)
    for i in unique_labels:
        plt.scatter(df[df.label == i]["dim1"], df[df.label == i]["dim2"], label=i)
    plt.legend()
    plt.savefig(f"plots/pca_plot_{run_label}.png")
    # plt.show()
    plt.clf()


def combine_pca_labels_as_df(pca, labels):
    labels = np.expand_dims(labels, axis=-1)
    combined = np.concatenate((pca, labels), axis=-1)
    return pd.DataFrame(combined, columns=["dim1", "dim2", "label"])


def get_cluster_to_category(category_dict, labels):
    cat_to_cluster = {cat: int(clust) for cat, clust in zip(category_dict.keys(), labels)}
    cluster_to_cat = defaultdict(list)
    for cat, cluster in cat_to_cluster.items():
        cluster_to_cat[cluster].append(cat)
    return cluster_to_cat
