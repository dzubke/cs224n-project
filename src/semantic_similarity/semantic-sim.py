# standard libs
import argparse
from collections import defaultdict
import json
from typing import Dict, List

# installed libs
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn import cluster
from sklearn.decomposition import PCA
import yaml

# project libs
from src.data_utils.load_data import load_semantic_sim_data


def main(cfg):

    data, task_to_categories = load_semantic_sim_data(cfg["data_path"])

    if cfg.get("load_embeddings", None):
        embeddings_dict = load_json(cfg["embedding_path"])
    else:
        model = load_semantic_sim_model()
        embeddings_dict = calc_embeddings(model, data)
        # print(list(embeddings_dict.items())[:3])
        save_json(embeddings_dict, cfg["embedding_path"])

    if cfg["embedding_aggregation"] == "mean":
        category_mean_embed = average_embeddings_by_category(embeddings_dict, task_to_categories)

    for n_clusters in [5, 7, 10, 12, 15, 20]:  # [cfg["num_clusters"]]:  #
        cfg["num_clusters"] = n_clusters

        clustering = get_embedding_clusters(
            category_mean_embed, cfg["num_clusters"], cfg["distance_threshold"]
        )
        labels = clustering.labels_
        embeddings_pca = get_pca_embeddings(category_mean_embed)
        run_label = f'{cfg["clustering_method"]}_{cfg["num_clusters"]}_{cfg["distance_threshold"]}'
        save_pca_plot(embeddings_pca, labels, run_label)
        cluster_to_category = get_cluster_to_category(category_mean_embed, labels)
        save_json(cluster_to_category, f"clusters/clusters_{run_label}.json")


def load_config(config_path: str) -> dict:
    with open(config_path, "r") as fid:
        return yaml.load(fid, Loader=yaml.CLoader)


def load_semantic_sim_model():
    return SentenceTransformer("all-MiniLM-L6-v2")


def calc_embeddings(model, data: Dict[str, str]) -> Dict[str, List[float]]:
    embeddings = model.encode(list(data.values()))
    return {id: embed.tolist() for id, embed in zip(data.keys(), embeddings)}


def save_json(object, save_path):
    with open(save_path, "w") as fid:
        json.dump(object, fid)


def load_json(load_path: str):
    with open(load_path, "r") as fid:
        return json.load(fid)


def average_embeddings_by_category(embeddings, task_to_categories) -> Dict[str, np.ndarray]:
    category_to_tasks = get_category_to_tasks(task_to_categories)
    category_embedings = get_category_embeddings(category_to_tasks, embeddings)
    mean_cat_embed = calc_mean_embeddings(category_embedings)
    return mean_cat_embed


def get_category_to_tasks(task_to_categories: Dict[str, List[str]]) -> Dict[str, List[str]]:
    cat_to_tasks = defaultdict(list)
    for task, categories in task_to_categories.items():
        for cat in categories:
            cat_to_tasks[cat].append(task)
    return cat_to_tasks


def get_category_embeddings(category_to_tasks, embeddings) -> Dict[str, List[List[float]]]:
    category_embeddings = {}
    for cat, tasks in category_to_tasks.items():
        category_embeddings[cat] = [embeddings[task] for task in tasks]
    return category_embeddings


def calc_mean_embeddings(category_embedings: Dict[str, List[List[str]]]) -> Dict[str, np.ndarray]:
    cat_mean_embed = {}
    for cat, embeddings in category_embedings.items():
        embeddings = np.asarray(embeddings, dtype=np.float32)
        mean_embed = embeddings.mean(axis=0)
        assert mean_embed.shape[0] == embeddings[0].shape[0]
        cat_mean_embed[cat] = mean_embed
    return cat_mean_embed


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


def calc_cosine_similarity(embeddings):
    pass


if __name__ == "__main__":
    # argp = argparse.ArgumentParser()
    # argp.add_argument("config_path", help="path to config file")
    # args = argp.parse_args()
    # config_path = args.config_path
    config_path = "/Users/dustin/Documents/School/1_stanford/classes/cs224n/project/cs224n-project/src/semantic_similarity/semantic-sim-config.yml"
    cfg = load_config(config_path)
    main(cfg)
