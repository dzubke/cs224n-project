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
from yellowbrick.cluster import KElbowVisualizer

# project libs
from src.data_utils.load_data import load_semantic_sim_data
from src.semantic_similarity.cluster import (
    get_embedding_clusters,
    get_pca_embeddings,
    save_pca_plot,
    get_cluster_to_category,
)


def main(cfg):

    data, task_to_categories = load_semantic_sim_data(cfg["data_path"])

    if cfg.get("load_embeddings", None):
        embeddings_dict = load_json(cfg["embedding_path"])
        task_to_categories = load_json(cfg["task_to_categories_path"])
    else:
        model = load_semantic_sim_model(cfg["model_name"])
        embeddings_dict = calc_embeddings(model, data)
        # print(list(embeddings_dict.items())[:3])
        save_json(embeddings_dict, cfg["task_to_categories_path"])

    if cfg["embedding_aggregation"] == "mean":
        embeddings_dict = average_embeddings_by_category(embeddings_dict, task_to_categories)

    if cfg["sub_select_categories"]:
        selected_embeddings_dict = {
            cat: emb for cat, emb in embeddings_dict.items() if cat in cfg["selected_categories"]
        }

    # if cfg["aggregate_embeddings"]:
    #     embeddings_dict = average_embeddings_by_category(embeddings_dict, task_to_categories)
    # else:
    #     embeddings_dict = {key: np.asarray(embed) for key, embed in embeddings_dict.items()}

    # if cfg["sub_select_embeddings"]:
    #     selected_embed_names = select_categories_by_instance_threshold(
    #         cfg["instance_count_path"], cfg["instance_count_threshold"]
    #     )
    # else:
    #     selected_embed_names = list(embeddings_dict)

    if cfg["use_elbow_method"]:
        mean_matrix = np.asarray(list(selected_embeddings_dict.values()))
        model = cluster.KMeans()
        visualizer = KElbowVisualizer(
            model, k=(2, 75), metric="distortion", timings=False
        )  # "distortion",  "silhouette", "calinski_harabasz"
        visualizer.fit(mean_matrix)  # Fit the data to the visualizer
        visualizer.show()  # Finalize and render the figure
    else:
        for num_clusters in cfg["num_clusters"]:
            clustering = get_embedding_clusters(
                selected_embeddings_dict, num_clusters, cfg["distance_threshold"]
            )
            labels = clustering.labels_
            embeddings_pca = get_pca_embeddings(selected_embeddings_dict)
            select_label = "sub-select" if cfg["sub_select_categories"] else "all-cats"
            run_label = (
                f'{cfg["model_name"]}_{cfg["clustering_method"]}_{num_clusters}_{select_label}'
            )
            save_pca_plot(embeddings_pca, labels, run_label)
            cluster_to_category = get_cluster_to_category(selected_embeddings_dict, labels)
            save_json(cluster_to_category, f"clusters/clusters_{run_label}.json")


def load_config(config_path: str) -> dict:
    with open(config_path, "r") as fid:
        return yaml.load(fid, Loader=yaml.CLoader)


def load_semantic_sim_model(model_name):
    return SentenceTransformer(model_name)


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
