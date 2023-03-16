# standard libs
import argparse
from collections import defaultdict
import json
from typing import Dict, List

# installed libs
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn import cluster
from scipy import spatial, special
from yellowbrick.cluster import KElbowVisualizer
import yaml

# project libs
from src.data_utils.load_data import load_semantic_sim_data
from src.semantic_similarity.cluster import (
    get_embedding_clusters,
    get_pca_embeddings,
    save_pca_plot,
    get_cluster_to_category,
)


def main(cfg):

    if cfg.get("load_embeddings", None):
        embeddings_dict = load_json(cfg["embedding_path"])
        task_to_categories = load_json(cfg["task_to_categories_path"])
    else:
        data, task_to_categories, task_to_source = load_semantic_sim_data(cfg["data_path"])
        model = load_semantic_sim_model(cfg["model_name"])
        embeddings_dict = calc_embeddings(model, data)
        # print(list(embeddings_dict.items())[:3])
        save_json(embeddings_dict, cfg["embedding_path"])
        save_json(task_to_categories, cfg["task_to_categories_path"])
        save_json(task_to_source, cfg["task_to_source_path"])

    if cfg["embed_type"] == "category":
        embeddings_dict = average_embeddings_by_category(embeddings_dict, task_to_categories)
    else:
        embeddings_dict = {key: np.asarray(embed) for key, embed in embeddings_dict.items()}

    if cfg["filter_by_instance_count"]:
        selected_embed_names = filter_by_instance_threshold(
            cfg["instance_count_path"], cfg["embed_type"], cfg["instance_count_threshold"]
        )
    else:
        selected_embed_names = list(embeddings_dict)

    selected_embed_dict = {
        cat: emb for cat, emb in embeddings_dict.items() if cat in selected_embed_names
    }

    if cfg["grouping_method"] == "iterative":
        sim_matrix, embed_names = calc_similarity_matrix(selected_embed_dict)
        pd.DataFrame(sim_matrix, index=selected_embed_names, columns=selected_embed_names).to_csv(
            f"similarity-matrix_{len(selected_embed_names)}_{cfg['embed_type']}.csv"
        )

        grouped_embed_names = get_seed_categories(
            sim_matrix, embed_names, min_or_max=cfg["min_or_max"]
        )

        grouped_embed_names = approx_select_cats(
            grouped_embed_names,
            selected_embed_dict,
            num_selected=cfg["group_size"],
            min_or_max=cfg["min_or_max"],
        )

        name_to_ind = {name: i for i, name in enumerate(embed_names)}
        selected_name_indices = [name_to_ind[name] for name in grouped_embed_names]
        normalized_sim_score = calc_normalized_sim_score(selected_name_indices, sim_matrix)

        print(f"selected names: {grouped_embed_names}")
        print(f"sim_score: {round(normalized_sim_score, 4)} for min_max: {cfg['min_or_max']}")
        sim_diff_name = "sim" if cfg["min_or_max"] == "max" else "diff"
        group_filepath = (
            f"grouping/{sim_diff_name}_size-{cfg['group_size']}_{cfg['embed_type']}.json"
        )
        with open(group_filepath, "w") as fid:
            json.dump(grouped_embed_names, fid)
        with open(group_filepath.replace(".json", "_sim-score.txt"), "w") as fid:
            fid.write(str(round(normalized_sim_score, 4)))

    elif cfg["grouping_method"] == "clustering":
        if cfg["use_elbow_method"]:
            mean_matrix = np.asarray(list(selected_embed_dict.values()))
            model = cluster.KMeans()
            visualizer = KElbowVisualizer(
                model, k=(2, 75), metric="distortion", timings=False
            )  # "distortion",  "silhouette", "calinski_harabasz"
            visualizer.fit(mean_matrix)  # Fit the data to the visualizer
            visualizer.show()  # Finalize and render the figure
        else:
            for num_clusters in cfg["num_clusters"]:
                clustering = get_embedding_clusters(
                    selected_embed_dict, num_clusters, cfg["distance_threshold"]
                )
                labels = clustering.labels_
                embeddings_pca = get_pca_embeddings(selected_embed_dict)
                select_label = "sub-select" if cfg["sub_select_embeddings"] else "all"
                run_label = (
                    f'{cfg["model_name"]}_{cfg["clustering_method"]}_{num_clusters}_{select_label}'
                )
                cluster_to_category = get_cluster_to_category(selected_embed_dict, labels)
                save_json(cluster_to_category, f"clusters/clusters_{run_label}.json")
                save_pca_plot(embeddings_pca, labels, run_label)


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


def filter_by_instance_threshold(count_path, embed_type, count_threshold=40_000):
    assert embed_type in ["task", "category"]

    with open(count_path) as fid:
        count = json.load(fid)

    selected_categories = []
    inner_key = f"{embed_type}_to_instance"
    for cat, count in count["train"][inner_key].items():
        if count > count_threshold:
            selected_categories.append(cat)

    return selected_categories


def calc_similarity_matrix(embed_dict):
    num_embeddings = len(embed_dict)
    cat_embed_list = list(embed_dict.items())
    sim_matrix = np.zeros((num_embeddings, num_embeddings))
    for i in range(num_embeddings):
        for j in range(i, num_embeddings):
            sim_score = cosine_similarity(cat_embed_list[i][1], cat_embed_list[j][1])
            sim_matrix[i][j] = sim_score
            sim_matrix[j][i] = sim_score

    categories = [x[0] for x in cat_embed_list]
    return sim_matrix, categories


def cosine_similarity(embed_1, embed_2):
    """Similarity is 1 - distance = 1 - (1 - cos(x)) = cos(x)"""
    return abs(1 - spatial.distance.cosine(embed_1, embed_2))


def get_seed_categories(sim_matrix, categories, min_or_max="max"):

    copy_sim_matrix = np.copy(sim_matrix)
    # adding -/+ inf on the diagonals so the argmax/min functions works
    diagonal_fill_value = -float("inf") if min_or_max == "max" else float("inf")
    np.fill_diagonal(copy_sim_matrix, diagonal_fill_value)
    max_min_value = copy_sim_matrix.argmax() if min_or_max == "max" else copy_sim_matrix.argmin()

    indices = np.unravel_index(max_min_value, sim_matrix.shape)
    return categories[indices[0]], categories[indices[1]]


def approx_select_cats(selected_cats, selected_cat_embeds, num_selected=10, min_or_max="max"):
    running_mean_embed = (
        selected_cat_embeds[selected_cats[0]] + selected_cat_embeds[selected_cats[1]]
    ) / 2
    selected_cats = list(selected_cats)
    while len(selected_cats) < num_selected:
        similarities = []
        for cat, embed in selected_cat_embeds.items():
            if cat in selected_cats:
                sim = -float("inf") if min_or_max == "max" else float("inf")
            else:
                sim = cosine_similarity(running_mean_embed, embed)
            similarities.append(sim)

        iter_selected_cat_ind = (
            np.argmax(similarities) if min_or_max == "max" else np.argmin(similarities)
        )
        iter_selected_cat = list(selected_cat_embeds)[iter_selected_cat_ind]
        running_mean_embed = (
            running_mean_embed * len(selected_cats) + selected_cat_embeds[iter_selected_cat]
        ) / (len(selected_cats) + 1)
        selected_cats.append(iter_selected_cat)

    return selected_cats


def calc_normalized_sim_score(selected_cat_indices, sim_matrix):
    num_selected_cats = len(selected_cat_indices)
    total_sim_score = 0
    for i in range(num_selected_cats):
        for j in range(i + 1, num_selected_cats):
            total_sim_score += sim_matrix[selected_cat_indices[i]][selected_cat_indices[j]]

    num_nodes = len(selected_cat_indices)
    num_edges = special.comb(num_nodes, 2)
    normalized_sim_score = total_sim_score / num_edges

    return normalized_sim_score


def calc_norm_sim_score_from_csv(cats, sim_matrix_path):
    with open(sim_matrix_path) as fid:
        sim_matrix = pd.read_csv(fid, header=0, index_col=0)

    num_cats = len(cats)
    total_sim_score = 0
    for i in range(num_cats):
        for j in range(i + 1, num_cats):
            total_sim_score += sim_matrix[cats[i]][cats[j]]

    num_edges = special.comb(num_cats, 2)
    normalized_sim_score = total_sim_score / num_edges

    return normalized_sim_score


if __name__ == "__main__":
    # argp = argparse.ArgumentParser()
    # argp.add_argument("config_path", help="path to config file")
    # args = argp.parse_args()
    # config_path = args.config_path
    config_path = "/Users/dustin/Documents/School/1_stanford/classes/cs224n/project/cs224n-project/src/semantic_similarity/semantic-sim-config.yml"
    cfg = load_config(config_path)
    main(cfg)
