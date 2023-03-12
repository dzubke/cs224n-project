# standard libs
import argparse
from collections import defaultdict
import json
from typing import Dict, List

# installed libs
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
from scipy import spatial, special
import yaml

# project libs
from src.data_utils.load_data import load_semantic_sim_data


def main(cfg):

    if cfg.get("load_embeddings", None):
        embeddings_dict = load_json(cfg["embedding_path"])
        task_to_categories = load_json(cfg["task_to_categories_path"])
    else:
        data, task_to_categories = load_semantic_sim_data(cfg["data_path"])
        model = load_semantic_sim_model(cfg["model_name"])
        embeddings_dict = calc_embeddings(model, data)
        # print(list(embeddings_dict.items())[:3])
        save_json(embeddings_dict, cfg["embedding_path"])
        save_json(task_to_categories, cfg["task_to_categories_path"])

    if cfg["embedding_aggregation"] == "mean":
        category_mean_embed = average_embeddings_by_category(embeddings_dict, task_to_categories)

    count_threshold = 40_000
    count_path = "/Users/dustin/Documents/School/1_stanford/classes/cs224n/project/cs224n-project/src/data_utils/split_instance_count_simple.json"
    selected_cats = select_categories_by_instance_threshold(count_path, count_threshold)
    selected_cat_embeds = {
        cat: emb for cat, emb in category_mean_embed.items() if cat in selected_cats
    }
    sim_matrix, categories = calc_similarity_matrix(
        selected_cat_embeds, score_type=cfg["similarity_score_type"]
    )

    pd.DataFrame(sim_matrix, index=categories).to_csv(f"{cfg['similarity_score_type']}_matrix.csv_")

    selected_cats = get_seed_categories(sim_matrix, categories, min_or_max=cfg["min_or_max"])

    selected_cats = approx_select_cats(
        selected_cats, selected_cat_embeds, num_selected=10, min_or_max=cfg["min_or_max"]
    )

    print(f"all cats: {categories}")
    print(f"selected cats: {selected_cats}")

    cat_to_ind = {cat: i for i, cat in enumerate(categories)}
    selected_cat_indices = [cat_to_ind[cat] for cat in selected_cats]

    normalized_sim_score = calc_normalized_sim_score(selected_cat_indices, sim_matrix)
    print(
        f"sim_score: {normalized_sim_score} for score_type: {cfg['similarity_score_type']} and min_max: {cfg['min_or_max']}"
    )


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


def select_categories_by_instance_threshold(count_path, count_threshold=40_000):
    with open(count_path) as fid:
        count = json.load(fid)

    selected_categories = []
    for cat, count in count["train"]["category_to_instance"].items():
        if count > count_threshold:
            selected_categories.append(cat)

    return selected_categories


def calc_similarity_matrix(embed_dict, score_type="similarity"):
    num_embeddings = len(embed_dict)
    cat_embed_list = list(embed_dict.items())
    sim_matrix = np.zeros((num_embeddings, num_embeddings))
    for i in range(num_embeddings):
        for j in range(i + 1, num_embeddings):
            sim_score = cosine_similarity(
                cat_embed_list[i][1], cat_embed_list[j][1], score_type=score_type
            )
            sim_matrix[i][j] = sim_score
            sim_matrix[j][i] = sim_score

    categories = [x[0] for x in cat_embed_list]
    return sim_matrix, categories


def cosine_similarity(embed_1, embed_2, score_type="similarity"):
    if score_type == "similarity":
        score = 1 - spatial.distance.cosine(embed_1, embed_2)
    elif score_type == "difference":
        score = spatial.distance.cosine(embed_1, embed_2)
    else:
        raise ValueError(f"score_type: {score_type} must be in ['similarity', 'difference']")
    return score


def get_seed_categories(sim_matrix, categories, min_or_max="max"):
    if min_or_max == "max":
        max_min_value = sim_matrix.argmax()
    else:
        copy_sim_matrix = np.copy(sim_matrix)
        np.fill_diagonal(copy_sim_matrix, float("inf"))
        max_min_value = copy_sim_matrix.argmin()

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


def compute_total_sim_score(cats, sim_matrix):
    num_cats = len(cats)
    total_sim_score = 0
    for i in range(num_cats):
        for j in range(i + 1, num_cats):
            total_sim_score += sim_matrix[cats[i]][cats[j]]

    return total_sim_score


if __name__ == "__main__":
    # argp = argparse.ArgumentParser()
    # argp.add_argument("config_path", help="path to config file")
    # args = argp.parse_args()
    # config_path = args.config_path
    config_path = "/Users/dustin/Documents/School/1_stanford/classes/cs224n/project/cs224n-project/src/semantic_similarity/semantic-sim-config.yml"
    cfg = load_config(config_path)
    main(cfg)
