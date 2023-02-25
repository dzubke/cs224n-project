# standard libs
import argparse
import json
from typing import Dict, List

# installed libs
from sentence_transformers import SentenceTransformer
import yaml

# project libs
from src.data_utils.load_data import load_semantic_sim_data

def main(config_path):

    cfg = load_config(config_path)

    model = load_semantic_sim_model()

    data = load_semantic_sim_data(cfg['data_path'])

    if cfg.get("load_embeddings", None):
        embeddings_dict = load_embeddings(cfg["embedding_path"])
    else:
        embeddings_dict = calc_embeddings(model, data)
        print(list(embeddings_dict.items())[:3])
        save_embeddings(embeddings_dict, cfg['embedding_path'])

    if cfg["embedding_aggregation"] == "mean":
        embeddings_dict = average_embeddings_by_category(embeddings_dict)

    cos_sims = calc_cosine_similarity(embeddings_dict)


def load_config(config_path: str) -> dict:
    with open(config_path, "r") as fid:
        return yaml.load(fid, Loader=yaml.CLoader)


def load_semantic_sim_model():
    return SentenceTransformer("all-MiniLM-L6-v2")


def calc_embeddings(model, data: Dict[str, str]) -> Dict[str, List[float]]:
    embeddings = model.encode(list(data.values()))
    return {
        id: embed.tolist() for id, embed in zip(data.keys(), embeddings)
    }


def save_embeddings(embeddings, save_path):
    with open(save_path, 'w') as fid:
        json.dump(embeddings, fid)


def load_embeddings(load_path: str):
    with open(load_path, 'r') as fid:
        return json.load(fid)


def average_embeddings_by_category(embeddings):
    pass


def calc_cosine_similarity(embeddings):
    pass


if __name__ == "__main__":
    argp = argparse.ArgumentParser()
    argp.add_argument("config_path", help="path to config file")
    args = argp.parse_args()
    main(args.config_path)
