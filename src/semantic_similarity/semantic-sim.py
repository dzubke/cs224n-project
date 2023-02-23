# standard libs
# installed libs
from sentence_transformers import SentenceTransformer
import yaml
# project libs
from src.load_data import load_task_data


def main(config_path):

    cfg = load_config(config_path)

    model = load_semantic_sim_model()

    data = load_task_data(cfg['data_path'])

    if cfg.get('load_embeddings', None):
        embeddings = load_embeddings(cfg['embedding_path'])
    else: 
        embeddings = calc_embeddings(model, data)
        print(embeddings)
        #save_embeddings(embeddings, cfg['embedding_path'])


    if cfg['embedding_aggregation'] == 'mean':
        embeddings = average_embeddings_by_category(embeddings)

    cos_sims = calc_cosine_similarity(embeddings)


def load_config(config_path: str)-> dict:
    with open(config_path, 'r') as fid:
        return yaml.load(fid, Loader=yaml.CLoader)
    
def load_semantic_sim_model():
    return SentenceTransformer('all-MiniLM-L6-v2')

def calc_embeddings(model, data):
    assert isinstance(data, list)
    assert isinstance(data[0], str)

    return model.encode(data)

def save_embeddings(embeddings, save_path):
    pass

def load_embeddings(embedding_path: str):
    pass

def average_embeddings_by_category(embeddings):
    pass

def calc_cosine_similarity(embeddings):
    pass


if __name__ == "__main__":
    config_path = "./semantic-sim-config.yml"
    main(config_path)