


def calc_semantic_sim_score(config):

    cfg = config

    model = load_semantic_sim_model()

    data = load_data(cfg['data_path'])


    if cfg.get('load_embeddings', None):
        embeddings = load_embeddings(cfg['embedding_path'])
    else: 
        embeddings = calc_embeddings(model, data)
        save_embeddings(embeddings, cfg['embedding_path'])


    if cfg['embedding_aggregation'] == 'mean':
        embeddings = average_embeddings_by_category(embeddings)

    cos_similarities = calc_cosine_similarity(embeddings)

def load_semantic_sim_model():
    pass


def calc_embeddings(model, data):
    pass

def save_embeddings(embeddings, save_path):
    pass

def load_embeddings(embedding_path: str):
    pass

def average_embeddings_by_category(embeddings):
    pass

def calc_cosine_similarity(embeddings):
    pass